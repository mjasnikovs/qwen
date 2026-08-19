/**
 * A/B replay harness for the pi-task `refine` phase.
 *
 * Feeds ONE captured refine request (system + user, tools=[read]) to an
 * OpenAI-compatible endpoint and runs the agent loop, executing `read` against a
 * real workspace, until the model answers with plain text or hits MAX_CALLS.
 *
 * Two independent knobs, so the four cells of the A/B are directly comparable:
 *   - which model llama-server has loaded (the arm)
 *   - GUARD=1: block a re-read of a path already read this run, returning the
 *     SAME text pi-task's single-read-guard returns. This is the candidate fix.
 *
 * Everything else — prompt, tool schema, workspace, sampler settings — is
 * byte-identical across cells.
 *
 * Quality check (regression guard): refine's contract is four exact headings.
 * A fix that stops the loop but breaks the deliverable is not a fix, so every
 * trial records which of the four headings survived.
 *
 * Output: one JSON line per trial on stdout.
 */
import fs from 'node:fs'
import path from 'node:path'

const STIMULUS = process.env.STIMULUS
const ROOT = process.env.ROOT || '/home/edgars/hub/mx5-n'
const BASE = process.env.BASE || 'http://127.0.0.1:8080/v1'
const MAX_CALLS = Number(process.env.MAX_CALLS || 200)
const TRIALS = Number(process.env.TRIALS || 5)
const ARM = process.env.ARM || 'unknown'
const GUARD = process.env.GUARD === '1'
// Candidate fix: a per-phase READ BUDGET. After BUDGET successful reads, every
// further read returns "budget spent, answer now" instead of file contents.
// Unlike the single-read guard this catches breadth, not repetition — which is
// what 3.8 actually does (143 distinct paths in 150 calls).
const BUDGET = Number(process.env.BUDGET || 0)
// Hard stop: after DEGRADE_AT tool calls, drop the tools entirely and re-ask.
// A no-tools turn cannot call anything, so this ALWAYS terminates — which a
// budget nudge does not (a model is free to ignore its own tool output forever).
// This models pi-task's existing runDegradedFinalAttempt, fired by a wall clock
// instead of only by a LoopDetector hit.
const DEGRADE_AT = Number(process.env.DEGRADE_AT || 0)
// ---- CONFIGURATION knobs (no harness logic, pure model settings) ----------
// THINK=1 re-opens the thinking channel. The server runs `--reasoning off`,
// which makes the template PREFILL an empty, pre-closed `<think></think>` --
// the model is structurally unable to plan before acting. Qwen3.8 ships
// thinking ON by default; its agentic planning is trained to live in there.
const THINK = process.env.THINK === '1'
const EFFORT = process.env.EFFORT || ''
// Sampler overrides. Sent per request only when set, so an unset knob keeps
// the server default and the cell stays comparable.
const num = k => (process.env[k] === undefined ? undefined : Number(process.env[k]))
const SAMPLERS = {
    temperature: num('TEMP'),
    top_p: num('TOP_P'),
    top_k: num('TOP_K'),
    min_p: num('MIN_P'),
    presence_penalty: num('PP'),
    repetition_penalty: num('RP'),
}
for (const k of Object.keys(SAMPLERS)) if (SAMPLERS[k] === undefined) delete SAMPLERS[k]
const TEMPLATE_KWARGS = THINK
    ? {chat_template_kwargs: {enable_thinking: true, ...(EFFORT ? {reasoning_effort: EFFORT} : {})}}
    : {}
/** With reasoning_format=none the trace comes back INLINE in content. */
function splitThink(s) {
    const m = String(s || '').match(/^([\s\S]*?)<\/think>/)
    if (!m) return {think: '', text: String(s || '')}
    return {think: m[1].replace(/^\s*<think>/, ''), text: String(s).slice(m[0].length)}
}
const DEGRADE_HINT =
    '[SYSTEM NOTE: Your previous attempt ran out of budget before answering — you were still '
    + 'reading source material. You have no tools now. Write the four-section output IMMEDIATELY '
    + 'from what you have already gathered. Put anything you could not confirm under '
    + 'KNOWN-UNKNOWNS.]'

const base = JSON.parse(fs.readFileSync(STIMULUS, 'utf8'))
// pi-task appends `/no_think` to the refine prompt (src/task/prompts.ts,
// appendNoThink). That is a THIRD thinking kill-switch on top of the server's
// `--reasoning off` and models.json `supportsReasoningEffort: false`. Removing it
// is the pi-task half of the candidate fix, so it has to be its own knob.
if (process.env.STRIP_NOTHINK === '1') {
    const strip = s => String(s).replace(/\s*\/no_think\s*$/, '')
    for (const m of base.messages) {
        if (typeof m.content === 'string') m.content = strip(m.content)
        else if (Array.isArray(m.content)) {
            for (const part of m.content) if (part?.text) part.text = strip(part.text)
        }
    }
}
const SECTIONS = ['GOAL', 'CONSTRAINTS', 'KNOWN-UNKNOWNS', 'EXTERNAL-DEPENDENCIES']

/** Verbatim from src/workers/single-read-guard.ts — singleReadReason(). */
function singleReadReason(p) {
    return (
        `You already read ${p} earlier in this run — its contents are in your context. `
        + `Re-reading the same file is blocked. Do not read it again: use what you have already `
        + `gathered and write your final answer now.`
    )
}

/** The text a budget-exhausted read returns. Mirrors singleReadReason's shape. */
function budgetSpentReason(n) {
    return (
        `You have already made ${n} file reads in this run — the read budget for this phase is `
        + `spent. Further reads are blocked. Write your final answer NOW from what you have `
        + `already gathered; state any remaining gap under KNOWN-UNKNOWNS instead of reading more.`
    )
}

/** pi's `read`, close enough for this: path + optional offset/limit, ENOENT on miss. */
function doRead(args) {
    const rel = String(args.path || '').replace(/^\/workspace\/?/, '')
    const p = path.resolve(ROOT, rel)
    if (!p.startsWith(ROOT)) return {error: `EACCES: outside workspace, read '${args.path}'`}
    let st
    try {
        st = fs.statSync(p)
    } catch {
        return {error: `ENOENT: no such file or directory, access '${args.path}'`}
    }
    if (st.isDirectory()) return {error: 'EISDIR: illegal operation on a directory, read'}
    const lines = fs.readFileSync(p, 'utf8').split('\n')
    const off = Number.isFinite(args.offset) ? Math.max(0, args.offset - 1) : 0
    const lim = Number.isFinite(args.limit) ? args.limit : 2000
    return {text: lines.slice(off, off + lim).join('\n').slice(0, 50000)}
}

async function trial(n) {
    const messages = JSON.parse(JSON.stringify(base.messages))
    const calls = []
    const seenPaths = new Set()
    let enoent = 0
    let reads = 0
    let blocked = 0
    let finishedWith = null
    let answer = ''
    let degraded = false
    let thinkChars = 0
    const t0 = Date.now()
    for (let i = 0; i < MAX_CALLS; i++) {
        // Wall-clock equivalent: budget spent -> re-ask with NO tools, once.
        if (DEGRADE_AT > 0 && !degraded && calls.length >= DEGRADE_AT) {
            degraded = true
            const first = JSON.parse(JSON.stringify(base.messages))
            const u = first[first.length - 1]
            if (typeof u.content === 'string') u.content = `${DEGRADE_HINT}\n\n${u.content}`
            else u.content[0].text = `${DEGRADE_HINT}\n\n${u.content[0].text}`
            messages.length = 0
            messages.push(...first)
        }
        let j
        try {
            const res = await fetch(`${BASE}/chat/completions`, {
                method: 'POST',
                headers: {'content-type': 'application/json', authorization: 'Bearer local'},
                body: JSON.stringify({
                    model: base.model,
                    messages,
                    ...(degraded ? {} : {tools: base.tools}),
                    stream: false,
                    ...SAMPLERS,
                    ...TEMPLATE_KWARGS,
                }),
            })
            j = await res.json()
        } catch (e) {
            finishedWith = 'http-error:' + String(e).slice(0, 80)
            break
        }
        const m = j.choices?.[0]?.message
        if (!m) {
            finishedWith = 'no-message'
            break
        }
        const tcs = m.tool_calls || []
        // llama.cpp returns the trace in `reasoning_content` when the template
        // opens a think block; older/none formats leave it inline in content.
        const split = splitThink(m.content)
        thinkChars += split.think.length + (m.reasoning_content || '').length
        if (tcs.length === 0) {
            finishedWith = 'answered'
            // Section counting must see the DELIVERABLE, not the trace — a think
            // block that muses about GOAL would otherwise fake a passing grade.
            answer = split.text || (split.think ? '' : m.content || '')
            break
        }
        messages.push({role: 'assistant', content: m.content || '', tool_calls: tcs})
        for (const tc of tcs) {
            let args = {}
            try {
                args = JSON.parse(tc.function.arguments || '{}')
            } catch {}
            calls.push(`${tc.function.name}:${JSON.stringify(args)}`)
            let content
            if (tc.function.name !== 'read') {
                content = 'unknown tool'
            } else {
                const abs = path.resolve(ROOT, String(args.path || '').replace(/^\/workspace\/?/, ''))
                if (BUDGET > 0 && reads >= BUDGET) {
                    content = budgetSpentReason(reads)
                    blocked++
                } else if (GUARD && seenPaths.has(abs)) {
                    content = singleReadReason(String(args.path))
                    blocked++
                } else {
                    reads++
                    seenPaths.add(abs)
                    const r = doRead(args)
                    if (r.error) enoent++
                    content = r.error ?? r.text
                }
            }
            messages.push({role: 'tool', tool_call_id: tc.id, content})
        }
    }
    if (!finishedWith) finishedWith = 'hit-cap'
    const distinct = new Set(calls)
    // Smallest distance between two identical calls — pi-task's LoopDetector has
    // a 20-call window, so a min gap >= 20 means it can never fire.
    const first = new Map()
    let minGap = null
    calls.forEach((c, i) => {
        if (first.has(c)) {
            const g = i - first.get(c)
            if (minGap === null || g < minGap) minGap = g
        }
        first.set(c, i)
    })
    return {
        arm: ARM,
        guard: GUARD,
        trial: n,
        outcome: finishedWith,
        seconds: Math.round((Date.now() - t0) / 1000),
        toolCalls: calls.length,
        distinct: distinct.size,
        repeats: calls.length - distinct.size,
        enoent,
        blocked,
        budget: BUDGET,
        degradeAt: DEGRADE_AT,
        degraded,
        reads,
        minGap,
        answerChars: answer.length,
        sections: SECTIONS.filter(s => answer.includes(s)).length,
        think: THINK ? EFFORT || 'xhigh' : 'off',
        stripNoThink: process.env.STRIP_NOTHINK === '1',
        thinkChars,
        samplers: SAMPLERS,
    }
}

for (let n = 1; n <= TRIALS; n++) {
    const r = await trial(n)
    console.log(JSON.stringify(r))
}
