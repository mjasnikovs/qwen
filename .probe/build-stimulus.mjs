/**
 * Rebuild the exact /task-auto refine request for STEP 1.
 *
 * refine's prompt is `planContext + "\n\n---\n\n" + <bare refine body>`
 * (src/task/prompts.ts REFINE_PROMPT). We already captured the bare body from a
 * plain `/task <same title>` run, and the plan titles from the coverage prompt of
 * a live /task-auto run — so prepending buildScopeFence's output reproduces the
 * real request byte-for-byte without waiting out another 30-minute plan phase.
 *
 * The fence text below is copied verbatim from
 * src/task/auto-orchestrator.ts buildScopeFence().
 */
import fs from 'node:fs'

const S = process.env.S
const bare = JSON.parse(fs.readFileSync(`${S}/refine-stimulus.json`, 'utf8'))
const src = fs.readFileSync(`${S}/titles-src.txt`, 'utf8')

const listPart = src.slice(src.indexOf('TASK LIST:') + 'TASK LIST:'.length)
const titles = []
for (const line of listPart.split('\n')) {
    const m = /^(\d+)\.\s+(.*)$/.exec(line.trim())
    if (!m) {
        if (titles.length && line.trim() === '') break
        continue
    }
    if (Number(m[1]) !== titles.length + 1) break
    titles.push(m[2])
}

function buildScopeFence(ts, currentIndex) {
    const n = ts.length
    const listing = ts
        .map((t, i) => {
            const head = t.split(' | ')[0].trim()
            const tag = i === currentIndex ? ' (THIS STEP)' : ''
            return `[${i + 1}]${tag} ${head}`
        })
        .join('\n')
    return (
        `PLAN CONTEXT — this task is STEP ${currentIndex + 1} of ${n} in an already-decomposed plan. `
        + `Each step below is implemented by its OWN separate run; the others are NOT your job and `
        + `are done in later runs. Implement ONLY the slice named in "Task" below.\n\n`
        + `The design/spec document the task references describes the WHOLE system across all ${n} `
        + `steps. Read it to get exact names, types, and signatures for YOUR step and to understand `
        + `how your step fits — but DO NOT design, scaffold, schema, route, page, query, component, or `
        + `test anything that belongs to another step listed below. Your GOAL / CONSTRAINTS / `
        + `KNOWN-UNKNOWNS must cover only THIS step's slice. Do not pull in tables, endpoints, pages, `
        + `components, or flows owned by a later step.\n\n`
        + `The full plan (these run separately — do NOT implement them here):\n${listing}`
    )
}

const fence = buildScopeFence(titles, 0)
const out = JSON.parse(JSON.stringify(bare))
const user = out.messages[out.messages.length - 1]
if (typeof user.content === 'string') {
    user.content = `${fence}\n\n---\n\n${user.content}`
} else {
    user.content[0].text = `${fence}\n\n---\n\n${user.content[0].text}`
}
fs.writeFileSync(`${S}/refine-fenced.json`, JSON.stringify(out))
console.log(`titles=${titles.length} fenceChars=${fence.length} wrote refine-fenced.json`)
