// Logging reverse proxy: 127.0.0.1:8081 -> 127.0.0.1:8080
// Writes one JSON line per completed request to LOG.
import http from 'node:http'
import fs from 'node:fs'

const LOG = process.env.PROXY_LOG || '/tmp/llama-proxy.jsonl'
const UP = {host: '127.0.0.1', port: 8080}
let seq = 0

const srv = http.createServer((req, res) => {
    const id = ++seq
    const t0 = Date.now()
    const chunks = []
    req.on('data', c => chunks.push(c))
    req.on('end', () => {
        const body = Buffer.concat(chunks)
        const p = http.request(
            {host: UP.host, port: UP.port, path: req.url, method: req.method, headers: req.headers},
            up => {
                res.writeHead(up.statusCode, up.headers)
                const out = []
                up.on('data', c => {
                    out.push(c)
                    res.write(c)
                })
                up.on('end', () => {
                    res.end()
                    let reqJson = null
                    try {
                        reqJson = JSON.parse(body.toString('utf8'))
                    } catch {}
                    const rec = {
                        id,
                        t: new Date(t0).toISOString(),
                        ms: Date.now() - t0,
                        url: req.url,
                        status: up.statusCode,
                        reqBytes: body.length,
                        resBytes: out.reduce((a, b) => a + b.length, 0),
                    }
                    if (reqJson) {
                        rec.model = reqJson.model
                        rec.stream = reqJson.stream
                        rec.nTools = (reqJson.tools || []).length
                        rec.msgs = (reqJson.messages || []).map(m => ({
                            role: m.role,
                            name: m.name,
                            tool_calls: (m.tool_calls || []).map(tc => ({
                                n: tc.function?.name,
                                a: (tc.function?.arguments || '').slice(0, 300),
                            })),
                            text:
                                typeof m.content === 'string'
                                    ? m.content.slice(0, 400)
                                    : JSON.stringify(m.content || '').slice(0, 400),
                            len:
                                typeof m.content === 'string'
                                    ? m.content.length
                                    : JSON.stringify(m.content || '').length,
                        }))
                    }
                    rec.res = Buffer.concat(out).toString('utf8').slice(0, 4000)
                    fs.appendFileSync(LOG, JSON.stringify(rec) + '\n')
                })
            }
        )
        p.on('error', e => {
            try {
                res.writeHead(502)
                res.end(String(e))
            } catch {}
            fs.appendFileSync(
                LOG,
                JSON.stringify({id, t: new Date(t0).toISOString(), url: req.url, error: String(e)}) +
                    '\n'
            )
        })
        p.end(body)
    })
})
srv.listen(8081, '127.0.0.1', () => console.log('proxy on 8081 -> 8080, log ' + LOG))
