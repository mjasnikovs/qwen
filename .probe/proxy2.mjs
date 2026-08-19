// Full-body capture proxy: 127.0.0.1:8081 -> 127.0.0.1:8080.
// Dumps every request body verbatim so a phase prompt can be replayed later.
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'

const DIR = process.env.CAP_DIR
fs.mkdirSync(DIR, {recursive: true})
let seq = 0

http.createServer((req, res) => {
    const id = ++seq
    const chunks = []
    req.on('data', c => chunks.push(c))
    req.on('end', () => {
        const body = Buffer.concat(chunks)
        if (body.length > 0) {
            fs.writeFileSync(path.join(DIR, `req-${String(id).padStart(4, '0')}.json`), body)
        }
        const p = http.request(
            {
                host: '127.0.0.1',
                port: 8080,
                path: req.url,
                method: req.method,
                headers: req.headers,
            },
            up => {
                res.writeHead(up.statusCode, up.headers)
                up.on('data', c => res.write(c))
                up.on('end', () => res.end())
            }
        )
        p.on('error', e => {
            try {
                res.writeHead(502)
                res.end(String(e))
            } catch {}
        })
        p.end(body)
    })
}).listen(8081, '127.0.0.1', () => console.log('capture proxy 8081 -> 8080, dir ' + DIR))
