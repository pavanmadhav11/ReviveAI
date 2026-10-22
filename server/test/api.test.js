import {test} from 'node:test';
import assert from 'node:assert/strict';
import app from '../src/app.js';
import http from 'node:http';
test('health endpoint returns API and DIP status fields',async()=>{
  const server=http.createServer(app); await new Promise(r=>server.listen(0,r));
  try { const response=await fetch(`http://127.0.0.1:${server.address().port}/api/health`); const body=await response.json(); assert.equal(body.status,'ok'); assert.ok('dipService' in body); }
  finally { server.close(); }
});
