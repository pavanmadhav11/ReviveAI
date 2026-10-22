import {spawn} from 'node:child_process';

const python = spawn('python', ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '8000'], {
  cwd: '/app/dip-service',
  stdio: 'inherit',
});
const api = spawn('node', ['src/server.js'], {
  cwd: '/app/server',
  stdio: 'inherit',
});

let shuttingDown = false;
function stop(signal = 'SIGTERM') {
  if (shuttingDown) return;
  shuttingDown = true;
  python.kill(signal);
  api.kill(signal);
}

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => stop(signal));
python.on('error', error => { console.error('Could not start the DIP service:', error); stop(); process.exitCode = 1; });
api.on('error', error => { console.error('Could not start the API:', error); stop(); process.exitCode = 1; });
api.on('exit', code => { stop(); process.exitCode = code ?? 1; });
python.on('exit', code => {
  if (!shuttingDown) {
    console.error(`DIP service exited unexpectedly (${code}).`);
    stop();
    process.exitCode = code ?? 1;
  }
});
