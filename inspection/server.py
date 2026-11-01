"""Local worker interface for the current wall-inspection application."""
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .digital_twin import TwinStore
from .mission_scope import preview_selection, selection_from_values
from .rpc import AirSimRPC
from .single_wall_gate import require_passed

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.state'
STATE.mkdir(exist_ok=True)
PORT = 8765
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
JOB = dict(running=False, kind='room', phase='Ready', logs=[], run_id=None,
           exit_code=None, report_url=None, inspection_wall_ids=None)
SIM_CACHE = dict(time=0, value={'connected': False})


def simulator_status():
    now = time.monotonic()
    if now - SIM_CACHE['time'] < 3:
        return SIM_CACHE['value']
    try:
        with AirSimRPC(timeout=.8) as client:
            state = client.call('getMultirotorState', '')
            value = dict(connected=True, landed=state['landed_state'] == 0,
                         position=state['kinematics_estimated']['position'])
    except (OSError, RuntimeError):
        value = {'connected': False}
    SIM_CACHE.update(time=now, value=value)
    return value


def read_output(process, folder):
    with (folder / 'mission.log').open('w', encoding='utf-8') as log:
        for line in process.stdout:
            line = line.rstrip()
            log.write(line + '\n')
            log.flush()
            with LOCK:
                JOB['logs'] = (JOB['logs'] + [line])[-80:]
                if line.startswith('RUN_DIR '):
                    JOB['run_id'] = line[8:].strip()
                elif line and not line.startswith(('Traceback', '  File', '    ')):
                    JOB['phase'] = line
    code = process.wait()
    with LOCK:
        JOB.update(running=False, exit_code=code,
                   phase='Completed: report ready' if code == 0 else 'Stopped or failed. Review the saved log')
        if JOB['run_id']:
            run = Path(JOB['run_id']).name
            report = ROOT / 'assets/construction-room/flights' / run / 'worker-report.html'
            if report.exists():
                JOB['report_url'] = f'/assets/construction-room/flights/{run}/worker-report.html'
        (STATE / 'last-job.json').write_text(json.dumps(JOB, indent=2), encoding='utf-8')


def start_mission(values):
    with LOCK:
        if JOB['running']:
            raise ValueError('An inspection is already running.')
        status = simulator_status()
        if status.get('connected') and not status.get('landed'):
            raise ValueError('The drone must be landed before a mission.')
        if values.get('scene_profile', 'sparse-demo-v4') != 'sparse-demo-v4':
            raise ValueError('Only the current four-wall scene is supported.')
        if not values.get('full', True) or not values.get('wall_map', True):
            raise ValueError('This application uses complete selected-wall inspection.')
        if values.get('unknown_test') or values.get('adaptive_trial'):
            raise ValueError('Experiment modes are not part of this application.')
        walls = selection_from_values(values)
        adaptive = values.get('adaptive_reinspection', False)
        if not isinstance(adaptive, bool):
            raise ValueError('Invalid adaptive verification setting.')
        if not (ROOT / 'weights/best.pt').is_file():
            raise ValueError('Missing weights/best.pt.')
        if not (ROOT / '.venv-model/Scripts/python.exe').is_file():
            raise ValueError('Create the model environment described in README.md first.')
        if not (ROOT / 'assets/construction-room/scene/truth.json').is_file():
            raise ValueError('Run python run.py --prepare first.')
        if len(walls) > 1:
            require_passed(ROOT, 'sparse-demo-v4', adaptive_reinspection=adaptive)
        folder = STATE / ('mission-' + uuid.uuid4().hex)
        folder.mkdir()
        environment = os.environ.copy()
        environment['FACADE_STOP_FILE'] = str(folder / 'stop')
        command = [sys.executable, '-u', '-m', 'inspection.mission', '--execute', '--walls', *walls]
        if adaptive:
            command.append('--adaptive')
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', env=environment,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        JOB.update(running=True, phase='Preparing inspection...', logs=[], run_id=None,
                   exit_code=None, report_url=None, inspection_wall_ids=walls,
                   stop_file=str(folder / 'stop'))
        threading.Thread(target=read_output, args=(process, folder), daemon=True).start()
    return {'started': True}


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, code=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def allowed_host(self):
        return self.headers.get('Host') in [f'127.0.0.1:{PORT}', f'localhost:{PORT}']

    def do_GET(self):
        try:
            if not self.allowed_host():
                return self.reply({'error': 'Invalid host'}, 403)
            path = unquote(urlsplit(self.path).path)
            if path == '/api/state':
                with LOCK:
                    job = {k: v for k, v in JOB.items() if k != 'stop_file'}
                return self.reply(dict(token=TOKEN, job=job, simulator=simulator_status(), workspace=str(ROOT)))
            if path == '/api/twin':
                return self.reply(TwinStore(STATE / 'digital-twin.sqlite').snapshot())
            if path == '/':
                path = '/assets/inspection-system/index.html'
            relative = Path(path.lstrip('/'))
            if not relative.parts or relative.parts[0] != 'assets':
                return self.reply({'error': 'Not found'}, 404)
            file = (ROOT / relative).resolve()
            types = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                     '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
                     '.json': 'application/json', '.f32': 'application/octet-stream'}
            if ROOT not in file.parents or not file.is_file() or file.suffix.lower() not in types:
                return self.reply({'error': 'Not found'}, 404)
            data = file.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', types[file.suffix.lower()])
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (OSError, ValueError, RuntimeError) as error:
            self.reply({'error': str(error)}, 400)

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            if (not self.allowed_host() or self.headers.get('X-Inspection-Token') != TOKEN
                    or origin and origin not in [f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}']):
                return self.reply({'error': 'Invalid request token or origin'}, 403)
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 1024 * 1024:
                raise ValueError('Invalid request length.')
            values = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            if path == '/api/room-plan':
                walls = selection_from_values(values)
                value = preview_selection(walls)
                value['launch_blocked_reason'] = None
                if len(walls) > 1:
                    try:
                        require_passed(ROOT, 'sparse-demo-v4',
                                       adaptive_reinspection=bool(values.get('adaptive_reinspection', False)))
                    except ValueError:
                        value['launch_blocked_reason'] = 'A fresh South-wall verification is required first.'
                return self.reply(value)
            if path == '/api/room-mission':
                return self.reply(start_mission(values))
            if path == '/api/twin-review':
                TwinStore(STATE / 'digital-twin.sqlite').review(values['defect_id'], values['status'], values.get('note', ''))
                return self.reply({'saved': True})
            if path == '/api/stop':
                with LOCK:
                    if not JOB['running']:
                        raise ValueError('No active inspection.')
                    Path(JOB['stop_file']).touch()
                    JOB['phase'] = 'Stop requested. Landing will follow.'
                try:
                    with AirSimRPC(timeout=2) as client:
                        client.call('cancelLastTask', '')
                except (OSError, RuntimeError):
                    pass
                return self.reply({'stopping': True})
            self.reply({'error': 'Unknown action'}, 404)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            self.reply({'error': str(error)}, 400)

    def log_message(self, format, *args):
        if not self.path.startswith('/api/state'):
            super().log_message(format, *args)


if __name__ == '__main__':
    print(f'Inspection interface: http://127.0.0.1:{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
