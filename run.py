"""Application entry point. --prepare creates scene assets without starting AirSim."""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8765'


def app_ready():
    try:
        with urllib.request.urlopen(URL + '/api/state', timeout=2) as response:
            value = json.load(response)
        if value.get('workspace') != str(ROOT):
            raise SystemExit('Port 8765 belongs to another application instance. Stop it before opening this folder.')
        return True
    except (OSError, urllib.error.URLError):
        return False


def main():
    parser = argparse.ArgumentParser(description='Concrete-wall drone inspection')
    parser.add_argument('--prepare', action='store_true', help='Generate the required scene assets only; no simulation')
    args = parser.parse_args()
    if args.prepare:
        from inspection.scene_assets import generate
        generate()
        return
    if not app_ready():
        state = ROOT / '.state'
        state.mkdir(exist_ok=True)
        with (state / 'server.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'inspection.server'], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + 15
        while not app_ready():
            if process.poll() is not None or time.monotonic() > deadline:
                raise SystemExit('The interface did not start. Check .state/server.log.')
            time.sleep(.5)
    webbrowser.open(URL)


if __name__ == '__main__':
    main()
