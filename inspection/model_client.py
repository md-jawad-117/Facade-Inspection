"""Flight-side adapter for the local segmentation worker; no silent fallback."""
import base64
import json
import hashlib
from functools import lru_cache
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import cv2
from .model_masks import clip_findings,restore_findings

ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:8766'
LOCK=threading.Lock()


def health():
    try:
        with urllib.request.urlopen(URL+'/health',timeout=2) as response:return json.load(response)
    except (OSError,ValueError):return None


@lru_cache(maxsize=4)
def expected_sha(modified,size):
    return hashlib.sha256((ROOT/"weights/best.pt").read_bytes()).hexdigest()


def ensure_worker():
    with LOCK:
        stamp=(ROOT/"weights/best.pt").stat()
        expected=expected_sha(stamp.st_mtime_ns,stamp.st_size)
        ready=health()
        if ready:
            if ready.get("workspace")!=str(ROOT):raise RuntimeError("Port 8766 belongs to another project. Close its model worker before using this copy.")
            if ready.get("sha256")!=expected:raise RuntimeError("A different model is running on port 8766; restart the model worker.")
            return ready
        python=ROOT/'.venv-model/Scripts/python.exe'
        if not python.exists():raise RuntimeError('Model runtime missing. Install requirements-model.txt in .venv-model.')
        folder=ROOT/'.state';folder.mkdir(exist_ok=True)
        with (folder/'model-worker.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen([str(python),'-u','-m','inspection.model_server'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            ready=health()
            if ready:
                if ready.get("workspace")!=str(ROOT):raise RuntimeError("Port 8766 belongs to another project.")
                if ready.get("sha256")!=expected:raise RuntimeError("Model hash mismatch.")
                return ready
            if process.poll() is not None:break
            time.sleep(.3)
        raise RuntimeError('Model worker did not start. See .state/model-worker.log; baseline was not substituted.')


def predict(image,sensitivity='balanced'):
    thresholds={'conservative':.45,'balanced':.25,'sensitive':.12}
    if sensitivity not in thresholds:raise ValueError('Unknown sensitivity')
    ensure_worker()
    ok,data=cv2.imencode('.png',image)
    if not ok:raise ValueError('Could not encode model input')
    payload=json.dumps({'image':base64.b64encode(data).decode(),'confidence':thresholds[sensitivity]}).encode()
    token=(ROOT/'.state/model-token').read_text().strip()
    request=urllib.request.Request(URL+'/infer',data=payload,headers={'Content-Type':'application/json','X-Model-Token':token})
    try:
        with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
    except urllib.error.HTTPError as error:
        try:message=json.load(error).get('error','Model inference failed')
        except ValueError:message='Model inference failed'
        raise RuntimeError(message) from error


def detect(image,roi=None,sensitivity='balanced',input_scale=1.0):
    if not 0<input_scale<=1:raise ValueError('Model input scale must be in (0,1].')
    height,width=image.shape[:2]
    model_image=image if input_scale==1 else cv2.resize(image,(max(1,round(width*input_scale)),max(1,round(height*input_scale))),interpolation=cv2.INTER_AREA)
    result=predict(model_image,sensitivity)
    findings=restore_findings(result['findings'],model_image.shape,image.shape)
    findings=clip_findings(findings,roi)
    for feature in findings:feature['model_sha256']=result['model']['sha256']
    return findings
