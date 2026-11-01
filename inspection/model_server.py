"""Loopback-only CPU segmentation worker, isolated from the flight environment."""
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.state/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','False')
import torch
import ultralytics
from ultralytics import YOLO,settings
settings.update({"sync":False})
from inspection.duplicate_masks import suppress
from inspection.model_masks import NAMES,candidate,stitch,tile_origins

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state';STATE.mkdir(exist_ok=True)
TOKEN_PATH=STATE/'model-token'
if not TOKEN_PATH.exists():TOKEN_PATH.write_text(secrets.token_urlsafe(32))
TOKEN=TOKEN_PATH.read_text().strip()
WEIGHTS=ROOT/'weights/best.pt'
SHA=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
torch.set_num_threads(min(4,os.cpu_count() or 1))
MODEL=YOLO(str(WEIGHTS))
if MODEL.task!='segment' or [MODEL.names[i] for i in range(len(MODEL.names))]!=NAMES:
    raise RuntimeError('Checkpoint must be segment with classes crack, rust, spall in that order.')
META={'workspace':str(ROOT),'model':'YOLO26n-seg','sha256':SHA,'classes':NAMES,'task':MODEL.task,'device':'cpu',
      'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,
      'checkpoint_epoch_zero_based':MODEL.ckpt.get('epoch'),'parameter_count':sum(p.numel() for p in MODEL.model.parameters()),
      'mask_postprocess_version':'overlap-nms-v1','native_tile_size':640,'overlap_pixels':128,'trained_model':True,
      'scope':'Synthetic-trained instance masks; actual AirSim accuracy and physical unique counts unverified.'}
(STATE/'model-metadata.json').write_text(json.dumps(META,indent=2))


def infer(image,confidence=.25):
    height,width=image.shape[:2];items=[];tile_id=0;start=time.perf_counter()
    for y in tile_origins(height):
        for x in tile_origins(width):
            tile=image[y:y+640,x:x+640]
            result=MODEL.predict(tile,imgsz=640,conf=confidence,device='cpu',retina_masks=True,
                                 verbose=False,max_det=100,rect=False)[0]
            if result.masks is not None:
                for mask,cls,score in zip(result.masks.data.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                    if mask.shape!=tile.shape[:2]:raise RuntimeError('Model returned a mask at the wrong image scale.')
                    obj=candidate(mask>.5,NAMES[int(cls)],float(score),x,y,[x,y,x+tile.shape[1],y+tile.shape[0]],tile_id)
                    if obj:items.append(obj)
            tile_id+=1
    findings=suppress(stitch(items))
    return {'findings':findings,'model':META,'inference_seconds':time.perf_counter()-start,
            'tiles':tile_id,'confidence_threshold':confidence,'raw_instances':len(items),
            'scope':'Native tiles, conservative overlap stitching; unobserved fragments and ambiguous identities can remain separate.'}


class Handler(BaseHTTPRequestHandler):
    def reply(self,value,status=200):
        data=json.dumps(value).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        if self.path!='/health':return self.reply({'error':'Not found'},404)
        self.reply(META)

    def do_POST(self):
        if self.path!='/infer':return self.reply({'error':'Not found'},404)
        if not secrets.compare_digest(self.headers.get('X-Model-Token',''),TOKEN):return self.reply({'error':'Invalid token'},403)
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=64*1024*1024:raise ValueError('Image request exceeds 64 MB')
            value=json.loads(self.rfile.read(length));raw=base64.b64decode(value['image'],validate=True)
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[0]*image.shape[1]>12_000_000:raise ValueError('Invalid image or exceeds 12 megapixels')
            confidence=float(value.get('confidence',.25))
            if not .01<=confidence<=.95:raise ValueError('Invalid confidence threshold')
            self.reply(infer(image,confidence))
        except (ValueError,KeyError,RuntimeError,TypeError) as exc:self.reply({'error':str(exc)},400)

    def log_message(self,*args):pass


if __name__=='__main__':
    print(json.dumps(META),flush=True)
    HTTPServer(('127.0.0.1',8766),Handler).serve_forever()
