# Facade v2 : paste this ENTIRE file into ONE Kaggle Python cell.
# Attach facade-training-v2-calibrated.zip with Add Input first.
# Kaggle Settings: GPU accelerator ON and Internet ON.
# With only the v2 dataset attached, no code changes are required.

# ================= SETTINGS YOU CAN CHANGE =================
DATASET_ROOT = None       # Auto-detect. Optional: folder containing summary.json.
EPOCHS = 80               # Maximum epochs; early stopping may finish sooner.
PATIENCE = 10             # Stop after 10 epochs without validation improvement.
BATCH = 8                 # If GPU runs out of memory, change to 4, then 2.
IMAGE_SIZE = 640          # KEEP 640: dataset uses native-resolution 640px tiles.
SMOKE_TEST = False        # True = short pipeline check, not a finished model.
RUN_FINAL_TEST = False    # True only once model/settings are frozen.
EXPORT_ONNX = True        # Also export ONNX; best.pt survives export failure.
SHOW_PREDICTIONS = True   # Show six validation examples after training.
INSTALL_PACKAGES = True   # Installs the pinned training/export dependencies.
WORKING_DIR = "/kaggle/working"
# ===========================================================

import sys
import subprocess
if INSTALL_PACKAGES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "ultralytics==8.4.176", "onnx", "onnxruntime"])

"""Kaggle training helpers; no training starts on import.

Install ultralytics==8.4.176 in a Kaggle GPU notebook first.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ULTRALYTICS_VERSION='8.4.176'
MODEL_WEIGHTS='yolo26n-seg.pt'
NAMES=['crack','rust','spall']
DATASET_VERSIONS={'procedural-walls-v1','calibrated-walls-v2'}


def find_dataset(input_root='/kaggle/input',working_root='/kaggle/working',hint=None):
    if hint:
        path=Path(hint).resolve()
        if not (path/'summary.json').exists():raise ValueError('DATASET_ROOT must contain summary.json and manifest.json.')
        return path
    base=Path(input_root)
    found=[]
    for summary in base.rglob('summary.json'):
        try:
            if json.loads(summary.read_text())['version'] in DATASET_VERSIONS:found.append(summary.parent)
        except (KeyError,ValueError,OSError):continue
    if len(found)==1:return found[0]
    if len(found)>1:raise ValueError(f'Multiple datasets found; set DATASET_ROOT explicitly: {found}')
    archives=[]
    for path in base.rglob('*.zip'):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if not item.filename.endswith('/summary.json') or item.file_size>1_000_000:continue
                try:
                    if json.loads(archive.read(item))['version'] in DATASET_VERSIONS:
                        archives.append((path,item.filename));break
                except (ValueError,KeyError):continue
    if len(archives)!=1:raise ValueError(f'Attach one facade training dataset (v1 or v2). Matching archives: {archives}')
    archive_path,summary_name=archives[0]
    digest=hashlib.sha256()
    with archive_path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    target=(Path(working_root)/('facade-extracted-'+digest.hexdigest()[:12])).resolve()
    dataset_root=(target/summary_name).parent.resolve()
    if not dataset_root.is_relative_to(target):raise ValueError('Unsafe archive path.')
    if target.exists():
        if (dataset_root/'summary.json').exists():return dataset_root
        raise ValueError('Extraction folder exists but is incomplete; choose a fresh working folder.')
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path=(target/member.filename).resolve()
            if not path.is_relative_to(target):raise ValueError('Unsafe archive path.')
            if member.is_dir():path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as source,path.open('wb') as destination:shutil.copyfileobj(source,destination)
    return dataset_root


def check_dataset(root):
    import cv2
    import numpy as np
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    validation=json.loads((root/'validation.json').read_text())
    if not validation['passed']:raise ValueError('Dataset validation did not pass.')
    if summary.get('version')=='calibrated-walls-v2':
        physical=json.loads((root/'physical-validation.json').read_text())
        if not physical['passed']:raise ValueError('Physical geometry/camera checks did not pass.')
    manifest=json.loads((root/'manifest.json').read_text())
    by_stem={r['stem']:r for r in manifest};seen={};counts={}
    for split in ['train','val','test']:
        items=(root/f'{split}.txt').read_text().splitlines()
        if not items:raise ValueError('Empty split: '+split)
        counts[split]=len(items)
        for item in items:
            relative=Path(item.removeprefix('./'))
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe dataset list path.')
            path=root/relative;record=by_stem[path.stem]
            if record['split']!=split or not record['yolo_eligible']:raise ValueError('Split eligibility mismatch.')
            if seen.setdefault(record['scene_id'],split)!=split:raise ValueError('Wall scene leakage.')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('RGB checksum mismatch.')
            rgb=cv2.imread(str(path))
            if rgb is None or rgb.shape[:2]!=(summary['image_size'],summary['image_size']):raise ValueError('Bad image shape.')
            rows=(root/'labels'/f'{path.stem}.txt').read_text().splitlines()
            if len(rows)!=len(record['instances']):raise ValueError('Instance count mismatch.')
            labels=cv2.imread(str(root/'masks'/f'{path.stem}.png'),cv2.IMREAD_UNCHANGED)
            for row,obj in zip(rows,record['instances']):
                tokens=row.split();points=np.array(list(map(float,tokens[1:]))).reshape(-1,2)
                if int(tokens[0])!=obj['class_id'] or len(points)<3 or np.any((points<0)|(points>1)):raise ValueError('Bad segmentation polygon.')
                polygon=np.rint(points*np.array([labels.shape[1],labels.shape[0]])).astype(np.int32)
                mask=np.zeros(labels.shape,np.uint8);cv2.fillPoly(mask,[polygon],1)
                truth=labels==obj['instance_id'];prediction=mask>0
                iou=np.count_nonzero(truth&prediction)/max(1,np.count_nonzero(truth|prediction))
                if iou<.95:raise ValueError('Polygon disagrees with exact instance mask.')
    print('Dataset checks passed:',counts)
    return summary,counts


def prepare_yaml(root,working='/kaggle/working'):
    import yaml
    root=Path(root).resolve();working=Path(working);working.mkdir(exist_ok=True,parents=True)
    value={'path':str(root),'train':str(root/'train.txt'),'val':str(root/'val.txt'),
           'test':str(root/'test.txt'),'names':dict(enumerate(NAMES))}
    path=working/'facade-data.yaml';path.write_text(yaml.safe_dump(value,sort_keys=False))
    return path


def train(root,epochs=80,batch=8,imgsz=640,smoke=False,working='/kaggle/working',patience=20):
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__!=ULTRALYTICS_VERSION:raise ValueError(f'Use ultralytics=={ULTRALYTICS_VERSION} for this notebook.')
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU accelerator in Kaggle notebook settings before training.')
    if not isinstance(patience,int) or isinstance(patience,bool) or patience<1:raise ValueError('patience must be a positive integer.')
    summary,counts=check_dataset(root)
    data=prepare_yaml(root,working)
    model=YOLO(MODEL_WEIGHTS)
    result=model.train(data=str(data),epochs=2 if smoke else epochs,fraction=.1 if smoke else 1.0,
        imgsz=imgsz,batch=batch,device=0,workers=2,seed=20261019,deterministic=True,
        patience=patience,project=str(Path(working)/'facade-training'),name='smoke' if smoke else 'nano-seg',
        exist_ok=False,pretrained=True,cache=False,plots=True,save=True,
        mask_ratio=1,overlap_mask=False,mosaic=0.0,mixup=0.0,degrees=0.0,
        translate=.05,scale=.15,fliplr=.5,flipud=0.0,hsv_h=.01,hsv_s=.12,hsv_v=.15)
    best=Path(model.trainer.best)
    if not best.exists():raise RuntimeError('Training did not produce best.pt.')
    trained=YOLO(str(best));val=trained.val(data=str(data),split='val',imgsz=imgsz,device=0,plots=True)
    metadata={'model_family':MODEL_WEIGHTS,'task':'instance-segmentation','class_names':NAMES,
        'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'image_size':imgsz,'training_epochs_requested':2 if smoke else epochs,
        'epochs_completed':int(model.trainer.epoch)+1,'early_stopping_patience':patience,
        'parameter_count':sum(parameter.numel() for parameter in trained.model.parameters()),
        'manifest_sha256':hashlib.sha256((Path(root)/'manifest.json').read_bytes()).hexdigest(),
        'smoke_test':smoke,'dataset_summary':summary,'eligible_split_counts':counts,
        'validation_metrics':{k:float(v) for k,v in val.results_dict.items()},
        'training_run':str(best.parent.parent),'test_evaluated':False,
        'evaluation_scope':'Synthetic same-generator validation; not proof of AirSim or real-world defect counting.',
        'deployment_status':'Requires independent AirSim wall labels, identity evaluation and latency measurement before integration.'}
    output=Path(working)/('facade-model-smoke' if smoke else 'facade-model')
    output.mkdir(exist_ok=True)
    shutil.copy2(best,output/'best.pt')
    for filename in ['results.csv','results.png','args.yaml']:
        source=best.parent.parent/filename
        if source.exists():shutil.copy2(source,output/filename)
    (output/'model-metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Model artifacts:',output)
    return trained,data,output


def evaluate_final_test(model,data,output,imgsz=640):
    result=model.val(data=str(data),split='test',imgsz=imgsz,device=0,plots=True)
    path=Path(output)/'model-metadata.json';metadata=json.loads(path.read_text())
    metadata.update(test_evaluated=True,test_metrics={k:float(v) for k,v in result.results_dict.items()})
    path.write_text(json.dumps(metadata,indent=2))
    return result


def export_model(model,output,imgsz=640):
    # opset 17, fixed shape, no graph simplifier; preserve best.pt if export fails.
    exported=Path(model.export(format='onnx',imgsz=imgsz,opset=17,simplify=False,dynamic=False,device='cpu'))
    target=Path(output)/'model.onnx';shutil.copy2(exported,target)
    return target



def run_kaggle_v2():
    import torch
    from PIL import Image as PILImage
    try:
        from IPython.display import display, Image, FileLink
    except ImportError:
        display = None

    if IMAGE_SIZE != 640:
        raise ValueError("Keep IMAGE_SIZE=640 for this native-tile dataset.")
    if EPOCHS < 1 or BATCH < 1:
        raise ValueError("EPOCHS and BATCH must be positive integers.")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in Kaggle Settings, then restart the session.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    dataset = find_dataset(working_root=WORKING_DIR, hint=DATASET_ROOT)
    summary = json.loads((dataset / "summary.json").read_text())
    if summary.get("version") != "calibrated-walls-v2":
        raise ValueError("Attach facade-training-v2-calibrated.zip. This run requires v2.")
    print("Dataset:", dataset, flush=True)
    print("Eligible images:", summary["yolo_images"], flush=True)
    print("Checking every eligible image and label before training...", flush=True)
    if display and (dataset / "annotation-preview.jpg").exists():
        display(Image(filename=str(dataset / "annotation-preview.jpg")))
    trained_model, data_yaml, model_output = train(
        dataset, epochs=EPOCHS, batch=BATCH, imgsz=IMAGE_SIZE,
        smoke=SMOKE_TEST, working=WORKING_DIR, patience=PATIENCE)
    metadata = json.loads((model_output / "model-metadata.json").read_text())
    print("Validation metrics:", json.dumps(metadata["validation_metrics"], indent=2))
    if display and (model_output / "results.png").exists():
        display(Image(filename=str(model_output / "results.png")))
    if SHOW_PREDICTIONS and display:
        for relative in (dataset / "val.txt").read_text().splitlines()[:6]:
            result = trained_model.predict(str(dataset / relative.removeprefix("./")),
                                           imgsz=IMAGE_SIZE, conf=0.25, device=0, verbose=False)[0]
            display(PILImage.fromarray(result.plot()[:, :, ::-1]))
    if RUN_FINAL_TEST:
        if SMOKE_TEST:
            print("Reserved test skipped during smoke check.")
        else:
            final_metrics = evaluate_final_test(trained_model, data_yaml, model_output, IMAGE_SIZE)
            print("Reserved test:", final_metrics.results_dict)
    else:
        print("Reserved test skipped; use it after model/settings are frozen.")
    if EXPORT_ONNX:
        try:
            print("ONNX:", export_model(trained_model, model_output, IMAGE_SIZE))
        except Exception as error:
            print("ONNX export failed; best.pt is preserved:", repr(error))
    archive = shutil.make_archive(str(model_output), "zip", root_dir=model_output)
    print("Download this model archive from Kaggle Output:", archive)
    if display:
        display(FileLink(archive))
    print("Synthetic validation does not establish real-camera accuracy or unique crack counting.")
    return archive


if __name__ == "__main__":
    MODEL_ARCHIVE = run_kaggle_v2()
