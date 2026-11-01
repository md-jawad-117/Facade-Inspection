"""Start the locally installed simulator with project camera settings."""
import json
import os
import subprocess
import time
from pathlib import Path
from .rpc import AirSimRPC

ROOT=Path(__file__).resolve().parents[1]
EXE=Path(os.environ.get('AIRSIM_EXE',str(Path.home()/'Downloads/AirSim/Blocks/Blocks/WindowsNoEditor/Blocks.exe'))).expanduser()


def ready():
    try:
        with AirSimRPC(timeout=1) as c:
            return bool(c.call('ping'))
    except (OSError,RuntimeError):
        return False


def ensure_simulator(config,log=print):
    running=ready()
    if running:
        with AirSimRPC() as c:
            if c.call('getMultirotorState','')['landed_state'] != 0:
                raise RuntimeError('The simulator drone must be landed before starting a new mission.')
            frame=c.call('simGetImages',[{'camera_name':'0','image_type':0,'pixels_as_float':False,'compress':True}],'',False)[0]
            if (frame['width'],frame['height'])!=(config['image_width_px'],config['image_height_px']):
                log('Restarting landed simulator to apply inspection camera resolution.')
                try:
                    c.call('simRunConsoleCommand','quit')
                except ConnectionError:
                    pass
                running=False
                time.sleep(2)
    if not running:
        if not EXE.exists():
            raise RuntimeError(f'AirSim executable not found: {EXE}')
        settings={'SettingsVersion':1.2,'SimMode':'Multirotor','ApiServerPort':41451,
                  'CameraDefaults':{'CaptureSettings':[{'ImageType':kind,'Width':config['image_width_px'],
                    'Height':config['image_height_px'],'FOV_Degrees':config['horizontal_fov_deg']} for kind in [0,1]]}}
        settings['Vehicles']={'SimpleFlight':{'VehicleType':'SimpleFlight','Cameras':{
            'nav':{'X':.46,'Y':0,'Z':0,'CaptureSettings':[{'ImageType':1,'Width':320,'Height':180,'FOV_Degrees':90}]}}}}
        path=ROOT/'configs/airsim.json'
        path.write_text(json.dumps(settings,indent=2),encoding='ascii')
        log('Starting AirSim with inspection cameras...')
        subprocess.Popen([str(EXE),f'-settings={path}','-windowed','-ResX=1280','-ResY=720'],cwd=EXE.parent)
        deadline=time.monotonic()+60
        while not ready():
            if time.monotonic()>deadline:
                raise RuntimeError('AirSim did not start within 60 seconds.')
            time.sleep(1)
    with AirSimRPC() as c:
        c.call('reset')
    time.sleep(1)
