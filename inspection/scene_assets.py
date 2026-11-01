"""Generate only the final deterministic four-defect-per-wall scene."""
import json
from pathlib import Path
import cv2
import numpy as np
from .room import make_room
from .scene_geometry import crack_geometry, rasterize_crack, stretch_crack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/construction-room/scene'


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []
    ppm = 2048 / 6
    retained = {1, 7, 10, 16}
    for index, wall in enumerate(make_room()['walls']):
        width, height = wall['length'], wall['height']
        w, h = round(width * ppm), round(height * ppm)
        pitch = 1000 / ppm
        rng = np.random.default_rng(20261023090 + index)
        coarse = cv2.resize(rng.normal(0, 3, (max(2, round(height * 8)), round(width * 8))).astype(np.float32),
                            (w, h), interpolation=cv2.INTER_CUBIC)
        gray = np.clip(173 + coarse + rng.normal(0, .8, (h, w)), 0, 255).astype(np.uint8)
        background = np.stack([gray, np.clip(gray.astype(np.int16) + 1, 0, 255).astype(np.uint8),
                               np.clip(gray.astype(np.int16) + 2, 0, 255).astype(np.uint8)], axis=-1)
        image = background.copy()
        ids = np.zeros((h, w), np.uint16)
        objects = []
        # Preserve the validated random sequence and retained object identities.
        for fraction, up in zip([.16, .38, .62, .84], [1.6, 3.6, 2.6, 4.4]):
            for kind, du, dh in [(0, -.65, -.15), (0, .05, .15), (1, -.15, -.45), (2, .65, .1)]:
                identity = len(objects) + 1
                centre = np.array([(width * fraction + du) * 1000, (height - up - dh) * 1000])
                obj = dict(instance_id=identity, class_id=kind,
                           physical_id=f'SparseWall_{wall["id"]}-D{identity:03d}')
                if kind == 0:
                    points = np.array(crack_geometry(rng)['polyline_mm']) * 4 + centre
                    obj.update(stretch_crack(points, 20., 2.5))
                    region, mask, alpha = rasterize_crack(obj, pitch, w)
                    if np.any(ids[region][mask > 0]):
                        raise ValueError('Procedural defects overlap')
                    image[region] = np.clip(image[region].astype(float) * (1 - alpha[:, :, None])
                                            + rng.uniform(35, 60) * alpha[:, :, None], 0, 255).astype(np.uint8)
                    ids[region][mask > 0] = identity
                    obj['wall_polyline_u_height_m'] = [[float(x / 1000), float(height - y / 1000)]
                                                       for x, y in obj['polyline_mm']]
                else:
                    rx, ry = ((rng.uniform(65, 90), rng.uniform(130, 170)) if kind == 1
                              else (rng.uniform(110, 145), rng.uniform(100, 135)))
                    lo = np.floor((centre - [rx, ry]) / pitch).astype(int)
                    hi = np.ceil((centre + [rx, ry]) / pitch).astype(int)
                    yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
                    relative = (np.stack([xx * pitch, yy * pitch], axis=-1) - centre) / [rx, ry]
                    radius = np.linalg.norm(relative, axis=-1)
                    angle = np.arctan2(relative[:, :, 1], relative[:, :, 0])
                    mask = radius <= (1 if kind == 1 else .9 + .07 * np.sin(angle * 7) + .03 * np.sin(angle * 13))
                    region = (slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
                    if np.any(ids[region][mask]):
                        raise ValueError('Procedural defects overlap')
                    colour = np.array([45, 85, 140]) if kind == 1 else np.full(3, 92.)
                    image[region][mask] = np.clip(colour + rng.normal(0, 4, (int(mask.sum()), 1)), 0, 255).astype(np.uint8)
                    ids[region][mask] = identity
                    obj.update(center_mm=centre.tolist(), half_extent_mm=[float(rx), float(ry)])
                objects.append(obj)
        original_ids = ids.copy()
        for obj in objects:
            if obj['instance_id'] in retained:
                continue
            if obj['class_id'] == 0:
                region, _, alpha = rasterize_crack(obj, pitch, w)
                image[region][alpha > 0] = background[region][alpha > 0]
            else:
                erased = original_ids == obj['instance_id']
                image[erased] = background[erased]
            ids[ids == obj['instance_id']] = 0
        objects = [obj for obj in objects if obj['instance_id'] in retained]
        if sorted(obj['class_id'] for obj in objects) != [0, 0, 1, 2]:
            raise ValueError('Unexpected final scene composition')
        texture, mask_file = f'wall-{wall["id"]}.png', f'wall-{wall["id"]}-ids.png'
        cv2.imwrite(str(OUT / texture), image)
        cv2.imwrite(str(OUT / mask_file), ids)
        panels.append(dict(id=f'FacadeRoom_SparseWall_{wall["id"]}', wall_id=wall['id'],
                           u_center_m=width / 2, height_center_m=height / 2, size_m=[width, height],
                           source_size_px=[w, h], front_inward_m=.002, negative=False,
                           texture=texture, mask=mask_file, objects=objects))
    truth = dict(version='whole-wall-sparse-demo-v4-proposal', source_pitch_mm=1000 / ppm,
                 panels=panels, counts=dict(crack=8, rust=4, spall=4),
                 scope='Deterministic procedural simulation scene; reference masks are for evaluation only.')
    (OUT / 'truth.json').write_text(json.dumps(truth, indent=2), encoding='utf-8')
    print('Scene assets prepared: 4 walls, 16 defects. No simulator was started.')
    return truth
