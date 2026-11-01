# Dataset creation reference

These scripts document and reproduce the final `calibrated-walls-v2` dataset. They are optional tools, separate from the inspection application. Generated datasets and archives are ignored by Git.

## What the code generates

1. **Concrete surfaces:** procedural 1.2 x 1.2 m wall patches, rasterized at 2048 x 2048 pixels, with varied surface texture, lighting-like intensity changes, joints, and clean negative examples.
2. **Defect geometry:** cracks, rust patches, and spalls. Crack traced length is 50 to 100 mm, nominal opening is 5 mm, and horizontal zigzag spread is 20 to 50 mm.
3. **Exact annotations:** each drawn defect receives an instance ID and class. Pixel masks are saved as PNGs; annotations are also exported as COCO RLE and YOLO segmentation polygons. No manual tracing or generative image model is involved.
4. **Camera views:** an ideal frontal pinhole model projects the wall into native 640 x 640 camera tiles. Profiles include 4K at 2 m and 3 m, and 1440p at 1.5 m. A low-resolution profile is reserved for stress checks rather than ordinary training.
5. **Grouped splits:** 75% of source walls go to training, 12.5% to validation, and 12.5% to testing. Every view and variant of a physical wall stays within its source split.
6. **Annotation checks:** polygons must agree with their exact instance masks at IoU of at least 0.95 to enter the YOLO lists. Entire views with unsuitable polygon conversion are excluded from those lists, while their exact reference annotations remain available.
7. **Augmentation:** each eligible training image gains five variants: 90, 180, and 270 degree rotations, horizontal flip, and vertical flip. Images, masks, polygons, and projection metadata receive the same transformation. Validation and test data stay unchanged.

Physical dimensions and projection are controlled by the renderer. This does not establish realistic camera optics or performance on real concrete. The larger demonstration cracks in the AirSim room are a separate scene, not the physical scale of this training dataset.

## Files

| Script | Purpose |
| --- | --- |
| `build_calibrated_dataset.py` | Generate the final physical-unit wall dataset and camera tiles. |
| `dataset_utils.py` | Shared annotation, polygon, and preview functions. |
| `validate_dataset.py` | Check masks, labels, hashes, split separation, and polygon agreement. |
| `validate_calibration.py` | Check physical crack dimensions and ideal camera projection. |
| `review_dataset.py` | Sample training records and display raw images beside exported annotation overlays. |
| `augment_dataset.py` | Add five geometric variants of eligible training images, then validate the output. |
| `check_augmentation_provenance.py` | Verify transformed variants against their parents and check that original/evaluation files are unchanged. |

The original v1 generator is omitted. Only the shared annotation functions needed by the final pipeline are retained in `dataset_utils.py`.

## Generate and check the source dataset

Run these commands from the repository root after installing the application requirements in `.venv` as described in the main README:

```powershell
.\.venv\Scripts\python.exe dataset_tools\build_calibrated_dataset.py --output datasets\facade-v2-calibrated --scenes 256 --seed 2026102301
.\.venv\Scripts\python.exe dataset_tools\validate_dataset.py datasets\facade-v2-calibrated
.\.venv\Scripts\python.exe dataset_tools\validate_calibration.py datasets\facade-v2-calibrated
.\.venv\Scripts\python.exe dataset_tools\review_dataset.py datasets\facade-v2-calibrated datasets\annotation-review --count 6
```

Open `datasets/annotation-review/index.html` and check the sampled raw-image/annotation pairs before augmentation. Change `--seed` on the review command to inspect another sample. The generator requires a new output folder and preserves existing datasets.

## Augment and verify

After the source dataset passes validation and its sample annotations look correct:

```powershell
.\.venv\Scripts\python.exe dataset_tools\augment_dataset.py datasets\facade-v2-calibrated datasets\facade-v2-calibrated-augmented
.\.venv\Scripts\python.exe dataset_tools\validate_calibration.py datasets\facade-v2-calibrated-augmented
.\.venv\Scripts\python.exe dataset_tools\check_augmentation_provenance.py datasets\facade-v2-calibrated datasets\facade-v2-calibrated-augmented
```

Use the original source dataset as the augmentation input, not an already augmented dataset. Review `validation.json`, `physical-validation.json`, and `augmentation-validation.json`; each must report `passed: true`.

With the documented seed and 256 source walls, the recorded dataset had 4,452 eligible training images, 124 validation images, and 123 test images. There were 742 eligible original training views; augmentation added 3,710 variants. These variants do not add independent source walls.

## Dataset layout

```text
facade-v2-calibrated-augmented/
├── images/                       # 640 x 640 image tiles
├── masks/                        # Exact per-instance PNG masks
├── labels/                       # YOLO segmentation polygons
├── annotations/                  # COCO RLE annotations by split
├── train.txt / val.txt / test.txt # Eligible image lists
├── data.yaml                     # YOLO configuration
├── manifest.json                 # Views, physical identities and provenance
├── wall-identities.json          # Physical source-wall geometry
├── summary.json                  # Dataset counts and camera profiles
└── *validation.json              # Consistency, geometry and provenance checks
```

Class IDs are fixed: `0 = crack`, `1 = rust`, `2 = spall`. Keep the full folder, including manifests and validation files, when preparing the training input.

## Package for Kaggle

```powershell
Compress-Archive -LiteralPath .\datasets\facade-v2-calibrated-augmented -DestinationPath .\facade-training-v2-calibrated.zip
```

Upload that archive as a Kaggle dataset, attach it to a notebook, and follow [the training reference](../training/README.md). The repository does not include the generated dataset or archive.
