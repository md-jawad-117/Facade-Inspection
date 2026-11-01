# Training reference

`train_facade_v2_single_cell.py` is the standalone Kaggle training code used for this project. It is included for reference and optional retraining. The inspection app uses the existing `weights/best.pt` and does not execute this file.

## Prepare the input

Generate, review, validate, and augment the dataset using [the dataset creation instructions](../dataset_tools/README.md). Upload `facade-training-v2-calibrated.zip` as a Kaggle dataset, or upload its extracted folder while preserving its contents.

Create a Kaggle Python notebook, attach that dataset with **Add Input**, enable a **GPU accelerator**, and enable **Internet** for package installation and the initial pretrained checkpoint download. Paste the entire training file into one notebook cell and run it when ready to train.

The cell installs `ultralytics==8.4.176` and ONNX export dependencies when `INSTALL_PACKAGES=True`. It checks GPU availability, the dataset version, validation results, all eligible image hashes and polygon labels, and wall-split isolation before starting training.

## Settings at the top of the cell

| Setting | Default | What to change |
| --- | --- | --- |
| `DATASET_ROOT` | `None` | Automatic discovery when exactly one matching dataset is attached. Set an explicit path if multiple datasets are present. |
| `EPOCHS` | `80` | Maximum training epochs. Use a smaller value for a shorter run. |
| `PATIENCE` | `10` | Early stopping after this many epochs without validation improvement. |
| `BATCH` | `8` | Reduce to `4` or `2` if GPU memory is insufficient. |
| `IMAGE_SIZE` | `640` | Keep this value for the native 640-pixel tiles. |
| `SMOKE_TEST` | `False` | Set `True` for a two-epoch run on a training subset to check the pipeline. |
| `RUN_FINAL_TEST` | `False` | Enable only after model and settings are fixed; keep the test split reserved during tuning. |
| `EXPORT_ONNX` | `True` | Set `False` to skip optional ONNX export. The app uses `best.pt`. |
| `SHOW_PREDICTIONS` | `True` | Display six validation predictions after training. |
| `INSTALL_PACKAGES` | `True` | Set `False` only if the required packages are already installed. |
| `WORKING_DIR` | `/kaggle/working` | Kaggle's writable output directory. |

For an explicit dataset path, edit the first setting, for example:

```python
DATASET_ROOT = "/kaggle/input/your-dataset-name/facade-v2-calibrated-augmented"
```

Replace the example with the actual folder containing `summary.json`, `manifest.json`, images, labels, and validation files. Do not point it at the `images` subfolder. If the dataset is attached as an archive, leave `DATASET_ROOT=None` so the code can discover and extract it.

## Model and training process

- Start from the pretrained `yolo26n-seg.pt` checkpoint.
- Train instance segmentation with class order `crack`, `rust`, `spall`.
- Use a fixed seed, validation monitoring, early stopping, and saved best weights.
- Preserve detailed masks with `mask_ratio=1` and `overlap_mask=False`.
- Disable mosaic and mixup. Training still applies the declared small photometric and geometric variations in addition to the offline dataset variants.
- Evaluate validation performance after training and optionally evaluate the reserved test split.
- Save the checkpoint, training metrics, dataset provenance, and optional ONNX export.

The included checkpoint came from the supplied five-epoch run. The reference cell now permits up to 80 epochs with patience 10; those defaults do not mean the included checkpoint was trained for 80 epochs. Early stopping monitors validation progress, not a fixed mAP target.

## Retrieve the result

The cell saves `/kaggle/working/facade-model/` and creates `/kaggle/working/facade-model.zip`. A smoke run uses `facade-model-smoke` instead. Download the archive from Kaggle Output. The main files include:

```text
best.pt               # Best checkpoint selected during training
model-metadata.json   # Model details, dataset hash, settings and metrics
results.csv           # Epoch metrics, when available
results.png           # Training plots, when available
args.yaml             # Training arguments, when available
model.onnx            # Optional export
```

To try a newly trained model in the application, keep a backup of the existing `weights/best.pt`, place the new compatible checkpoint there, and stop the old model worker before restarting. Changing the checkpoint invalidates prior acceptance results and requires a new South-wall verification before multiple-wall inspection.

Synthetic validation metrics do not establish real-camera accuracy, structural severity, or correct unique physical defect counts. Those require independent inspection evidence.
