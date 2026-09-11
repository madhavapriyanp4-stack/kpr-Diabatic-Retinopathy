# DR Severity Classifier

EfficientNet-B0 classifier for 5-stage diabetic retinopathy grading, with
Grad-CAM explainability and MC Dropout confidence scoring. This is the
model layer for the RetinaCheck clinic screening tool.

## 1. Get the data

Download APTOS 2019 from Kaggle (needs a free account + API key):

```bash
pip install kaggle
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d aptos_data
```

You should end up with `aptos_data/train.csv` and `aptos_data/train_images/`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended: run training on Google Colab with a GPU runtime (Runtime →
Change runtime type → GPU) unless you have a local CUDA GPU.

## 3. Train

```bash
python train.py --csv aptos_data/train.csv --img_dir aptos_data/train_images --epochs 20
```

This saves `best_model.pt`, the checkpoint with the highest validation
Cohen's Kappa (used instead of accuracy since the classes are imbalanced
- see the dataset's class distribution: No DR and Moderate dominate,
Severe and Proliferative are rare).

Expect roughly 30-60 minutes for 20 epochs on a Colab T4 GPU, depending
on batch size.

## 4. Run inference on a single image

```bash
python predict.py --image sample_scan.png --checkpoint best_model.pt --out overlay.png
```

Prints a JSON result:
```json
{
  "stage_id": 2,
  "stage_name": "Moderate NPDR",
  "confidence": 84.2,
  "recommended_action": "Review in 3-6 months",
  "class_probabilities": {...},
  "overlay_path": "overlay.png",
  "low_confidence_flag": false
}
```

`overlay.png` is the original image with the Grad-CAM heatmap blended in
- this is what the "Show Grad-CAM overlay" toggle in the frontend displays.

## Files

| File | Purpose |
|---|---|
| `preprocessing.py` | Retina crop + CLAHE contrast enhancement |
| `dataset.py` | PyTorch Dataset for APTOS + augmentation |
| `model.py` | EfficientNet-B0 backbone + 5-class head |
| `gradcam.py` | Grad-CAM heatmap generation |
| `uncertainty.py` | MC Dropout confidence scoring |
| `train.py` | Training loop, saves best checkpoint by Kappa |
| `predict.py` | Single-image inference - this is what the backend API calls |

## Next step

Wrap `predict_image()` from `predict.py` in a FastAPI endpoint
(`POST /predict`) so the React frontend can call it instead of using mock
random results.
