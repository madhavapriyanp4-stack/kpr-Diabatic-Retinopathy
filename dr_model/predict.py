"""
Single-image inference: this is the function the backend API wraps.
Given a fundus photo, returns the predicted DR stage, a confidence score
from MC Dropout, and a Grad-CAM overlay image showing what the model
looked at.

CLI usage:
    python predict.py --image path/to/scan.png --checkpoint best_model.pt

As a library:
    from predict import predict_image
    result = predict_image("scan.png", "best_model.pt")
    # result = {"stage_id": 2, "stage_name": "Moderate NPDR",
    #           "confidence": 84.2, "overlay_path": "overlay.png"}
"""
import argparse
import cv2
import numpy as np
import torch

from dataset import STAGE_NAMES, get_transforms
from gradcam import GradCAM, overlay_heatmap
from model import build_model
from preprocessing import preprocess_fundus
from uncertainty import mc_dropout_predict

STAGE_ACTIONS = [
    "Routine annual screening",
    "Rescreen in 12 months",
    "Review in 3-6 months",
    "Refer to ophthalmologist within 2 weeks",
    "Urgent referral - vision-threatening",
]


def load_model(checkpoint_path: str, device: str = "cpu"):
    model = build_model(num_classes=5, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_image(image_path: str, checkpoint_path: str, size: int = 380,
                   device: str = None, overlay_path: str = "overlay.png"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    raw = cv2.imread(image_path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    processed = preprocess_fundus(raw, size=size)

    transform = get_transforms(size, train=False)
    tensor = transform(image=processed)["image"].unsqueeze(0).to(device) if transform else \
        torch.from_numpy(processed.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

    # 1. Prediction + confidence via MC Dropout
    mean_probs, confidence, stage_id = mc_dropout_predict(model, tensor, n_passes=25)

    # 2. Grad-CAM for the predicted class (single deterministic pass)
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)
    cam = cam_engine.generate(tensor, class_idx=stage_id)
    overlay = overlay_heatmap(processed, cam)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "stage_id": stage_id,
        "stage_name": STAGE_NAMES[stage_id],
        "confidence": round(confidence, 1),
        "recommended_action": STAGE_ACTIONS[stage_id],
        "class_probabilities": {STAGE_NAMES[i]: round(float(p), 3) for i, p in enumerate(mean_probs)},
        "overlay_path": overlay_path,
        "low_confidence_flag": confidence < 75.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="overlay.png")
    args = ap.parse_args()

    result = predict_image(args.image, args.checkpoint, overlay_path=args.out)
    import json
    print(json.dumps(result, indent=2))
