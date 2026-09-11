"""
Loads the trained checkpoint once and keeps it in memory, so each API
request just runs a forward pass instead of reloading the model from
disk every time (predict.py's predict_image() is fine for one-off CLI
use, but too slow to call per-request in a live backend).
"""
import io
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# dr_model/ is expected to sit next to this backend/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dr_model"))

from dataset import STAGE_NAMES, get_transforms          # noqa: E402
from gradcam import GradCAM, overlay_heatmap              # noqa: E402
from model import build_model                             # noqa: E402
from preprocessing import preprocess_fundus                # noqa: E402
from uncertainty import mc_dropout_predict                 # noqa: E402

STAGE_ACTIONS = [
    "Routine annual screening",
    "Rescreen in 12 months",
    "Review in 3-6 months",
    "Refer to ophthalmologist within 2 weeks",
    "Urgent referral - vision-threatening",
]
URGENCY = ["Routine", "Routine", "Review", "Refer", "Urgent"]


class DRPredictor:
    def __init__(self, checkpoint_path: str, size: int = 380, device: str = None):
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(num_classes=5, device=self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = get_transforms(size, train=False)
        target_layer = self.model.features[-1]
        self.cam_engine = GradCAM(self.model, target_layer)

        print(f"DRPredictor loaded on {self.device} from {checkpoint_path}")

    def predict(self, image_bytes: bytes, overlay_dir: str = "storage/overlays"):
        """Runs the full pipeline on raw image bytes (as received from an
        HTTP upload) and returns a result dict plus the overlay's saved path."""
        os.makedirs(overlay_dir, exist_ok=True)

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw = np.array(pil_img)
        processed = preprocess_fundus(raw, size=self.size)

        if self.transform:
            tensor = self.transform(image=processed)["image"].unsqueeze(0).to(self.device)
        else:
            tensor = torch.from_numpy(processed.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

        mean_probs, confidence, stage_id = mc_dropout_predict(self.model, tensor, n_passes=20)

        cam = self.cam_engine.generate(tensor, class_idx=stage_id)
        overlay = overlay_heatmap(processed, cam)

        overlay_filename = f"{uuid.uuid4().hex}.png"
        overlay_path = os.path.join(overlay_dir, overlay_filename)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            "stage_id": stage_id,
            "stage_name": STAGE_NAMES[stage_id],
            "confidence": round(confidence, 1),
            "urgency": URGENCY[stage_id],
            "recommended_action": STAGE_ACTIONS[stage_id],
            "class_probabilities": {STAGE_NAMES[i]: round(float(p), 3) for i, p in enumerate(mean_probs)},
            "low_confidence_flag": confidence < 75.0,
            "overlay_filename": overlay_filename,
        }


_predictor_instance = None


def get_predictor(checkpoint_path: str = None) -> DRPredictor:
    """Singleton accessor so the (slow-ish) model load only happens once."""
    global _predictor_instance
    if _predictor_instance is None:
        checkpoint_path = checkpoint_path or os.environ.get(
            "DR_CHECKPOINT_PATH", str(Path(__file__).resolve().parent / "best_model.pt")
        )
        _predictor_instance = DRPredictor(checkpoint_path)
    return _predictor_instance
