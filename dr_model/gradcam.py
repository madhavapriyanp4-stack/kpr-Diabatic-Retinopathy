"""
Grad-CAM: highlights the image regions that most influenced the model's
prediction, computed from gradients flowing into the last conv layer.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a normalized (0-1) heatmap the size of the input image."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global avg pool of grads
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(rgb_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original RGB image (both 0-255 uint8)."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay
