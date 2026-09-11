"""
Monte Carlo Dropout: instead of trusting a single softmax score, run the
model N times with dropout kept active and look at how much the
predictions agree. Low agreement -> low confidence -> flag for review.
"""
import torch
import torch.nn as nn
import numpy as np


def _enable_dropout(model: nn.Module):
    """Keep the model in eval() (so BatchNorm uses running stats) but
    switch Dropout layers back to train() mode so they stay stochastic."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, input_tensor: torch.Tensor, n_passes: int = 25):
    """
    Returns:
        mean_probs: (num_classes,) averaged softmax probabilities
        confidence: 0-100 score, higher = more agreement across passes
        pred_class: argmax of mean_probs
    """
    model.eval()
    _enable_dropout(model)

    all_probs = []
    for _ in range(n_passes):
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)     # (n_passes, num_classes)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    pred_class = int(mean_probs.argmax())
    # Confidence: high mean probability on the predicted class, penalized
    # by how much that probability wobbled across the stochastic passes.
    confidence = float(mean_probs[pred_class] * (1 - std_probs[pred_class]) * 100)
    confidence = max(0.0, min(100.0, confidence))

    return mean_probs, confidence, pred_class
