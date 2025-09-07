import torch
from model_loading import model, scaler_mean, scaler_scale, device
import numpy as np
def preprocess_landmarks(landmarks):
    """Extract and preprocess hand landmarks (2D coordinates)."""
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y])  # 2D coordinates only
        
    coords = np.array(coords).reshape(1, -1)
    print("Landmarks shape:", coords.shape)  # Should be (1, 42)
    coords = (coords - scaler_mean) / scaler_scale  # Standardize
    return torch.tensor(coords, dtype=torch.float32).to(device)