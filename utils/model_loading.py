from utils.gesture_model import GestureMLP
import torch


model_path='gesture_model.pth'
scaler_mean_path='scaler_mean.npy'
scaler_scale_path='scaler_scale.npy'


model = GestureMLP(input_size=42)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)
print("Loaded scaler mean shape:", scaler_mean.shape)  # Should be (42,)
print("Loaded scaler scale shape:", scaler_scale.shape)