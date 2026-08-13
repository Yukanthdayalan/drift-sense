import torch
import torch.nn as nn
import torch.nn.functional as F

class DisambiguatorCNN(nn.Module):
    def __init__(self):
        super(DisambiguatorCNN, self).__init__()
        # Input: 1 channel, 64x64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x is (B, 1, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global Average Pool
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_disambiguator(model_path="models/disambiguator.pt"):
    model = DisambiguatorCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

def extract_crop(image, cx, cy, size=64):
    import cv2
    h, w = image.shape
    x_min = int(cx - size // 2)
    y_min = int(cy - size // 2)
    
    pad_t = max(0, -y_min)
    pad_b = max(0, y_min + size - h)
    pad_l = max(0, -x_min)
    pad_r = max(0, x_min + size - w)
    
    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        y_min += pad_t
        x_min += pad_l
        
    crop = image[y_min:y_min+size, x_min:x_min+size]
    return crop.copy()
