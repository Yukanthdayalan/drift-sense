import os
import sys
import json
import random
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from drift_sense.config import get_default_config
from drift_sense.preprocess import preprocess_image
from drift_sense.geometry import get_scaled_dimensions, resize_reference_for_scale
from drift_sense.zncc import compute_zncc_fft
from drift_sense.peak_detector import detect_peaks
from drift_sense.disambiguator import DisambiguatorCNN

class CropDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        crop, label = self.data[idx]
        # Crop is already Z-score normalized
        crop_tensor = torch.from_numpy(crop).float().unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return crop_tensor, label_tensor

def extract_crop(image, cx, cy, size=64):
    h, w = image.shape
    x_min = int(cx - size // 2)
    y_min = int(cy - size // 2)
    
    # Pad if necessary
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

def generate_crops(dataset_dir):
    config = get_default_config()
    
    eval_dir = os.path.join(dataset_dir, "eval")
    samples = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
    
    # Hold out 20% of pairs
    random.seed(42)
    random.shuffle(samples)
    num_train = int(len(samples) * 0.8)
    
    train_samples = samples[:num_train]
    val_samples = samples[num_train:]
    
    def process_split(split_samples):
        crops = []
        for sample in split_samples:
            sample_dir = os.path.join(eval_dir, sample)
            ref_path = os.path.join(sample_dir, "reference.png")
            search_path = os.path.join(sample_dir, "search.png")
            gt_path = os.path.join(sample_dir, "ground_truth.json")
            
            if not os.path.exists(gt_path):
                continue
                
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
                
            scale = gt_data["scale"]
            gt_cx = gt_data["gt_center_x"]
            gt_cy = gt_data["gt_center_y"]
            
            ref_norm = preprocess_image(ref_path, config.preprocessing)
            search_norm = preprocess_image(search_path, config.preprocessing)
            
            ref_h, ref_w = ref_norm.shape
            scaled_h, scaled_w = get_scaled_dimensions(ref_h, ref_w, scale)
            ref_scaled = resize_reference_for_scale(ref_norm, scale)
            
            response_map = compute_zncc_fft(ref_scaled, search_norm)
            try:
                candidates = detect_peaks(response_map, config.nms)
            except Exception:
                continue
            
            # Positive crops: jittered around GT center
            for _ in range(3):
                jx = gt_cx + random.uniform(-2, 2)
                jy = gt_cy + random.uniform(-2, 2)
                pos_crop = extract_crop(search_norm, jx, jy)
                crops.append((pos_crop, 1.0))
                
            # Negative crops: highest scoring false peaks
            false_peaks = 0
            for cand in candidates:
                cand_cx = cand.x + (scaled_w - 1) / 2.0
                cand_cy = cand.y + (scaled_h - 1) / 2.0
                dist = math.sqrt((cand_cx - gt_cx)**2 + (cand_cy - gt_cy)**2)
                if dist > 5.0:  # Far enough to be false
                    neg_crop = extract_crop(search_norm, cand_cx, cand_cy)
                    crops.append((neg_crop, 0.0))
                    false_peaks += 1
                    if false_peaks >= 5: # Limit negative crops per sample
                        break
        return crops
        
    print("Processing training pairs...")
    train_crops = process_split(train_samples)
    print("Processing validation pairs...")
    val_crops = process_split(val_samples)
    
    return train_crops, val_crops

def main():
    train_crops, val_crops = generate_crops("dataset_train")
    print(f"Train crops: {len(train_crops)}, Val crops: {len(val_crops)}")
    
    train_dataset = CropDataset(train_crops)
    val_dataset = CropDataset(val_crops)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = DisambiguatorCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_crops)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_crops)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/disambiguator.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("Training complete. Model saved to models/disambiguator.pt")

if __name__ == "__main__":
    main()
