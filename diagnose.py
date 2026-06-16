import torch, cv2, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.train import get_model
from src.augmentations import DeepShieldAugmentations

model = get_model('hybrid', pretrained=False)
ckpt = torch.load(os.path.join(Config.CHECKPOINT_DIR, 'hybrid_best.pth'), map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
transforms = DeepShieldAugmentations.get_val_transforms()

print(f"NORM_MEAN: {Config.NORM_MEAN}")
print(f"NORM_STD:  {Config.NORM_STD}")
print(f"IMAGE_SIZE: {Config.IMAGE_SIZE}")
print("-" * 60)

# Test on known real and fake images from test set
for label_name in ['real', 'fake']:
    folder = os.path.join(Config.TEST_DIR, label_name)
    imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))][:5]
    for img_name in imgs:
        img = cv2.imread(os.path.join(folder, img_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = transforms(image=img_rgb)['image'].unsqueeze(0)
        with torch.no_grad():
            logit = model(t).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        status = "[CORRECT]" if (label_name == 'fake') == (prob >= 0.5) else "[WRONG]"
        print(f"[{label_name.upper():4s}] prob={prob:.4f} logit={logit:+.3f} -> {'FAKE' if prob>=0.5 else 'REAL'} {status}  ({img_name})")

print("-" * 60)
print("Done. If all above are CORRECT, the model itself works fine.")
print("If fakes show prob near 0.00, the label assignment is INVERTED (real=1, fake=0).")
