from PIL import Image
import json
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

# 1. Öppna bild
img = Image.open("images/robin_pos.jpg").convert("RGB")
img

# 2. Ladda modell
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# 3. Preprocess
preprocess = weights.transforms()

# 4. Gör input
input_tensor = preprocess(img).unsqueeze(0)

# 5. Kör modellen
outputs = model(input_tensor)

# 6. Sannolikheter
probs = F.softmax(outputs, dim=1)

# 7. Top-5
top_probs, top_idxs = probs.topk(5, dim=1)

# 8. Ladda JSON
with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)

# 9. Skriv ut topp-5
for i in range(5):
    idx = top_idxs[0, i].item()
    prob = top_probs[0, i].item()
    label = class_idx[str(idx)][1]
    print(f"{i+1}. {label}: {prob:.4f}")