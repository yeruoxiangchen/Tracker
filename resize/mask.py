import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# 1. 加载模型
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # 小模型，速度快
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# 2. 图像预处理
img = cv2.imread("/home/zjr/Tracker/resize/RealSenseRecorder/3/color/1761723173108.png")  # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
input_batch = transform(img).to(device)

# 3. 前向推理
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# 4. 可视化深度图
import matplotlib.pyplot as plt
plt.imshow(depth_map, cmap='plasma')
plt.colorbar()
plt.show()
