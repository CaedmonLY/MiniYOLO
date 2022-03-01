
import torch

org=torch.load("yolov5s.pt")
for k in org:
    print(k)
print(org["model"])


init=torch.load("yolo5")

