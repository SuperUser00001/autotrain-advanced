import os
import json5 as json
import numpy as np
import torch


k=3
batch_size=100
height,width=9,3

value = torch.randn(batch_size,height,width, k)
mask = torch.randint(low=0, high=2, size=(batch_size,height,width,k,))
print(f"value shape dtype: {value.shape},{value.dtype}")
print(f"mask shape dtype: {mask.shape},{mask.dtype}")

x=torch.cat([value,mask], dim=-1)

print(f"x shape dtype: {x.shape},{x.dtype}")

y0=torch.randint(low=0,high=2, size=(batch_size,1))
y1=torch.randint(low=0,high=9, size=(batch_size,1))
y2=torch.randint(low=1,high=99,size=(batch_size,1),dtype=torch.float32)
y3=torch.randint(low=0,high=3, size=(batch_size,1))
y4=torch.randint(low=0,high=1,size=(batch_size,1))

y=torch.cat([y0,y1,y2,y3,y4],dim=-1)

mask_y = torch.ones_like(y, dtype=torch.int32)
mask_y[y[:, 0]==0,1:]=0

print(f"y shape dtype: {y.shape},{y.dtype}")
print(f"mask_y shape dtype: {mask_y.shape},{mask_y.dtype}")

cfg_path = os.path.join(os.path.dirname(__file__), "a0003.json5")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = json.load(f)

file_folder_path = os.path.dirname(config["x"])
print(f"file_folder_path: {file_folder_path}")
os.makedirs(file_folder_path, exist_ok=True)

np.save(config["x"], x)
np.save(config["y"], y)
np.save(config["mask"], mask_y)




