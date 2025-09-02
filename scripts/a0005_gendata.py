import os
import json5 as json
import numpy as np
import torch

batch_size =  2000

y0 = torch.randint(low=0, high=3, size=(batch_size,1),dtype=torch.float32)
y1=torch.zeros_like(y0)
y2=torch.zeros_like(y0)
y3=torch.zeros_like(y0)

y=torch.cat([y0,y1,y2,y3], dim=-1)

mask_y = torch.ones_like(y, dtype=torch.int32) # 1 有意义 0 忽略不计算误差
# y[y[:,0]==0,1:]=0 # 无故障


# 设置母线故障时，线路索引0-8
y[y[:,0]==1, 1:2]=torch.randint(low=0,high=9, size=(y[y[:,0]==1,1].numel(),1),dtype=y.dtype)
# 设置母线间故障时，线路索引9-15
y[y[:,0]==2, 1:2]=torch.randint(low=9,high=16, size=(y[y[:,0]==2,1:2].numel(),1),dtype=y.dtype)
# 设置母线间故障时，百分比位置1-99
y[y[:,0]==2, 2:3]=torch.randint(low=1,high=99,size=(y[y[:,0]==2,2:3].numel(),1),dtype=y.dtype)
# 设置母线上或母线间的3相随机
y[(y[:,0]==1)|(y[:,0]==2), 3:4]=torch.randint(low=0,high=3,size=(y[(y[:,0]==1)|(y[:,0]==2),3:4].numel(),1),dtype=y.dtype)

# 设置y掩码
mask_y[y[:,0]==0,1:]=0
mask_y[y[:,0]==1,2]=0

# print(f"y={y}, mask_y={mask_y}")

x=torch.rand(batch_size, 25,3,2)*1.3

mask_x = torch.ones(batch_size, 25, dtype=torch.int32)
print(mask_x, mask_y)


cfg_path = os.path.join(os.path.dirname(__file__), "a0005.json5")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = json.load(f)

file_folder_path = os.path.dirname(config["x"])
print(f"file_folder_path: {file_folder_path}")
os.makedirs(file_folder_path, exist_ok=True)

np.save(config["x"], x)
np.save(config["mask_x"], mask_x)
np.save(config["y"], y)
np.save(config["mask_y"], mask_y)
