import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

# ----------------
# 数据集类
# ----------------
class MyDataset(Dataset):
    def __init__(self, x_path, y_path, mask_path, cls_idx, reg_idx):
        self.X = np.load(x_path)  # (N, 9, 3)
        self.Y = np.load(y_path)  # (N, 5)
        self.mask = np.load(mask_path) # (N, 5)

        # 拆分分类 & 回归标签
        self.cls_idx = cls_idx  # 分类
        self.reg_idx = reg_idx  # 回归

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # (499, 9, 3)
        y_cls = torch.tensor(self.Y[idx, self.cls_idx], dtype=torch.long)  # 分类
        y_reg = torch.tensor(self.Y[idx, self.reg_idx], dtype=torch.float32)  # 回归
        mask_cls = torch.tensor(self.mask[idx, self.cls_idx], dtype=torch.int16) # 分类掩码
        mask_reg = torch.tensor(self.mask[idx, self.reg_idx], dtype=torch.float32) # 回归掩码
        return x, y_cls, y_reg, mask_cls, mask_reg


# ----------------
# 模型定义
# ----------------
class MyModel(nn.Module):
    def __init__(self, num_classes_list, num_regression, input_shape):
        super().__init__()
        ######################## Bgn: 可以自行定义模型层 ########################
        # 输入: (9, 3) → 展平
        self.flatten = nn.Flatten()
        input_dim = np.prod(input_shape[1:])

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # 多个分类头
        self.class_heads = nn.ModuleList([
            nn.Linear(256, n_cls) for n_cls in num_classes_list
        ])

        # 一个回归头（多个输出）
        self.reg_head = nn.Linear(256, num_regression)
        ######################## End: 可以自行定义模型层 ########################

    def forward(self, x):
        ######################## Bgn: 根据模型层定义自行搭建模型 ########################
        x = self.flatten(x)
        feat = self.backbone(x)
        cls_outs = [head(feat) for head in self.class_heads]
        reg_out = self.reg_head(feat)
        ######################## End: 根据模型层定义自行搭建模型 ########################
        return cls_outs, reg_out


# ----------------
# 训练函数
# ----------------
def train_model(config):
    cls_ignore_index = -100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset(config["x"], config["y"], config["mask"], config["cls_idx"], config["reg_idx"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = MyModel(num_classes_list=config["num_classes_list"],
                    num_regression=len(train_dataset.reg_idx),
                    input_shape=train_dataset.X.shape,
                    ).to(device)

    # 损失函数
    cls_criterions = [nn.CrossEntropyLoss(ignore_index=cls_ignore_index) for _ in config["num_classes_list"]]
    reg_criterion = nn.MSELoss(reduction="none")  # 改为逐元素

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    with tqdm(range(config["epochs"]), desc=f"Epoch 0/{config['epochs']} - Loss: NaN", total=config["epochs"]) as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0
            for x, y_cls, y_reg, mask_cls, mask_reg in train_loader:
                x, y_cls, y_reg, mask_cls, mask_reg = x.to(device), y_cls.to(device), y_reg.to(device), mask_cls.to(device), mask_reg.to(device)

                optimizer.zero_grad()
                cls_outs, reg_out = model(x)

                # loss_cls = sum(crit(out, y_cls[:, i]) for i, (crit, out) in enumerate(zip(cls_criterions, cls_outs)))
                loss_cls = 0
                for i, (crit, out ) in enumerate(zip(cls_criterions, cls_outs)): # 计算带掩码的分类损失
                    target = y_cls[:,i].clone()
                    target[mask_cls[:, i] == 0] = cls_ignore_index # 掩码为 0-> 忽略
                    loss_cls += crit(out, target)

                # loss_reg = reg_criterion(reg_out, y_reg)
                diff_reg = reg_criterion(reg_out, y_reg) # reg_out, y_reg ** 2
                loss_reg = (diff_reg * mask_reg).sum()/ mask_reg.sum().clamp(min=1.0)

                loss = loss_cls + config["lambda_reg"] * loss_reg
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if avg_loss < config["loss_update_threshold"]:
                pbar.set_description(f"Epoch {epoch + 1}/{config['epochs']} - Loss: {avg_loss:.4f}, less then threshold: {config['loss_update_threshold']:.4f}, break")
                pbar.update(config['epochs'])
                break
            else:
                pbar.set_description(f"Epoch {epoch + 1}/{config['epochs']} - Loss: {avg_loss:.4f}")


    # 保存模型
    save_path = config["save_path"]
    ext = os.path.splitext(save_path)[-1].lower()

    if ext == ".safetensors":
        save_file(model.state_dict(), save_path)
    elif ext in [".pth", ".pt"]:
        torch.save(model.state_dict(), save_path)
    else:
        raise ValueError(f"不支持的模型文件后缀: {ext}")
    print("模型已保存到", config["save_path"])


# 主入口
if __name__ == "__main__":
    import json5 as json

    cfg_path = os.path.join(os.path.dirname(__file__), "a0001.json5")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    train_model(config)
