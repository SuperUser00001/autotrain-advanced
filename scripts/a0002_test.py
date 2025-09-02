import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

# ----------------
# 数据集类 (同训练)
# ----------------
class MyDataset(Dataset):
    def __init__(self, x_path, y_path, mask_path, cls_idx, reg_idx):
        self.X = np.load(x_path)  # (N, 9, 3)
        self.Y = np.load(y_path)  # (N, 5)
        self.mask = np.load(mask_path) # (N, 5)

        self.cls_idx = cls_idx
        self.reg_idx = reg_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_cls = torch.tensor(self.Y[idx, self.cls_idx], dtype=torch.long)
        y_reg = torch.tensor(self.Y[idx, self.reg_idx], dtype=torch.float32)
        mask_cls = torch.tensor(self.mask[idx, self.cls_idx], dtype=torch.int16)
        mask_reg = torch.tensor(self.mask[idx, self.reg_idx], dtype=torch.float32)
        return x, y_cls, y_reg, mask_cls, mask_reg

# ----------------
# 模型定义 (需要和训练保持一致)
# ----------------
class TransformerBackbone(nn.Module):
    def __init__(self, input_shape=(9,3), imbed_dim=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.H, self.W = input_shape
        self.seq_len = self.H * self.W

        self.embedding = nn.Linear(1, imbed_dim)
        self.row_embed = nn.Parameter(torch.randn(1, self.H, 1, imbed_dim))
        self.col_embed = nn.Parameter(torch.randn(1, 1, self.W, imbed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=imbed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        b, h, w = x.shape
        x = x.unsqueeze(-1)                          # (b,h,w,1)
        x = self.embedding(x)                        # (b,h,w,imbed_dim)
        x = x + self.row_embed[:, :h, :, :] + self.col_embed[:, :, :w, :]
        x = x.view(b, h*w, -1)                       # (b, seq_len, imbed_dim)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)                 # (b,imbed_dim)
        return x

class MyModel(nn.Module):
    def __init__(self, num_classes_list, num_regression, input_shape):
        super().__init__()
        self.backbone = TransformerBackbone(
            input_shape=input_shape[1:],  # (9,3)
            imbed_dim=128,
            nhead=4,
            num_layers=2
        )
        self.class_heads = nn.ModuleList([
            nn.Linear(128, n_cls) for n_cls in num_classes_list
        ])
        self.reg_head = nn.Linear(128, num_regression)

    def forward(self, x):
        feat = self.backbone(x)
        cls_outs = [head(feat) for head in self.class_heads]
        reg_out = self.reg_head(feat)
        return cls_outs, reg_out

# ----------------
# 测试函数
# ----------------
def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    dataset = MyDataset(config["x"], config["y"], config["mask"], config["cls_idx"], config["reg_idx"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # 模型
    model = MyModel(
        num_classes_list=config["num_classes_list"],
        num_regression=len(dataset.reg_idx),
        input_shape=dataset.X.shape
    ).to(device)

    # 加载权重
    ext = os.path.splitext(config["save_path"])[-1].lower()
    if ext == ".safetensors":
        state_dict = load_file(config["save_path"])
    else:
        state_dict = torch.load(config["save_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 统计量
    total_correct = [0 for _ in config["num_classes_list"]]
    total_count   = [0 for _ in config["num_classes_list"]]
    total_sq_err = 0.0
    total_abs_err = 0.0
    total_reg_count = 0

    with torch.no_grad():
        for x, y_cls, y_reg, mask_cls, mask_reg in dataloader:
            x, y_cls, y_reg, mask_cls, mask_reg = x.to(device), y_cls.to(device), y_reg.to(device), mask_cls.to(device), mask_reg.to(device)

            cls_outs, reg_out = model(x)

            # 分类准确率
            for i, out in enumerate(cls_outs):
                preds = out.argmax(dim=1)
                mask = mask_cls[:, i] > 0
                total_correct[i] += (preds[mask] == y_cls[:, i][mask]).sum().item()
                total_count[i]   += mask.sum().item()

            # 回归误差
            sq_err = ((reg_out - y_reg) ** 2) * mask_reg
            abs_err = (reg_out - y_reg).abs() * mask_reg
            total_sq_err += sq_err.sum().item()
            total_abs_err += abs_err.sum().item()
            total_reg_count += mask_reg.sum().item()

    # 输出结果
    print("=== 测试结果 ===")
    for i, n_cls in enumerate(config["num_classes_list"]):
        if total_count[i] > 0:
            acc = total_correct[i] / total_count[i]
            print(f"分类任务{i}: 准确率 = {acc:.4f} (类别数={n_cls})")
        else:
            print(f"分类任务{i}: 无有效样本")

    if total_reg_count > 0:
        mse = total_sq_err / total_reg_count
        mae = total_abs_err / total_reg_count
        print(f"回归任务: MSE={mse:.4f}, MAE={mae:.4f}")
    else:
        print("回归任务: 无有效样本")


if __name__ == "__main__":
    import json5 as json
    cfg_path = os.path.join(os.path.dirname(__file__), "a0002.json5")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    test_model(config)