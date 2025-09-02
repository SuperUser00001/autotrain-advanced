import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

# ----------------
# 数据集类
# ----------------
class MyDataset(Dataset):
    def __init__(self, x_path, y_path, mask_path, cls_idx, reg_idx):
        self.X = np.load(x_path)  # (BatchSize, 9, 3, 2*k)
        self.Y = np.load(y_path)  # (BatchSize, 5)
        self.mask = np.load(mask_path) # （BatchSize, 5)

        # 拆分分类 & 回归标签
        #      正常or故障   故障线索引  故障位置    故障相   故障类型
        self.cls_idx = cls_idx  # 分类
        self.reg_idx = reg_idx  # 回归

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # (9, 3, 2*k)
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
        self.backbone = TransformerBackbone(
            input_shape=input_shape[1:],  # (9,3,2*k)
            embed_dim=128,
            nhead=4,
            num_layers=2
        )

        # 分类头
        self.class_heads = nn.ModuleList([
            nn.Linear(128, n_cls) for n_cls in num_classes_list
        ])
        # 回归头
        self.reg_head = nn.Linear(128, num_regression)

        ######################## End: 可以自行定义模型层 ########################

    def forward(self, x):
        ######################## Bgn: 根据模型层定义自行搭建模型 ########################
        feat = self.backbone(x)   # (batch, embed_dim)
        cls_outs = [head(feat) for head in self.class_heads]
        reg_out = self.reg_head(feat)
        ######################## End: 根据模型层定义自行搭建模型 ########################
        return cls_outs, reg_out

# ----------------
# TransformerBackbone
# ----------------
class TransformerBackbone(nn.Module):
    def __init__(self, input_shape=(9,3,6), embed_dim=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.H, self.W, self.C = input_shape  # 输入形状 H=9 W=3 C=2*k
        if self.C %2 != 0: raise Exception("must has value and mask, even number for C")
        self.k = self.C // 2
        self.seq_len = self.H * self.W

        # 数值部分 embedding：把原始向量 → embed_dim 向量
        self.value_embedding  = nn.Linear(self.k, embed_dim)
        # 缺失标记
        self.mask_embedding = nn.Embedding(2, embed_dim)
        # self.mask_embedding = nn.Linear(self.k, embed_dim)

        # 2D 位置编码（可学习参数）
        self.row_embed = nn.Parameter(torch.randn(1, self.H, 1, embed_dim))
        self.col_embed = nn.Parameter(torch.randn(1, 1, self.W, embed_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 池化
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (batch, H=9, W=3, C=2*k)
        """
        b, h, w, c = x.shape
        if not ( h == self.H and w == self.W and c == self.C):
            raise Exception("输入尺寸不匹配: {x.shape}, 需要 {(self.H, self.W, self.C)}")

        # (b, h, w, c) → (b*h*w, 1) → embedding → (b, h*w, embed_dim)
        values = x[:, :, :, :self.k]
        mask = x[:, :, :, self.k].long()
        # 数值embedding
        val_emb = self.value_embedding(values.view(b, h*w, self.k))    # (b, h*w, embed_dim)

        # 缺失embedding
        mask_emb = self.mask_embedding(mask.view(b, h*w))

        # 融合数值和缺失
        x = val_emb + mask_emb

        # 加 2D 位置编码
        pos = (self.row_embed[:, :h, :, :] + self.col_embed[:, :, :w, :]).view(1, h*w, -1)
        x = x + pos                                  # (b, seq_len, embed_dim)

        # Transformer 编码
        x = self.transformer(x)                      # (b, seq_len, embed_dim)

        # 池化 → (b, embed_dim)
        x = x.transpose(1, 2)                        # (b, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)                 # (b, embed_dim)
        return x

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
    folder_path = os.path.dirname(save_path)
    os.makedirs(folder_path, exist_ok=True)
    ext = os.path.splitext(save_path)[-1].lower()

    if ext == ".safetensors":
        save_file(model.state_dict(), save_path)
    elif ext in [".pth", ".pt"]:
        torch.save(model.state_dict(), save_path)
    else:
        raise ValueError(f"不支持的模型文件后缀: {ext}")
    print("模型已保存到", config["save_path"])


if __name__ == '__main__':
    import json5 as json

    cfg_path = os.path.join(os.path.dirname(__file__), "a0003.json5")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    train_model(config)