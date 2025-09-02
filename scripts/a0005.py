import os
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import save_file, load_file
from datetime import datetime

class DataInfo:
    num_nodes = 9
    num_lines = 7
    num_parts = num_nodes + num_lines # 16
    num_types = 2
    num_rows = 25
    num_phases = 3
    num_complex = 2

# dataset
class MyDataset(Dataset):
    def __init__(self, x_path, mask_x_path, y_path, mask_y_path, cls_idx, reg_idx):
        self.x = np.load(x_path)                # [bs,25,3,2]
        self.mask_x = np.load(mask_x_path)      # [bs,25]
        self.y = np.load(y_path)                # [bs,4]
        self.mask_y = np.load(mask_y_path)      # [bs,4]
        print(f"MyDataset: x.shape={self.x.shape}")
        print(f"MyDataset: mask_x.shape={self.mask_x.shape}")
        print(f"MyDataset: y.shape={self.y.shape}")
        print(f"MyDataset: mask_y.shape={self.mask_y.shape}")
        # 拆分分类 & 回归标签
        #      正常or故障   故障线索引  故障位置    故障相
        self.cls_idx = cls_idx  # 分类
        self.reg_idx = reg_idx  # 回归
        print(f"MyDataset: cls_idx={self.cls_idx}")
        print(f"MyDataset: reg_idx={self.reg_idx}")
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)  # (25, 3, 2)
        mask_x = torch.tensor(self.mask_x[idx], dtype=torch.int16) # (25, )
        y_cls = torch.tensor(self.y[idx, self.cls_idx], dtype=torch.long)  # 分类
        y_reg = torch.tensor(self.y[idx, self.reg_idx], dtype=torch.float32)  # 回归
        mask_cls = torch.tensor(self.mask_y[idx, self.cls_idx], dtype=torch.int16) # 分类掩码
        mask_reg = torch.tensor(self.mask_y[idx, self.reg_idx], dtype=torch.float32) # 回归掩码
        return x, mask_x, y_cls, y_reg, mask_cls, mask_reg


class LinRelu(nn.Module): # linear + relu?
    def __init__(self, in_features, out_features, bias:bool=True, device=None, dtype=None, inplace:bool=False)-> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return self.relu(self.linear(x))

class MyModule(nn.Module):
    def __init__(self, num_classes_list, num_regression):
        super().__init__()
        self.step_001 = Step001(512, "cat")  # [bs,25, 3,2, embed_dim]
        self.step_002 = Step002(512, 8, 6, 6) # [bs,25]
        # 分类头
        self.class_heads = nn.ModuleList([
            nn.Linear(DataInfo.num_rows, n_cls) for n_cls in num_classes_list
        ])
        # 回归头
        self.reg_head = nn.Linear(DataInfo.num_rows, num_regression)
        pass
    def forward(self, x, mask_x):
        """
        x     : bs, 25, 3, 2,
        mask_x: bs, 25,
        """
        x = self.step_001(x) # [bs, 25, 3, 2] -> [bs, 25, 3, 2, embed_dim]
        x = self.step_002(x, mask_x) # ([bs, 25, 3, 2, embed_dim], [bs, 25,]) -> [bs, 25, embed_dim]
        cls_outs = [head(x) for head in self.class_heads]
        reg_out = self.reg_head(x)
        return cls_outs, reg_out

class Step001(nn.Module):
    def __init__(self, embed_dim, mode = "add"):
        super().__init__()
        self.linear = LinRelu(1, embed_dim)
        self.ids_embed = nn.Embedding(DataInfo.num_parts, embed_dim)
        self.types_embed = nn.Embedding(DataInfo.num_types, embed_dim)
        self.mode = mode
        if mode == "cat":
            self.fuse = LinRelu(3*embed_dim, embed_dim)

    def forward(self, x): # [bs, 25, 3, 2]
        bs, num_rows, num_phases, num_complex = x.shape
        if (num_rows != DataInfo.num_rows or num_phases != DataInfo.num_phases or num_complex != DataInfo.num_complex):
            raise f"输入尺寸不匹配: {(num_rows, num_phases, num_complex)}, 需要 {(DataInfo.num_rows, DataInfo.num_phases, DataInfo.num_complex)}"

        x = x.unsqueeze(-1) # [bs,25,3,2,1]
        x = self.linear(x) # [bs,25,3,2,embed_dim]
        ids = torch.tensor(list(range(0, 9)) + list(range(0, 9)) + list(range(9, 16)))  # shape [25]
        types = torch.tensor([0] * 9 + [1] * 16)  # shape [25]
        # [bs, 25, 3,2]
        ids = ids.unsqueeze(-1).repeat(1, DataInfo.num_phases).unsqueeze(-1).repeat(1, 1, DataInfo.num_complex).unsqueeze(0).expand(bs, -1,-1,-1).to(x.device)
        types = types.unsqueeze(-1).repeat(1, DataInfo.num_phases).unsqueeze(-1).repeat(1, 1, DataInfo.num_complex).unsqueeze(0).expand(bs, -1,-1,-1).to(x.device)

        ids = self.ids_embed(ids) # [bs,25, 3,2, embed_dim]
        types = self.types_embed(types) # [bs,25, 3,2, embed_dim]

        if self.mode == "cat":
            return self.fuse(torch.cat([x, ids, types], dim = -1))
        return x + ids + types  # [bs,25, 3,2, embed_dim]

class Step002(nn.Module):
    def __init__(self, embed_dim, nhead, local_depth, global_depth):
        super().__init__()
        local_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, batch_first=True)
        self.local_encoder = nn.TransformerEncoder(local_encoder_layer, local_depth)

        global_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, batch_first=True)
        self.global_encoder = nn.TransformerEncoder(global_encoder_layer, global_depth)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask_x):
        """
        x     : bs, 25, 3, 2, embed_dim,
        mask_x: bs, 25,
        """
        bs, num_rows, num_phases, num_complex, embed_dim = x.shape
        local_mask = torch.log(mask_x.unsqueeze(-1).repeat(1, 1,  num_phases*num_complex)) # [bs, num_rows, 6]  0有效，-inf 忽略
        global_mask = torch.log(mask_x) # [bs, num_rows,]
        # reshape 3相复数作为整体
        x = x.view(bs*num_rows, num_phases*num_complex, embed_dim) # [B*N, 6, embed_dim]
        x = self.local_encoder(
                x,
                src_key_padding_mask=local_mask.view(bs*num_rows, num_phases*num_complex),
            ) # [B*N, 6, embed_dim]

        # 汇成一个 embedding
        x = x.mean(dim=1) # [bs*N, embed_dim]
        x = x.view(bs, num_rows, embed_dim)

        x = self.global_encoder(
                x,                   # [B, N, embed_dim]
                src_key_padding_mask = global_mask,
            ) # [B, N, embed_dim]
        x = self.pool(x).squeeze(-1) # [B,N]
        return x # [B,N]


class TrainTester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_by_safetensors_file(self, source_model_path, model, optimizer):
        state_dict = load_file(source_model_path)
        model.load_state_dict(state_dict, strict=True)
        print("✅ 仅加载模型参数（适合微调/再训练）")
        return model, optimizer

    def _init_by_pth_file(self, source_model_path, model, optimizer):
        checkpoint = torch.load(source_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        # state_dict = torch.load(source_path, map_location=device)
        if self.config.get("resume_optimizer", True) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("✅ 完整恢复：模型 + 优化器")
        else:
            print("✅ 仅加载模型参数（适合微调/再训练）")
        return model, optimizer
    def _init_model_optimizer(self, model, optimizer):
        # 如果指定了 source_model ：
        # 如果给了文件扩展名，就按照文件扩展名来加载 safetensors or pth,pt
        # 如果没给扩展名，
        # 判断有无 source_model_base_name.pth,pt 有的话从这个地方加载 model 和 optimizer
        # 如果没有，判断 source_model_base_name.safetensor 有的话从这个地方加载 model
        # 如果都没有，退化成首次训练
        if (source_model_name:=self.config.get("source_model_name", None)) is not None:
            ext = os.path.splitext(source_model_name)[-1].lower()
            if ext == ".safetensors":
                source_model_path = os.path.join(self.config["model_folder_path"], source_model_name)
                if os.path.exists(source_model_path):
                    model, optimizer = self._init_by_safetensors_file(source_model_path, model, optimizer)
                    return model, optimizer
            elif ext in [".pth", ".pt"]:
                source_model_path = os.path.join(self.config["model_folder_path"], source_model_name)
                if os.path.exists(source_model_path):
                    model, optimizer = self._init_by_pth_file(source_model_path, model, optimizer)
                    return model, optimizer
            else: # 未提供扩展名，优先按照 checkpoint 配置
                source_model_path = os.path.join(self.config["model_folder_path"], source_model_name+".pth")
                if os.path.exists(source_model_path): # pth
                    model, optimizer = self._init_by_pth_file(source_model_path, model, optimizer)
                    return model, optimizer
                source_model_path = os.path.join(self.config["model_folder_path"], source_model_name+".pt")
                if os.path.exists(source_model_path):
                    model, optimizer = self._init_by_pth_file(source_model_path, model, optimizer)
                    return model, optimizer
                source_model_path = os.path.join(self.config["model_folder_path"], source_model_name+".safetensors")
                if os.path.exists(source_model_path):
                    model, optimizer = self._init_by_safetensors_file(source_model_path, model, optimizer)
                    return model, optimizer
        print("从头开始训练")
        return model, optimizer

    def _train_test_split(self, dataset, train_set_rate=0.9, test_set_rate=0.1, seed=100):
        train_set, test_set = random_split(dataset,
                lengths = [train_set_rate, test_set_rate],
                generator=torch.Generator().manual_seed(seed),
            )
        return train_set, test_set

    def train(self):
        cls_ignore_index = -100
        dataset = MyDataset( self.config["x"], self.config["mask_x"], self.config["y"],  self.config["mask_y"], self.config["cls_idx"], self.config["reg_idx"])
        train_set, _ = self._train_test_split(dataset)
        train_loader = DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)
        # test_loader = DataLoader(test_set, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)

        model = MyModule(num_classes_list=self.config["num_classes_list"],
                        num_regression=len(self.config["reg_idx"])).to(self.device)

        # 损失函数
        cls_criterions = [nn.CrossEntropyLoss(ignore_index=cls_ignore_index) for _ in self.config["num_classes_list"]]
        reg_criterion = nn.MSELoss(reduction="none")  # 改为逐元素

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        # 如果配置了 source_model_path，就加载已有模型继续训练
        model, optimizer = self._init_model_optimizer(model, optimizer)

        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

        # 保存模型路路径
        model_folder_path = self.config["model_folder_path"]
        # folder_path = os.path.dirname(target_model_path)
        os.makedirs(model_folder_path, exist_ok=True)

        with tqdm(range(self.config["epochs"]), desc=f"Epoch 0/{self.config['epochs']} - Loss: NaN", total=self.config["epochs"]) as pbar:
            for epoch in pbar:
                model.train()
                total_loss = 0
                for x, mask_x, y_cls, y_reg, mask_cls, mask_reg in train_loader:
                    x, mask_x, y_cls, y_reg, mask_cls, mask_reg = x.to(self.device), mask_x.to(self.device), y_cls.to(self.device), y_reg.to(self.device), mask_cls.to(self.device), mask_reg.to(self.device)
                    optimizer.zero_grad()
                    cls_outs, reg_out = model(x, mask_x)

                    # loss_cls = sum(crit(out, y_cls[:, i]) for i, (crit, out) in enumerate(zip(cls_criterions, cls_outs)))
                    loss_cls = 0
                    for i, (crit, out ) in enumerate(zip(cls_criterions, cls_outs)): # 计算带掩码的分类损失
                        target = y_cls[:,i].clone()
                        target[mask_cls[:, i] == 0] = cls_ignore_index # 掩码为 0-> 忽略
                        loss_cls += crit(out, target) # TODO 似乎可以调整各项目损失的重要性权重，对于第0个损失是否有故障和故障分类，权重应该增大

                    # loss_reg = reg_criterion(reg_out, y_reg)
                    diff_reg = reg_criterion(reg_out, y_reg) # reg_out, y_reg ** 2
                    loss_reg = (diff_reg * mask_reg).sum()/ mask_reg.sum().clamp(min=1.0)

                    loss = loss_cls + self.config["lambda_reg"] * loss_reg
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                if (epoch + 1) % self.config["saving_for_every_cnt_epoches"] == 0:
                    checkpoint = dict(
                        model = model.state_dict(),
                        optimizer = optimizer.state_dict(),
                    )
                    torch.save(checkpoint, os.path.join(model_folder_path, f"{now_str}-{epoch+1:06d}.pth"))
                if avg_loss < self.config["loss_update_threshold"]:
                    pbar.set_description(f"Epoch {epoch + 1}/{self.config['epochs']} - Loss: {avg_loss:.4f}, less then threshold: {self.config['loss_update_threshold']:.4f}, break")
                    pbar.update(self.config['epochs'])
                    break
                else:
                    pbar.set_description(f"Epoch {epoch + 1}/{self.config['epochs']} - Loss: {avg_loss:.4f}")


        checkpoint = dict(
            model = model.state_dict(),
            optimizer = optimizer.state_dict(),
        )

        target_model_name = self.config["target_model_name"]
        basename, ext = os.path.splitext(target_model_name)
        # 保存 safetensors
        target_safetensors_path = os.path.join(model_folder_path, basename+".safetensors")
        save_file(model.state_dict(), target_safetensors_path)
        # 保存 checkpoint pth
        target_pth_path = os.path.join(model_folder_path, basename+".pth")
        torch.save(checkpoint, target_pth_path)
        print(f"模型已保存到 {target_safetensors_path} 和 {target_pth_path}")

    def test(self):
        cls_ignore_index = -100
        dataset = MyDataset( self.config["x"], self.config["mask_x"], self.config["y"],  self.config["mask_y"], self.config["cls_idx"], self.config["reg_idx"])
        _, test_set = self._train_test_split(dataset)
        test_loader = DataLoader(test_set, batch_size=self.config["batch_size"], shuffle=False, num_workers=4)


        # 初始化模型
        model = MyModule(
            num_classes_list=self.config["num_classes_list"],
            num_regression=len(self.config["reg_idx"])
        ).to(self.device)

        # 加载测试用模型
        model_name = self.config["testing_model_name"]
        model_path_pth = os.path.join(self.config["model_folder_path"], model_name + ".pth")
        model_path_safetensors = os.path.join(self.config["model_folder_path"], model_name + ".safetensors")
        if os.path.exists(model_path_pth):
            checkpoint = torch.load(model_path_pth, map_location=self.device)
            model.load_state_dict(checkpoint["model"], strict=True)
            print(f"✅ 从 {model_path_pth} 加载模型（含优化器状态，但这里不需要）")
        elif os.path.exists(model_path_safetensors):
            state_dict = load_file(model_path_safetensors)
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ 从 {model_path_safetensors} 加载模型")
        else:
            raise FileNotFoundError(f"没有找到 {model_name}.pth 或 {model_name}.safetensors")

        model.eval()

        total_correct = [0 for _ in self.config["num_classes_list"]]
        total_samples = [0 for _ in self.config["num_classes_list"]]
        total_reg_loss = 0.0
        total_reg_count = 0

        reg_criterion = nn.MSELoss(reduction="none")

        with torch.no_grad():
            for x, mask_x, y_cls, y_reg, mask_cls, mask_reg in test_loader:
                x, mask_x, y_cls, y_reg, mask_cls, mask_reg = (
                    x.to(self.device), mask_x.to(self.device),
                    y_cls.to(self.device), y_reg.to(self.device),
                    mask_cls.to(self.device), mask_reg.to(self.device)
                )

                cls_outs, reg_out = model(x, mask_x)

                # 分类准确率
                for i, out in enumerate(cls_outs):
                    preds = out.argmax(dim=-1)   # [bs]
                    valid_mask = (mask_cls[:, i] == 1)
                    total_correct[i] += (preds[valid_mask] == y_cls[:, i][valid_mask]).sum().item()
                    total_samples[i] += valid_mask.sum().item()

                # 回归 MSE
                diff_reg = reg_criterion(reg_out, y_reg)  # [bs, num_reg]
                masked_diff = diff_reg * mask_reg
                total_reg_loss += masked_diff.sum().item()
                total_reg_count += mask_reg.sum().item()

        # 汇总
        for i, n_cls in enumerate(self.config["num_classes_list"]):
            acc = total_correct[i] / max(1, total_samples[i])
            print(f"分类任务 {i} (num_classes={n_cls}): 准确率 = {acc:.4f}")

        if total_reg_count > 0:
            mse = total_reg_loss / total_reg_count
            print(f"回归任务 MSE = {mse:.6f}")
        else:
            print("⚠️ 测试集中没有有效的回归标签")

if __name__ == "__main__":
    import json5 as json
    parser = ArgumentParser(description="")
    parser.add_argument("mode", type=str, choices=["train", "test"], )

    args = parser.parse_args()
    cfg_path = os.path.join(os.path.dirname(__file__), "a0005.json5")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if args.mode == "train":
        TrainTester(config).train()
    else:
        TrainTester(config).test()