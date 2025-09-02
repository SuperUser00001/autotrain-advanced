import os
import torch
import numpy as np
from safetensors.torch import load_file
import json5 as json
from a0005 import MyModule, MyDataset  # 直接复用训练脚本的模型定义
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error



def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 临时使用训练集的 data_x 作为测试集
    x_path = os.path.join("datasets", "data_x.npy")
    y_path = os.path.join("datasets", "data_y.npy")

    # 数据加载
    X = np.load(x_path)
    Y_true = np.load(y_path)

    # 模型加载
    model = MyModule(
        num_classes_list=config["num_classes_list"],
        num_regression=len(config["reg_idx"]),
    ).to(device)
    model_folder_path = config["model_folder_path"]
    target_model_name = config["target_model_name"]
    basename, ext = os.path.splitext(target_model_name)
    target_safetensors_path = os.path.join(model_folder_path, basename+".safetensors")
    if not os.path.exists(target_safetensors_path): raise Exception(f"model does not exist: {target_safetensors_path}")
    state_dict = load_file(target_safetensors_path)
    model.load_state_dict(state_dict)
    model.eval()

    # 推理
    preds_cls_all = []
    preds_reg_all = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).to(device)
            cls_outs, reg_out = model(x)

            # 分类取 argmax
            cls_preds = [torch.argmax(out, dim=1).cpu().numpy()[0] for out in cls_outs]
            preds_cls_all.append(cls_preds)

            # 回归直接输出
            preds_reg_all.append(reg_out.cpu().numpy()[0])

    # 拼接成 (N, C+R) 的数组
    preds_cls_all = np.array(preds_cls_all)  # (N, num_class_heads)
    preds_reg_all = np.array(preds_reg_all)  # (N, num_regression)
    preds_all = np.concatenate([preds_cls_all, preds_reg_all], axis=1)

    # 保存预测结果
    # np.save(os.path.join("datasets", "test_y.npy"), preds_all)
    # print("预测结果已保存到 datasets/test_y.npy")

    # 评估
    dataset = ( config["x"], config["mask_x"], config["y"],  config["mask_y"], config["cls_idx"], config["reg_idx"])
    cls_idx = dataset.cls_idx
    reg_idx = dataset.reg_idx

    # 分类评估（逐个头计算准确率）
    for i, idx in enumerate(cls_idx):
        acc = accuracy_score(Y_true[:, idx], preds_cls_all[:, i])
        print(f"分类头{i} (原y列 {idx}) 准确率: {acc:.4f}")

    # 回归评估（逐个头计算MSE）
    for i, idx in enumerate(reg_idx):
        mse = mean_squared_error(Y_true[:, idx], preds_reg_all[:, i])
        print(f"回归头{i} (原y列 {idx}) MSE: {mse:.4f}")


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "a0001.json5")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    test_model(config)