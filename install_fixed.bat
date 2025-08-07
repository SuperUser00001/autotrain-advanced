@echo off
echo 正在安装 AutoTrain Advanced 修复版本...
echo.

echo 步骤 1: 降级 NumPy 到兼容版本
pip install "numpy<2.0.0"

echo.
echo 步骤 2: 安装其他核心依赖
pip install torch transformers datasets accelerate peft trl bitsandbytes

echo.
echo 步骤 3: 安装 AutoTrain Advanced
pip install -e .

echo.
echo 安装完成！
echo 现在可以运行: autotrain --config configs/test_smollm2.yml
pause 