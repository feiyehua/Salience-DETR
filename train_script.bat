@echo off
REM 设置CUDA相关环境变量
SET "CUDA_HOME=%~dp0cuda_home_placeholder"
SET "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REM 创建临时CUDA目录结构
IF NOT EXIST "%CUDA_HOME%" (
    mkdir "%CUDA_HOME%\bin"
    mkdir "%CUDA_HOME%\include"
    mkdir "%CUDA_HOME%\lib64"
    echo 创建了临时CUDA目录结构
)

echo CUDA_HOME设置为: %CUDA_HOME%
echo 启动训练...

REM 启动训练
accelerate launch --mixed_precision fp16 --num_processes 1 main.py --config-file configs/train_config.py --mixed-precision fp16

echo 训练命令执行完毕 