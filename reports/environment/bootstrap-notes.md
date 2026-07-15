# `brain` 环境初始化记录

> 日期：2026-07-15
>
> 平台：macOS 26.2 / Apple M4 / arm64
>
> 状态：本机验收通过

## 环境策略与实际偏差

`environment.yml` 保持项目批准的最小、跨平台 `defaults` 规格：Python 3.12、pip、setuptools 和 wheel。本机执行时，完整 Anaconda repodata 下载连续无进展，因此使用现有 Conda 缓存离线创建；缓存中的可用构建来自 conda-forge。实际来源已完整记录在 `osx-arm64-conda.explicit.txt`，没有伪装成 defaults 结果。

PyTorch 2.12.1 与 NumPy 2.5.1 使用 PyPI 官方 macOS arm64 wheel。下载采用 HTTP Range 分片，合并后在安装前核对：

- PyTorch SHA-256：`d2dd0f2c5f7ccbddaf34cade0deaf476808368f902b9cdb7f36a2ab42301bc0e`
- NumPy SHA-256：`78798bd5b9ad744056af8efa90e3b9ddaa53272a0848a483084a1cc0a13b2dc0`

首次混合安装暴露 Conda OpenBLAS/LLVM OpenMP 与官方 PyTorch wheel 的双 OpenMP 冲突。已移除 Conda NumPy、OpenBLAS、LLVM OpenMP 栈并改用官方 NumPy wheel；没有设置不安全的 `KMP_DUPLICATE_LIB_OK`。

## 复现顺序

1. 常规跨平台开发使用 `make env`，以 `environment.yml` 为声明来源。
2. 精确复现当前 Apple Silicon 环境时，先使用 `locks/osx-arm64-conda.explicit.txt` 创建 Conda 环境。
3. 再安装 `locks/osx-arm64-pip.txt`；`-e .` 必须从仓库根目录执行。
4. 运行 `make check` 与 `make smoke-mps`。

Linux/CUDA 锁文件必须在真实 Linux/CUDA 主机上生成，本次没有伪造。
