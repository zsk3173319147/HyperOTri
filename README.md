# HMTL

超图多任务学习项目。主方法使用共享超图编码器、任务 Adapter 和统一预测头，联合训练以下任务：

- `node_cls`：节点分类。
- `edge_pred`：超边预测。
- `hg_cls`：超图分类。


## 项目结构

```text
HMTL/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                         # 被多任务入口引用的编码器、数据准备等函数
├── parameter_parser_normal.py     # 底层任务和编码器参数
├── config/
│   └── multitask_unified.yaml      # 主方法配置
├── HNN/                           # 超图编码器及配套实现
├── MTL/                           # Adapter、联合训练、子超图和任务样本
├── dataset/                       # 数据读取、转换、预处理和划分
├── utils/                         # 统一预测头、聚合和其他必要依赖
├── tests/
│   ├── test_multitask_unified.py   # 主方法多任务训练入口
│   └── test_multitask.py           # 主入口依赖的数据加载器和指标函数
└── data/
    └── README.md                  # 数据准备说明；数据文件另行准备
```

## 安装

本次运行验证使用 Linux、Python 3.9.25、PyTorch 2.5.1（CUDA 12.4 构建）和 PyG 2.6.1。依赖版本列在 `requirements.txt` 中。

创建独立环境：

```bash
conda create -n hmtl python=3.9 -y
conda activate hmtl
```

使用支持 CUDA 12.4 的 NVIDIA GPU 时，先安装 PyTorch 和匹配的扩展：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
python -m pip install -r requirements.txt
```

## 数据准备

运行前按 [data/README.md](data/README.md) 放置原始数据。代码会读取本地文件并生成缓存，不会联网下载数据。

代码发布包只包含源码和说明，不包含数据集。原始工作区的数据保留在本地；使用发布包时，将所需数据复制到新目录的 `data/` 下。

## 多任务运行

从项目根目录执行：

```bash
python tests/test_multitask_unified.py --config config/multitask_unified.yaml --device cuda:0
```

原配置包含 `NTU2012` 节点分类、`citeseer` 超边预测、`RHG_3` 超图分类，默认训练 100 个 epoch、运行 5 个随机种子。配置中的设备为 `cuda:2`，可用命令行参数覆盖。

短运行示例：

```bash
python tests/test_multitask_unified.py \
  --config config/multitask_unified.yaml \
  --node-dataset coauthor_cora --edge-dataset coauthor_cora --graph-dataset RHG_10 \
  --device cpu --epochs 1 --runs 1 --eval-every 1 \
  --hidden-dim 16 --feature-dim 16 --adapter-hidden 4
```

该命令运行三个任务的联合训练及训练集、验证集、测试集评估，适合检查环境和数据链路。它不用于报告正式性能。

查看命令行参数：

```bash
python tests/test_multitask_unified.py --help
```

