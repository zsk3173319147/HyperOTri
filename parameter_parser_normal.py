import argparse
import os.path as osp


PROJECT_ROOT = osp.abspath(osp.dirname(__file__))
DEFAULT_DATA_ROOT = osp.join(PROJECT_ROOT, "data")
DEFAULT_EDGE_SAVE_DIR = osp.join(DEFAULT_DATA_ROOT, "edge_splits")

SINGLE_DATASETS = (
    "cora",
    "citeseer",
    "pubmed",
    "coauthor_cora",
    "coauthor_dblp",
    "20newsW100",
    "ModelNet40",
    "NTU2012",
    "Mushroom",
    "zoo",
)

MULTI_DATASETS = (
    "RHG_3",
    "RHG_10",
    "RHG_table",
    "RHG_pyramid",
    "IMDB_dir_form",
    "IMDB_dir_genre",
    "IMDB_wri_form",
    "IMDB_wri_genre",
)

TASKS = ("node_cls", "edge_pred", "hg_cls")

METHODS = (
    "HGNN",
    "HNHN",
    "HyperGCN",
    "AllDeepSets",
    "AllSetformer",
    "UniGIN",
)


def _add_bool_argument(group, option, dest, default=False, help=None):
    """同时生成 --option 和 --no-option。"""
    group.add_argument(
        f"--{option}",
        dest=dest,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help,
    )


def set_task_args(args):
    """检查参数组合，并转换成现有模型代码使用的字段名。"""

    # 现有 Encoder 仍使用这些旧字段名，统一在这里映射。
    args.embedding_hidden = args.hidden_dim
    args.MLP_hidden = args.hidden_dim
    args.All_num_layers = args.num_layers

    # 三种预测头结构相同，只是现有代码中的字段名前缀不同。
    if args.task_type == "node_cls":
        args.n_embed_hidden = args.head_hidden
        args.n_embed_layer = args.head_layers
        args.n_embed_dropout = args.head_dropout
        args.n_embed_norm = args.head_norm
    elif args.task_type == "edge_pred":
        args.e_embed_hidden = args.head_hidden
        args.e_embed_layer = args.head_layers
        args.e_embed_dropout = args.head_dropout
        args.e_embed_norm = args.head_norm
        args.edge_split_mode = "trand"
        args.edge_aggr = "group"
    elif args.task_type == "hg_cls":
        args.g_embed_hidden = args.head_hidden
        args.g_embed_layer = args.head_layers
        args.g_embed_dropout = args.head_dropout
        args.g_embed_norm = args.head_norm
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}")

    if args.task_type in ("node_cls", "edge_pred"):
        if args.dname not in SINGLE_DATASETS:
            raise ValueError(
                f"{args.dname} is not a single-hypergraph dataset for "
                f"{args.task_type}"
            )
    elif args.dname not in MULTI_DATASETS:
        raise ValueError(
            f"{args.dname} is not a multi-hypergraph dataset for hg_cls"
        )

    if args.runs <= 0 or args.epochs <= 0:
        raise ValueError("runs and epochs must be positive")
    if args.hidden_dim <= 0 or args.num_layers <= 0:
        raise ValueError("hidden_dim and num_layers must be positive")
    if args.head_hidden <= 0 or args.head_layers <= 0:
        raise ValueError("head_hidden and head_layers must be positive")
    if args.train_prop <= 0 or args.valid_prop < 0:
        raise ValueError("train_ratio must be positive and val_ratio non-negative")
    if args.train_prop + args.valid_prop >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1")
    if args.task_type == "edge_pred" and args.edge_batch_size <= 0:
        raise ValueError("edge_batch_size must be positive")
    if args.task_type == "hg_cls" and args.hg_batch_size <= 0:
        raise ValueError("graph_batch_size must be positive")

    # 节点分类和超边预测默认添加单节点超边；超图分类默认不添加。
    if args.add_self_loop is None:
        args.add_self_loop = args.task_type in ("node_cls", "edge_pred")

    # AllSetformer 通过 PMA=True 区分于 AllDeepSets。
    args.PMA = args.method == "AllSetformer"
    return args


def parameter_parser(argv=None):
    parser = argparse.ArgumentParser(
        description="训练和评估超图模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # 1. 基础参数：所有 Encoder 和任务共同使用
    # ------------------------------------------------------------------
    basic = parser.add_argument_group("1. 基础参数")
    basic.add_argument(
        "--task",
        dest="task_type",
        choices=TASKS,
        default="node_cls",
        help="任务类型：节点分类、超边预测或超图分类。",
    )
    basic.add_argument(
        "--dataset",
        dest="dname",
        choices=(*SINGLE_DATASETS, *MULTI_DATASETS),
        default="cora",
        help="数据集名称；超图分类需要选择 RHG 或 IMDB 数据集。",
    )
    basic.add_argument(
        "--encoder",
        dest="method",
        choices=METHODS,
        default="HGNN",
        help="使用的超图 Encoder。",
    )
    basic.add_argument(
        "--device",
        default="cuda:0",
        help="运行设备，例如 cpu、cuda:0 或 cuda:2。",
    )
    basic.add_argument("--seed", type=int, default=0, help="第一个随机种子。")
    basic.add_argument(
        "--runs",
        type=int,
        default=1,
        help="连续运行的随机种子数量，从 seed 开始递增。",
    )
    basic.add_argument(
        "--data-root",
        dest="data_root",
        default=DEFAULT_DATA_ROOT,
        help="数据集根目录。",
    )

    # 训练参数
    basic.add_argument("--epochs", type=int, default=100, help="训练轮数。")
    basic.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率。")
    basic.add_argument(
        "--weight-decay",
        dest="wd",
        type=float,
        default=0.0,
        help="Adam 权重衰减。",
    )
    basic.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Encoder 内部的 dropout 概率。",
    )
    basic.add_argument(
        "--eval-every",
        dest="display_step",
        type=int,
        default=20,
        help="每隔多少个 epoch 打印一次评估结果。",
    )
    # 数据划分和通用 Encoder 参数
    basic.add_argument(
        "--train-ratio",
        dest="train_prop",
        type=float,
        default=0.6,
        help="训练集比例。",
    )
    basic.add_argument(
        "--val-ratio",
        dest="valid_prop",
        type=float,
        default=0.2,
        help="验证集比例，剩余数据作为测试集。",
    )
    _add_bool_argument(
        basic,
        "self-loops",
        "add_self_loop",
        default=None,
        help="是否添加单节点超边；节点分类和超边预测默认开启。",
    )
    basic.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Encoder 隐藏表示和最终嵌入的维度。",
    )
    basic.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Encoder 的消息传递层数。",
    )

    # ------------------------------------------------------------------
    # 2. 特殊 Encoder 参数：只有选择对应 Encoder 时才会生效
    # ------------------------------------------------------------------
    hnhn = parser.add_argument_group("2.1 HNHN 特殊参数")
    _add_bool_argument(
        hnhn,
        "hnhn-nonlinear-between",
        "HNHN_nonlinear_inbetween",
        default=True,
        help="V2E 和 E2V 传播之间是否加入非线性。",
    )
    hnhn.add_argument(
        "--hnhn-alpha",
        dest="HNHN_alpha",
        type=float,
        default=-0.5,
        help="HNHN 超边度归一化指数。",
    )
    hnhn.add_argument(
        "--hnhn-beta",
        dest="HNHN_beta",
        type=float,
        default=-0.5,
        help="HNHN 节点度归一化指数。",
    )

    hypergcn = parser.add_argument_group("2.2 HyperGCN 特殊参数")
    _add_bool_argument(
        hypergcn,
        "hypergcn-fast",
        "HyperGCN_fast",
        default=True,
        help="是否复用预计算的近似图结构。",
    )
    _add_bool_argument(
        hypergcn,
        "hypergcn-mediators",
        "HyperGCN_mediators",
        default=True,
        help="构图时是否加入 mediator 节点。",
    )

    allset = parser.add_argument_group("2.3 AllSet 特殊参数")
    allset.add_argument(
        "--allset-aggregation",
        dest="aggregate",
        choices=("add", "mean"),
        default="add",
        help="AllSet 消息聚合方式。",
    )
    allset.add_argument(
        "--allset-mlp-layers",
        dest="MLP_num_layers",
        type=int,
        default=2,
        help="每个 AllSet 卷积内部的 MLP 层数。",
    )
    allset.add_argument(
        "--allset-normalization",
        dest="normalization",
        choices=("bn", "ln", "None"),
        default="ln",
        help="AllSet 内部 MLP 的归一化。",
    )
    allset.add_argument(
        "--incidence-normalization",
        dest="normtype",
        choices=("all_one", "deg_half_sym"),
        default="all_one",
        help="AllSet incidence message 的归一化方式。",
    )
    _add_bool_argument(
        allset,
        "allset-input-normalization",
        "deepset_input_norm",
        default=False,
        help="是否归一化 DeepSets MLP 的输入。",
    )
    _add_bool_argument(allset, "allset-gpr", "GPR", default=False)
    _add_bool_argument(allset, "allset-learn-mask", "LearnMask", default=False)
    _add_bool_argument(allset, "allset-exclude-self", "exclude_self", default=False)
    allset.add_argument("--allset-decoder-hidden", dest="decoder_hidden", type=int, default=128)
    allset.add_argument("--allset-decoder-layers", dest="decoder_num_layers", type=int, default=1)
    allset.add_argument("--allset-heads", dest="heads", type=int, default=1)

    unigin = parser.add_argument_group("2.4 UniGIN 特殊参数")
    unigin.add_argument("--unigin-heads", dest="uni_heads", type=int, default=1)
    unigin.add_argument(
        "--unigin-activation",
        dest="activation",
        choices=("relu", "prelu"),
        default="relu",
    )
    unigin.add_argument("--unigin-input-dropout", dest="input_drop", type=float, default=0.0)
    unigin.add_argument("--unigin-attention-dropout", dest="attn_drop", type=float, default=0.0)
    unigin.add_argument(
        "--unigin-first-aggregation",
        dest="first_aggregate",
        choices=("sum", "mean"),
        default="mean",
    )
    _add_bool_argument(unigin, "unigin-normalize", "use_norm", default=False)

    # ------------------------------------------------------------------
    # 3. 任务参数
    # ------------------------------------------------------------------
    task = parser.add_argument_group("3.1 通用预测头参数")
    task.add_argument(
        "--head-hidden",
        type=int,
        default=128,
        help="预测头隐藏维度；head-layers=1 时不会使用。",
    )
    task.add_argument(
        "--head-layers",
        type=int,
        default=1,
        help="预测头层数；设为 1 时就是一层 Linear。",
    )
    task.add_argument(
        "--head-dropout",
        type=float,
        default=0.2,
        help="预测头内部的 dropout。",
    )
    task.add_argument(
        "--head-normalization",
        dest="head_norm",
        choices=("bn", "ln", "None"),
        default="ln",
        help="预测头内部的归一化方式。",
    )

    edge = parser.add_argument_group("3.2 超边预测参数")
    edge.add_argument(
        "--edge-batch-size",
        dest="edge_batch_size",
        type=int,
        default=8,
        help="每批候选超边数；超过样本总数时自动使用全部样本。",
    )
    edge.add_argument(
        "--edge-pooling",
        dest="aggr_mode",
        choices=("max", "mean", "maxmin"),
        default="max",
        help="将候选超边中的节点嵌入聚合为超边表示。",
    )
    edge.add_argument(
        "--negative-sampling",
        dest="ns_method",
        choices=("mns", "sns", "cns", "mixed"),
        default="mixed",
        help="训练负超边的采样方法。",
    )
    edge.add_argument(
        "--edge-save-dir",
        default=DEFAULT_EDGE_SAVE_DIR,
        help="超边划分文件的缓存目录。",
    )

    graph = parser.add_argument_group("3.3 超图分类参数")
    graph.add_argument(
        "--graph-batch-size",
        dest="hg_batch_size",
        type=int,
        default=32,
        help="每批独立超图数；超过数据集大小时自动使用全部超图。",
    )
    graph.add_argument(
        "--pooling",
        choices=("mean", "max"),
        default="max",
        help="将一张超图的节点嵌入聚合为图级表示。",
    )
    _add_bool_argument(
        graph,
        "weighted-loss",
        "use_weighted_loss",
        default=False,
        help="是否按照训练集类别频率设置交叉熵权重。",
    )

    args = parser.parse_args(argv)
    return set_task_args(args)

