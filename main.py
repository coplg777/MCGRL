import time
import argparse
import json
import random
import time
import numpy as np
from tqdm import tqdm, trange
from functools import reduce

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup


from eval import eval_metrics

from data import load_data, num_graphs
from my_utils import save_results



parser = argparse.ArgumentParser(description='Cluster_GT')
parser.add_argument('--config', default=None, type=str,
                    help='optional JSON config file; keys override parsed args')


parser.add_argument('--data', default='IMDB-BINARY', type=str,
                    choices=['DD', 'NCI1', 'PROTEINS', 'IMDB-BINARY',
                             'IMDB-MULTI', 'MUTAG', 'COLLAB'],
                    help='dataset type')


parser.add_argument("--model", type=str,
                    default='Cluster_GT', choices=['Cluster_GT'])
parser.add_argument("--model-string", type=str, default='Cluster_GT')


parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument("--grad-norm", type=float, default=1.0)
parser.add_argument("--lr-schedule", action='store_true')
parser.add_argument("--normalize", action='store_true')
parser.add_argument('--num-epochs', default=300,
                    type=int, help='train epochs number')



parser.add_argument('--patience', type=int, default=30,
                    help='patience for earlystopping')



parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')
parser.add_argument('--batch-size', default=64,
                    type=int, help='train batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight-decay', type=float,
                    default=0.00001, help='weight decay')
parser.add_argument("--dropout", type=float, default=0.)


parser.add_argument("--gpu", type=int, default=1)


parser.add_argument("--use-gnn", action='store_true')
parser.add_argument("--conv", type=str, default='GIN', choices=['GCN', 'GIN'])
parser.add_argument("--num-convs", type=int, default=1)


parser.add_argument("--online", action='store_true')
parser.add_argument("--layernorm", action='store_true')
parser.add_argument("--remain-k1", action='store_true')
parser.add_argument("--diffQ", action='store_true')
parser.add_argument("--residual", type=str, default='cat',
                    choices=['None', 'cat', 'sum'])
parser.add_argument("--kernel_method", type=str,
                    default='elu', choices=['relu', 'elu'])
parser.add_argument("--deepset-layers", type=int, default=2)
parser.add_argument("--pos-enc-rw-dim", type=int, default=8)
parser.add_argument("--pos-enc-lap-dim", type=int, default=0)
parser.add_argument("--n-patches", type=int, default=8)
parser.add_argument("--prop-w-norm-on-coarsened", action='store_true')
parser.add_argument("--pos-enc-patch-rw-dim", type=int, default=0)
parser.add_argument("--pos-enc-patch-num-diff", type=int, default=-1)
parser.add_argument("--attention-based-readout", action='store_true')



parser.add_argument("--static_weights", action='store_true',
                    help="使用静态固定权重，否则使用动态权重（默认）")


parser.add_argument("--divergence_type", type=str, default='js',
                    choices=['js', 'kl'],
                    help="选择散度类型：js=封闭高斯JS散度，kl=简单KL散度")


parser.add_argument("--lambda_intra", type=float, default=0.6,
                    help="λ₁: 内部连接奖励权重 (推荐0.6)")
parser.add_argument("--lambda_balance", type=float, default=0.3,
                    help="λ₂: 平衡奖励权重 (推荐0.3)")


parser.add_argument("--ablate_R_intra", action='store_true',
                    help="CGSO消融：移除内部连接奖励(R_intra)")
parser.add_argument("--ablate_R_balance", action='store_true',
                    help="CGSO消融：移除平衡奖励(R_balance)")
parser.add_argument("--ablate_R_comm", action='store_true',
                    help="CGSO消融：移除通信奖励(R_comm)")
parser.add_argument("--ablate_policy_network", action='store_true',
                    help="CGSO消融：移除策略网络，使用随机分区")
parser.add_argument("--ablate_value_network", action='store_true',
                    help="CGSO消融：移除价值网络，使用简化奖励")


parser.add_argument("--no_type_embedding", action='store_true',
                    help="UAF消融：移除类型感知嵌入")
parser.add_argument("--single_head_only", action='store_true',
                    help="UAF消融：使用单头注意力而非多头")
parser.add_argument("--no_residual", action='store_true',
                    help="UAF消融：移除残差连接")
parser.add_argument("--no_layernorm", action='store_true',
                    help="UAF消融：移除层归一化")
parser.add_argument("--self_attention_only", action='store_true',
                    help="UAF消融：仅使用自注意力，不使用跨注意力")



parser.add_argument('--num_attention_heads', type=int, default=4,
                    help='多头注意力的头数')
parser.add_argument('--attention_dropout',   type=float, default=0.0,
                    help='MultiheadAttention 的 dropout')
parser.add_argument('--attention_bias',      action='store_true',
                    help='是否在 attention 中使用 bias')
parser.add_argument('--attention_add_bias_kv',  action='store_true')
parser.add_argument('--attention_add_zero_attn', action='store_true')



args = parser.parse_args()

if args.config:
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    for key, value in config.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(args, key, value)
    print(f"[INFO] Loaded config: {args.config}")


if args.static_weights:
    print("[INFO] Using static weights")
else:
    print("[INFO] Using dynamic weights (default)")

print(f"[INFO] RL partition enabled by default")
print(f"[INFO] Divergence type: {args.divergence_type.upper()} ({'封闭高斯JS散度' if args.divergence_type == 'js' else '简单KL散度'})")


print(f"[INFO] Loss weights - λ₁ (intra): {args.lambda_intra}, λ₂ (balance): {args.lambda_balance}")
if args.lambda_intra != 0.6 or args.lambda_balance != 0.3:
    print("[INFO] Using custom loss weights for parameter sensitivity analysis")
else:
    print("[INFO] Using default loss weights")

print("[INFO] Full model: 包含完整UAF模块 (使用Cluster_GT_N2C_T_Q)")


cgso_ablations = []
if args.ablate_R_intra:
    cgso_ablations.append("R_intra")
if args.ablate_R_balance:
    cgso_ablations.append("R_balance")
if args.ablate_R_comm:
    cgso_ablations.append("R_comm")
if args.ablate_policy_network:
    cgso_ablations.append("Policy Network")
if args.ablate_value_network:
    cgso_ablations.append("Value Network")

if cgso_ablations:
    print(f"[CGSO ABLATION] 移除组件: {', '.join(cgso_ablations)}")
else:
    print("[CGSO] 使用完整CGSO模块")


uaf_ablations = []
if args.no_type_embedding:
    uaf_ablations.append("Type Embedding")
if args.single_head_only:
    uaf_ablations.append("Multi-head")
if args.no_residual:
    uaf_ablations.append("Residual")
if args.no_layernorm:
    uaf_ablations.append("LayerNorm")
if args.self_attention_only:
    uaf_ablations.append("Cross-attention")

if uaf_ablations:
    print(f"[UAF ABLATION] 移除组件: {', '.join(uaf_ablations)}")
else:
    print("[UAF] 使用完整UAF模块")

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = args.gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

torch.set_num_threads(1)

dataset = load_data(args)

print(f"Dataset: {args.data}")
print(f"Input transfrom: {'GNN' if args.use_gnn else 'MLP'}")
print(f'Metis Online: {args.online}')
print(f"Model: {args.model}")
print(f"Device: {args.device}")

overall_results = {
    'best_val_loss': [],
    'best_val_acc': [],
    'best_val_f1_macro': [],
    'best_test_loss': [],
    'best_test_acc': [],
    'best_test_f1_macro': [],
    'durations': []
}

train_fold_iter = tqdm(range(1, 11), desc='Training')
val_fold_iter = [i for i in range(1, 11)]


for fold_number in train_fold_iter:
    val_fold_number = val_fold_iter[fold_number - 2]

    train_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/train_idx-%d.txt' % (args.data, fold_number),
                                             dtype=np.int32), dtype=torch.long)
    val_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (args.data, val_fold_number),
                                           dtype=np.int32), dtype=torch.long)
    test_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (args.data, fold_number),
                                            dtype=np.int32), dtype=torch.long)

    all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
    assert len(all_idxes) == len(dataset)

    train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))

    train_set, val_set, test_set = dataset[train_idxes], dataset[val_idxes], dataset[test_idxes]

    if not args.online:
        train_set = [x for x in train_set]
    val_set = [x for x in val_set]
    test_set = [x for x in test_set]

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False)

    if args.model != 'Cluster_GT':
        raise ValueError("Model Name <{}> is Unknown".format(args.model))

    from model.classification.Cluster_GT_N2C_T_Q import Cluster_GT
    print("[FULL] Using Cluster_GT_N2C_T_Q (with UAF)")
    model = Cluster_GT(args)

    if use_cuda:
        model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_epochs//10, num_training_steps=args.num_epochs)

    patience = 0
    best_loss = 1e9
    best_val_acc = 0
    best_val_f1_macro = 0
    best_test_loss = 1e9
    best_test_acc = 0
    best_test_f1_macro = 0

    t_start = time.perf_counter()

    for epoch in trange(0, (args.num_epochs), desc='[Epoch]', position=1):
        model.train()
        total_loss = 0

        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()

            if args.lr_schedule:
                scheduler.step()

        total_loss = total_loss / len(train_loader.dataset)

        val_metrics = eval_metrics(val_loader, model, args)
        test_metrics = eval_metrics(test_loader, model, args)
        val_acc = val_metrics['acc']
        val_loss = val_metrics['loss']
        val_f1_macro = val_metrics['f1_macro']
        test_acc = test_metrics['acc']
        test_loss = test_metrics['loss']
        test_f1_macro = test_metrics['f1_macro']

        if val_loss < best_loss:
            best_loss = val_loss
            best_val_acc = val_acc
            best_val_f1_macro = val_f1_macro
            best_val_epoch = epoch
            patience = 0

            best_test_acc = test_acc
            best_test_f1_macro = test_f1_macro
            best_test_loss = test_loss
        else:
            patience += 1







        train_fold_iter.set_description('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f VaF1: %.2f TestAcc: %.2f TestF1: %.2f' % (
            fold_number, epoch, total_loss, val_loss, val_acc, val_f1_macro, test_acc, test_f1_macro))
        train_fold_iter.refresh()

        if args.patience > 0 and patience > args.patience:
            break




    t_end = time.perf_counter()

    overall_results['durations'].append(t_end - t_start)
    overall_results['best_val_loss'].append(best_loss)
    overall_results['best_val_acc'].append(best_val_acc)
    overall_results['best_val_f1_macro'].append(best_val_f1_macro)
    overall_results['best_test_loss'].append(best_test_loss)
    overall_results['best_test_acc'].append(best_test_acc)
    overall_results['best_test_f1_macro'].append(best_test_f1_macro)

    print("[Test: Fold {}] Test Acc: {} Test Macro-F1: {} with Time: {}".format(
        fold_number, best_test_acc, best_test_f1_macro, (t_end - t_start)))

print("Overall result - overall_best_val: {} with std: {}; overall_best_test: {} with std: {}; overall_best_test_f1_macro: {} with std: {}\n".format(
    np.array(overall_results['best_val_acc']).mean(),
    np.array(overall_results['best_val_acc']).std(),
    np.array(overall_results['best_test_acc']).mean(),
    np.array(overall_results['best_test_acc']).std(),
    np.array(overall_results['best_test_f1_macro']).mean(),
    np.array(overall_results['best_test_f1_macro']).std()
))


save_results(args,overall_results)
