#!/usr/bin/env python
"""
eval_checkpoint.py
==================
Danh gia chat luong cua bat ky checkpoint nao (epoch3 hoac retrain).

Cach dung tren Kaggle:
    python eval_checkpoint.py --checkpoint /kaggle/working/TAGAPT_retrain/exp_ASG_CTI_retrain/checkpoint
    python eval_checkpoint.py --checkpoint /path/to/save_pretrain/exp_ASG_CTI_epoch3_1gpu/checkpoint

Cach dung local (can GPU):
    python eval_checkpoint.py --checkpoint ./save_pretrain/exp_ASG_CTI_epoch3_1gpu/checkpoint --no_cuda

Metrics duoc do:
    - UK rate  : ti le edge co verb 'UK' (unknown) - THAP = tot
    - SO rate  : ti le graph co node mang (Socket) - CAO = tot
    - FR hub   : muc do mot process fork qua nhieu child - THAP = tot
    - Avg nodes: trung binh so node moi graph
    - Unique%  : ti le graph unique (khong bi trung lap)
"""

import os, sys, re, json, argparse
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)


def _patch_cpu_mode():
    """
    model_CTI.py hardcodes .cuda() calls throughout generate().
    When no GPU is available, patch torch so .cuda() is a no-op
    returning the tensor/module on CPU unchanged.
    """
    import torch
    if torch.cuda.is_available():
        return  # GPU present, no patch needed

    # Patch Tensor.cuda
    _orig_tensor_cuda = torch.Tensor.cuda
    def _cpu_tensor_cuda(self, *args, **kwargs):
        return self  # stay on CPU
    torch.Tensor.cuda = _cpu_tensor_cuda

    # Patch nn.Module.cuda
    import torch.nn as nn
    _orig_module_cuda = nn.Module.cuda
    def _cpu_module_cuda(self, *args, **kwargs):
        return self  # stay on CPU
    nn.Module.cuda = _cpu_module_cuda

    print("  [CPU mode] .cuda() patched to no-op (no GPU available)")


def load_checkpoint_and_model(checkpoint_path, no_cuda=False):
    import torch
    from model_CTI import GraphFlowModel

    # Apply CPU patch FIRST before any model code touches .cuda()
    if no_cuda or not torch.cuda.is_available():
        _patch_cpu_mode()

    # Xac dinh config file:
    # 1. Tim config.json cung thu muc voi checkpoint
    # 2. Fallback: tim config.json trong cung thu muc voi script nay
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        # Fallback: dung config.json cua project (neu checkpoint nam o noi khac)
        config_path = os.path.join(BASE, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError("Khong tim thay config.json tai: " + config_path)

    cfg = json.load(open(config_path))
    print("  Config: epochs=" + str(cfg.get("epochs")) +
          " | num_flow_layer=" + str(cfg.get("num_flow_layer")) +
          " | temperature=" + str(cfg.get("temperature")))

    # Load data config de biet max_size, node_dim, bond_dim
    data_prefix = os.path.join(BASE, "data_preprocessed", "CTI")
    data_config = eval(open(data_prefix + "_config.txt").read())
    max_size  = data_config["max_size"]
    node_dim  = data_config["node_dim"] - 1
    bond_dim  = data_config["bond_dim"]
    edge_unroll = cfg.get("edge_unroll", 10)

    print("  Graph config: max_size=" + str(max_size) +
          " node_dim=" + str(node_dim) + " bond_dim=" + str(bond_dim))

    # Build model
    import argparse as _ap
    args = _ap.Namespace(**cfg)
    for attr in ["penalty","reward_type","moving_coeff","reward_decay",
                 "qed_coeff","plogp_coeff","exp_temperature","exp_bias","property"]:
        if not hasattr(args, attr):
            setattr(args, attr, 0.0 if attr not in ("reward_type","property") else "linear")
    args.cuda = not no_cuda and __import__("torch").cuda.is_available()

    model = GraphFlowModel(max_size, node_dim, bond_dim, edge_unroll, args)
    ckpt  = __import__("torch").load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if args.cuda:
        model = model.cuda()
    model.eval()

    meta = {
        "cur_epoch" : ckpt.get("cur_epoch", "?"),
        "best_loss" : ckpt.get("best_loss", "?"),
        "max_size"  : max_size,
        "temperature": cfg.get("temperature", 0.75),
    }
    return model, meta


def evaluate(model, meta, n_samples=50, verbose=True):
    """Sinh n_samples graphs va tinh cac metrics."""
    uk_total = 0
    edge_total = 0
    so_count = 0
    fr_max_degrees = []
    node_counts = []
    all_graph_strs = []

    max_atoms = meta["max_size"]
    temperature = meta["temperature"]

    for i in range(n_samples):
        graph_str, pure_valid, node_cnt = model.generate(
            temperature=temperature, mute=True, max_atoms=max_atoms
        )
        parts = graph_str.strip().split()
        nodes = [p.lstrip("*") for p in parts if p.startswith("*")]
        verbs = [p for p in parts if not p.startswith("*") and re.match(r"^[A-Z]{2}$", p)]

        uk_total   += verbs.count("UK")
        edge_total += len(verbs)
        so_count   += int("SO" in nodes)
        node_counts.append(len(nodes))
        all_graph_strs.append(graph_str)

        # FR hub: max out-degree per node
        fr_out = defaultdict(int)
        tokens = graph_str.split()
        for j, t in enumerate(tokens):
            if t == "FR" and j >= 2:
                try:
                    fr_out[tokens[j-2]] += 1
                except:
                    pass
        fr_max_degrees.append(max(fr_out.values()) if fr_out else 0)

        if verbose and (i + 1) % 10 == 0:
            print("  Generated " + str(i+1) + "/" + str(n_samples) + " samples...")

    # Compute metrics
    uk_rate  = uk_total  / edge_total if edge_total > 0 else 0.0
    so_rate  = so_count  / n_samples
    avg_fr   = sum(fr_max_degrees) / n_samples
    max_fr   = max(fr_max_degrees)
    avg_nodes = sum(node_counts) / n_samples
    unique_rate = len(set(all_graph_strs)) / n_samples

    metrics = {
        "uk_rate"    : uk_rate,
        "so_rate"    : so_rate,
        "avg_fr_hub" : avg_fr,
        "max_fr_hub" : max_fr,
        "avg_nodes"  : avg_nodes,
        "unique_rate": unique_rate,
        "n_samples"  : n_samples,
    }
    return metrics


def print_report(checkpoint_path, meta, metrics):
    print()
    print("=" * 60)
    print("CHECKPOINT EVALUATION REPORT")
    print("=" * 60)
    print("  File      :", checkpoint_path)
    print("  Epoch     :", meta["cur_epoch"])
    print("  Best loss :", meta["best_loss"])
    print("  Samples   :", metrics["n_samples"])
    print()
    print("  Metric              Value    Rating")
    print("  " + "-" * 45)

    def rating(val, thresholds, labels, reverse=False):
        # thresholds: ascending. reverse=True means lower is better
        if reverse:
            if val <= thresholds[0]: return labels[0]
            if val <= thresholds[1]: return labels[1]
            return labels[2]
        else:
            if val >= thresholds[0]: return labels[0]
            if val >= thresholds[1]: return labels[1]
            return labels[2]

    uk_r  = rating(metrics["uk_rate"],  [0.05, 0.15], ["GOOD", "OK  ", "POOR"], reverse=True)
    so_r  = rating(metrics["so_rate"],  [0.75, 0.50], ["GOOD", "OK  ", "POOR"])
    fr_r  = rating(metrics["max_fr_hub"],[4,   6   ], ["GOOD", "OK  ", "POOR"], reverse=True)
    uniq_r= rating(metrics["unique_rate"],[0.8, 0.5 ], ["GOOD", "OK  ", "POOR"])

    print("  UK rate (lower=better): {:.1%}   [{}]".format(metrics["uk_rate"], uk_r))
    print("  SO rate (higher=better): {:.1%}  [{}]".format(metrics["so_rate"], so_r))
    print("  Avg FR hub degree:       {:.2f}  [{}]".format(metrics["avg_fr_hub"], fr_r))
    print("  Max FR hub degree:       {}".format(metrics["max_fr_hub"]))
    print("  Avg nodes/graph:         {:.1f}".format(metrics["avg_nodes"]))
    print("  Unique graph rate:       {:.1%}  [{}]".format(metrics["unique_rate"], uniq_r))
    print()

    # Overall verdict
    good_count = [uk_r, so_r, fr_r, uniq_r].count("GOOD")
    ok_count   = [uk_r, so_r, fr_r, uniq_r].count("OK  ")
    if good_count >= 3:
        verdict = "GOOD - Checkpoint chat luong cao, san sang dung"
    elif good_count + ok_count >= 3:
        verdict = "OK   - Checkpoint chap nhan duoc, co the dung"
    else:
        verdict = "POOR - Can train them hoac dieu chinh temperature"

    print("  Overall: [" + verdict + "]")
    print("=" * 60)

    return metrics


def print_usage_instructions(checkpoint_path):
    print()
    print("=" * 60)
    print("HUONG DAN SU DUNG CHECKPOINT NAY")
    print("=" * 60)

    ckpt_dir = os.path.dirname(checkpoint_path)
    print("""
Buoc 1: Download checkpoint tu Kaggle Output
    - Vao tab Output cua Kaggle notebook
    - Download toan bo thu muc: """ + ckpt_dir + """
    - Giai nen vao: TAGAPT_generated_samples/save_pretrain/exp_ASG_CTI_retrain/

Buoc 2: Cap nhat duong dan trong kaggle_pipeline.py hoac graph_instance.py
    Chinh sua bien CHECKPOINT_DIR hoac tuong duong:
        CHECKPOINT_DIR = "./save_pretrain/exp_ASG_CTI_retrain"

Buoc 3: Cap nhat duong dan trong MaskGAF.py (neu can sinh graph moi)
    Tim dong co 'init_checkpoint' va doi thanh:
        --init_checkpoint ./save_pretrain/exp_ASG_CTI_retrain/checkpoint

    Hoac them vao ham generate():
        restore_path = "./save_pretrain/exp_ASG_CTI_retrain/checkpoint"

Buoc 4: Chay lai pipeline sinh mau APT
    python graph_instance.py
    (hoac chay kaggle_pipeline.py tren Kaggle voi checkpoint moi)

Buoc 5: So sanh DOT files moi voi DOT files cu
    - Kiem tra UK edges: khong con sau khi resolve_uk_edges() chay
    - Kiem tra SO nodes: da co sau khi inject_network_node_if_missing()
    - Visual comparison: mo .dot files trong result-visualization/
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TAGAPT checkpoint quality")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(BASE, "save_pretrain",
                                             "exp_ASG_CTI_epoch3_1gpu", "checkpoint"),
                        help="Path to checkpoint file (no extension)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of graphs to generate for evaluation (default 50)")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--compare", type=str, default=None,
                        help="Optional second checkpoint to compare against")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading checkpoint: " + args.checkpoint)
    print("=" * 60)

    model, meta = load_checkpoint_and_model(args.checkpoint, no_cuda=args.no_cuda)
    print("  [OK] Checkpoint loaded (epoch=" + str(meta["cur_epoch"]) +
          ", best_loss=" + str(round(meta["best_loss"], 4) if isinstance(meta["best_loss"], float) else meta["best_loss"]) + ")")

    print()
    print("Generating " + str(args.n_samples) + " samples for evaluation...")
    metrics = evaluate(model, meta, n_samples=args.n_samples)
    print_report(args.checkpoint, meta, metrics)

    # Compare with second checkpoint if provided
    if args.compare:
        print()
        print("=" * 60)
        print("COMPARING WITH: " + args.compare)
        print("=" * 60)
        model2, meta2 = load_checkpoint_and_model(args.compare, no_cuda=args.no_cuda)
        metrics2 = evaluate(model2, meta2, n_samples=args.n_samples)
        print_report(args.compare, meta2, metrics2)

        print()
        print("=== COMPARISON SUMMARY ===")
        print("Metric              Checkpoint1         Checkpoint2         Winner")
        print("-" * 75)
        pairs = [
            ("UK rate",    metrics["uk_rate"],     metrics2["uk_rate"],     "lower",  lambda x: "{:.1%}".format(x)),
            ("SO rate",    metrics["so_rate"],     metrics2["so_rate"],     "higher", lambda x: "{:.1%}".format(x)),
            ("Avg FR hub", metrics["avg_fr_hub"],  metrics2["avg_fr_hub"],  "lower",  lambda x: "{:.2f}".format(x)),
            ("Unique%",    metrics["unique_rate"], metrics2["unique_rate"], "higher", lambda x: "{:.1%}".format(x)),
        ]
        for name, v1, v2, better, fmt in pairs:
            if better == "lower":
                winner = "checkpoint1" if v1 < v2 else "checkpoint2"
            else:
                winner = "checkpoint1" if v1 > v2 else "checkpoint2"
            print("{:<20} {:<20} {:<20} {}".format(
                name, fmt(v1), fmt(v2), winner))

    print_usage_instructions(args.checkpoint)
