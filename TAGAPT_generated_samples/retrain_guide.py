#!/usr/bin/env python
"""
retrain_guide.py
================
Script retrain GraphFlowModel them epochs tren Kaggle.

Cach dung:
    1. Upload file nay va toan bo TAGAPT_generated_samples/ len Kaggle
    2. Dat accelerator = GPU T4 x2 hoac P100
    3. Chay: python retrain_guide.py --additional_epochs 7 --save

Muc tieu: tu checkpoint 3 epochs -> train them 7 epochs = tong 10 epochs
Du kien: UK rate giam tu 14.1% xuong <5%, graph quality tang ro.
"""

import os, sys, json, argparse
import warnings
warnings.filterwarnings("ignore")

# ── Hyperparameters ────────────────────────────────────────────────────────

# BASE = directory where this script lives (may be read-only on Kaggle input)
BASE = os.path.dirname(os.path.abspath(__file__))

# Ensure local project modules (train_CTI, dataloader, utils, logmaking, ...) are importable
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# On Kaggle, /kaggle/input/ is READ-ONLY.
# All write-able output must go to /kaggle/working/.
if os.path.exists("/kaggle"):
    WORKING_DIR  = "/kaggle/working/TAGAPT_retrain"
else:
    WORKING_DIR  = os.path.join(BASE, "retrain_output")

CHECKPOINT_DIR  = os.path.join(BASE, "save_pretrain", "exp_ASG_CTI_epoch3_1gpu")
DATA_PREFIX     = os.path.join(BASE, "data_preprocessed", "CTI")
NEW_SAVE_DIR    = os.path.join(WORKING_DIR, "exp_ASG_CTI_retrain")

# These match the original training config exactly
ORIGINAL_CONFIG = {
    "num_flow_layer": 12,
    "gcn_layer"     : 3,
    "nhid"          : 128,
    "nout"          : 128,
    "st_type"       : "exp",
    "edge_unroll"   : 10,
    "deq_type"      : "random",
    "deq_coeff"     : 0.9,
    "is_bn"         : True,
    "is_bn_before"  : False,
    "scale_weight_norm": False,
    "divide_loss"   : True,
    "learn_prior"   : False,
    "sigmoid_shift" : 2.0,
    "dropout"       : 0.0,
}


def build_args(additional_epochs=7, batch_size=4, lr=0.0005, temperature=0.75,
               save=True, use_cuda=True):
    """Build argparse.Namespace equivalent to what train_CTI.py expects."""
    import argparse
    args = argparse.Namespace(
        dataset         = "ASG",
        path            = DATA_PREFIX,
        batch_size      = batch_size,
        edge_unroll     = ORIGINAL_CONFIG["edge_unroll"],
        shuffle         = True,
        num_workers     = 2,          # lower on Kaggle to avoid memory issues
        name            = "retrain",
        deq_type        = ORIGINAL_CONFIG["deq_type"],
        deq_coeff       = ORIGINAL_CONFIG["deq_coeff"],
        num_flow_layer  = ORIGINAL_CONFIG["num_flow_layer"],
        gcn_layer       = ORIGINAL_CONFIG["gcn_layer"],
        nhid            = ORIGINAL_CONFIG["nhid"],
        nout            = ORIGINAL_CONFIG["nout"],
        st_type         = ORIGINAL_CONFIG["st_type"],
        sigmoid_shift   = ORIGINAL_CONFIG["sigmoid_shift"],
        all_save_prefix = WORKING_DIR + os.sep,
        train           = True,
        save            = save,
        no_cuda         = not use_cuda,
        learn_prior     = ORIGINAL_CONFIG["learn_prior"],
        seed            = 2019,
        epochs          = additional_epochs,
        lr              = lr,          # lower LR for fine-tuning
        weight_decay    = 0.0,
        dropout         = ORIGINAL_CONFIG["dropout"],
        is_bn           = ORIGINAL_CONFIG["is_bn"],
        is_bn_before    = ORIGINAL_CONFIG["is_bn_before"],
        scale_weight_norm = ORIGINAL_CONFIG["scale_weight_norm"],
        divide_loss     = ORIGINAL_CONFIG["divide_loss"],
        init_checkpoint = os.path.join(CHECKPOINT_DIR, "checkpoint"),
        show_loss_step  = 50,
        temperature     = temperature,
        min_atoms       = 5,
        # IMPORTANT: max_atoms MUST be >= model.max_size (from data_config).
        # model.generate() allocates cur_adj_features with shape [1,bond_dim,max_atoms,max_atoms]
        # and rule_check() loops for i1 in range(self.max_size) -> IndexError if max_atoms < max_size.
        # Original training config used max_atoms=100; keep it consistent.
        max_atoms       = 100,
        gen_num         = 10,
        gen             = False,
        gen_out_path    = None,
        # Fields needed by GraphFlowModel but not in argparse
        penalty         = False,
        reward_type     = "linear",
        moving_coeff    = 0.0,
        reward_decay    = 0.9,
        qed_coeff       = 1.0,
        plogp_coeff     = 1.0,
        exp_temperature = 1.0,
        exp_bias        = 0.0,
        property        = "qed",
    )
    import torch
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save_path = NEW_SAVE_DIR
    # Ensure working dir exists before anything tries to write
    os.makedirs(WORKING_DIR, exist_ok=True)
    return args


def read_molecules(path_prefix):
    import numpy as np
    node_features = np.load(path_prefix + "_np_node.npy")
    adj_features  = np.load(path_prefix + "_np_adj.npy")
    mol_sizes     = np.load(path_prefix + "_np_mol.npy")
    data_config   = eval(open(path_prefix + "_config.txt").read())
    all_smiles    = [l.strip() for l in open(path_prefix + "_graph.txt")]
    return node_features, adj_features, mol_sizes, data_config, all_smiles


def measure_uk_rate(model, data_config, args, n_samples=50):
    """Generate n_samples graphs and compute UK edge rate."""
    import re
    model.eval()
    uk_total, edge_total = 0, 0
    so_count = 0
    # max_atoms MUST equal model.max_size so rule_check() tensor access stays in bounds.
    # model.max_size is set from data_config['max_size'] during __init__.
    safe_max_atoms = getattr(model, 'max_size', data_config.get('max_size', 100))
    for _ in range(n_samples):
        graph_str, _, _ = model.generate(
            temperature=args.temperature, mute=True, max_atoms=safe_max_atoms
        )
        parts = graph_str.strip().split()
        verbs = [p for p in parts if re.match(r'^[A-Z]{2}$', p) and not p.startswith('*')]
        nodes = [p.lstrip('*') for p in parts if p.startswith('*')]
        uk_total   += verbs.count('UK')
        edge_total += len(verbs)
        so_count   += int('SO' in nodes)
    uk_rate = uk_total / edge_total if edge_total else 0.0
    so_rate = so_count / n_samples
    return uk_rate, so_rate


def run_retrain(additional_epochs=7, batch_size=4, lr=0.0005, save=True):
    import torch
    from torch.utils.data import DataLoader
    from train_CTI import Trainer
    from dataloader import PretrainZinkDataset
    from utils import set_seed

    args = build_args(additional_epochs=additional_epochs,
                      batch_size=batch_size, lr=lr, save=save)

    print("=" * 60)
    print("TAGAPT GraphFlowModel -- Retrain from checkpoint")
    print("=" * 60)
    print("Device:", "GPU" if args.cuda else "CPU (slow!)")
    print("Resuming from:", args.init_checkpoint)
    print("Save dir:", args.save_path)
    print("Additional epochs:", additional_epochs)
    print("LR:", lr, " (lower than original 0.001 -- fine-tune mode)")
    print()

    if not os.path.exists(args.init_checkpoint):
        print("ERROR: checkpoint not found at", args.init_checkpoint)
        sys.exit(1)

    if save and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_seed(args.seed, args.cuda)

    # Load data
    print("Loading training data from", DATA_PREFIX + "_np_node.npy ...")
    node_features, adj_features, mol_sizes, data_config, all_smiles = read_molecules(DATA_PREFIX)

    train_loader = DataLoader(
        PretrainZinkDataset(node_features, adj_features, mol_sizes),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    print("Training graphs:", len(mol_sizes))

    # Build trainer (will load checkpoint in initialize_from_checkpoint)
    trainer = Trainer(train_loader, data_config, args, all_train_smiles=all_smiles)
    trainer.initialize_from_checkpoint(gen=False)

    # Measure baseline before retrain
    print()
    print("--- Baseline (epoch 3 checkpoint) ---")
    base_uk, base_so = measure_uk_rate(trainer._model, data_config, args, n_samples=30)
    print(f"  UK rate: {base_uk:.1%}  SO rate: {base_so:.1%}")

    # Train
    print()
    print("--- Starting retrain ---")
    mol_out_dir = os.path.join(args.save_path, "asg") if save else None
    if mol_out_dir and not os.path.exists(mol_out_dir):
        os.makedirs(mol_out_dir)

    trainer.fit(mol_out_dir=mol_out_dir)

    # Measure after retrain
    print()
    print("--- After retrain ---")
    after_uk, after_so = measure_uk_rate(trainer._model, data_config, args, n_samples=30)
    print(f"  UK rate: {after_uk:.1%}  SO rate: {after_so:.1%}")
    improvement = (base_uk - after_uk) / base_uk * 100 if base_uk > 0 else 0
    print(f"  UK rate improvement: {improvement:.1f}%")

    print()
    print("Retrain complete. New checkpoint saved to:", args.save_path)
    print()
    print("Next steps:")
    print("  1. Copy new checkpoint to TAGAPT_generated_samples/save_pretrain/exp_ASG_CTI_retrain/")
    print("  2. Update graph_instance.py to point to new checkpoint if needed")
    print("  3. Re-run graph generation pipeline to compare graphs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAGAPT fine-tune from checkpoint")
    parser.add_argument("--additional_epochs", type=int, default=7,
                        help="Extra epochs to train (default 7, total = 3+7 = 10)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Fine-tune LR (lower than original 0.001)")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no_save", dest="save", action="store_false")
    cli = parser.parse_args()

    run_retrain(
        additional_epochs=cli.additional_epochs,
        batch_size=cli.batch_size,
        lr=cli.lr,
        save=cli.save,
    )
