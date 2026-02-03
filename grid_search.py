import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple

import torch

# Workspace-local training script
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")


def parse_list(arg: str, cast_type=float):
    items = [s.strip() for s in arg.split(',') if s.strip()]
    return [cast_type(x) for x in items]


def run_training(dataset: str, model: str, batch_size: int, epochs: int,
                fid_epoch_calc: int, lr: float, beta_schedule: str, 
                run_name: str,
                models_root: Optional[str] = None,
                extra_args: Optional[List[str]] = None,
                python_exe: Optional[str] = None) -> int:
    cmd = [python_exe or sys.executable, TRAIN_SCRIPT,
        "--dataset", dataset,
        "--model", model,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--fid_epoch_calc", str(fid_epoch_calc),
        "--beta_schedule", beta_schedule,
        "--run_name", run_name,
           *( ["--models_root", models_root] if models_root else [] ),
        "--disable_mlflow"
        ]
    if extra_args:
        cmd.extend(extra_args)
    print("\n>>> Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=os.path.dirname(TRAIN_SCRIPT))
    return proc.returncode


def find_best_fid(models_dir: str) -> Tuple[Optional[float], Optional[str]]:
    if not os.path.isdir(models_dir):
        return None, None
    best_val = None
    best_path = None
    for fname in os.listdir(models_dir):
        if not fname.endswith('.pth'):
            continue
        fpath = os.path.join(models_dir, fname)
        try:
            ckpt = torch.load(fpath, map_location='cpu', weights_only=False)
        except Exception:
            continue
        fid = ckpt.get('best_fid')
        if isinstance(fid, (int, float)):
            if best_val is None or fid < best_val:
                best_val, best_path = float(fid), fpath
    return best_val, best_path


def discover_run_folders(models_root: str, dataset: str) -> List[str]:
    base = os.path.join(models_root, dataset)
    if not os.path.isdir(base):
        return []
    return [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]


def parse_run_metadata(run_folder: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    # Expected format: "<bs>_<schedule>_lr<lr>_bs<bs>"
    m = re.match(r"^(?P<bs>\d+)_?(?P<sched>[^_]+)?_lr(?P<lr>[^_]+)_bs(?P<bs2>\d+)$", run_folder)
    if not m:
        # Try simpler pattern: "<bs>_<rest>" then parse rest
        m2 = re.match(r"^(?P<bs>\d+)_(?P<rest>.+)$", run_folder)
        if m2:
            bs = int(m2.group("bs"))
            rest = m2.group("rest")
            m3 = re.match(r"^(?P<sched>[^_]+)_lr(?P<lr>[^_]+)_bs(?P<bs2>\d+)$", rest)
            if m3:
                bs2 = int(m3.group("bs2"))
                sched = m3.group("sched")
                try:
                    lr = float(m3.group("lr"))
                except ValueError:
                    lr = None
                return (bs if bs == bs2 else bs2, lr, sched)
        return None, None, None
    try:
        bs = int(m.group("bs"))
    except Exception:
        bs = None
    try:
        lr = float(m.group("lr"))
    except Exception:
        lr = None
    sched = m.group("sched")
    bs2 = m.group("bs2")
    try:
        bs2 = int(bs2)
    except Exception:
        bs2 = None
    if bs is not None and bs2 is not None and bs != bs2:
        bs = bs2
    return bs, lr, sched


def main():
    parser = argparse.ArgumentParser(description="Grid search for diffusion hyperparams")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "custom"])
    parser.add_argument("--model", default="UNET", choices=["UNET", "Basic"])
    parser.add_argument("--epochs", type=int, default=6, help="epochs per run")
    parser.add_argument("--batch_sizes", type=str, default="8,16")
    parser.add_argument("--lrs", type=str, default="1e-4,5e-5")
    parser.add_argument("--schedules", type=str, default="squaredcos_cap_v2,scaled_linear")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--fid_epoch_calc", type=int, default=3, help="epoch to calculate FID at")
    parser.add_argument("--extra_args", type=str, default="", help="extra args to pass to train.py")
    parser.add_argument("--python", type=str, default=None, help="path to python interpreter")
    parser.add_argument("--models_root", type=str, default="grid_search_results/models", help="Root directory to store and scan models")
    parser.add_argument("--scan_only", action="store_true", help="Only scan existing runs; do not train")
    parser.add_argument("--auto_discover", action="store_true", help="Scan all run folders under models_root/dataset")
    args = parser.parse_args()

    batch_sizes = parse_list(args.batch_sizes, int)
    lrs = parse_list(args.lrs, float)
    schedules = [s.strip() for s in args.schedules.split(',') if s.strip()]

    results = []
    
    # In auto-discover mode, ignore provided grids and scan all run folders
    discovered = []
    if args.scan_only and args.auto_discover:
        discovered = discover_run_folders(args.models_root, args.dataset)

    combos = []
    if discovered:
        for run_folder in discovered:
            bs, lr, sched = parse_run_metadata(run_folder)
            combos.append((bs, lr, sched, run_folder))
    else:
        for bs, lr, sched in itertools.product(batch_sizes, lrs, schedules):
            run_name = f"{sched}_lr{lr}_bs{bs}"
            run_folder = f"{bs}_{run_name}"
            combos.append((bs, lr, sched, run_folder))

    for bs, lr, sched, run_folder in combos:
        run_name = f"{sched}_lr{lr}_bs{bs}" if (sched is not None and lr is not None and bs is not None) else run_folder
        # pass optional extra args
        extra = []
        if args.augment:
            extra.append("--augment")
        if args.extra_args:
            extra.extend(args.extra_args.split())
        # Use a single models_root for both training and scanning
        models_root = args.models_root
        os.makedirs(models_root, exist_ok=True)
        if not args.scan_only:
            rc = run_training(
                args.dataset, args.model, bs, args.epochs, args.fid_epoch_calc, lr, sched, run_name,
                models_root=models_root, extra_args=extra, python_exe=args.python
            )
            if rc != 0:
                print(f"Run failed (exit {rc}) for {run_name}")
        # After run (or in scan-only), inspect model directory for best FID
        models_dir = os.path.join(models_root, args.dataset, run_folder)
        best_fid, best_path = find_best_fid(models_dir)
        results.append({
            "batch_size": bs,
            "lr": lr,
            "schedule": sched,
            "run_name": run_name,
            "models_dir": models_dir,
            "best_fid": best_fid,
            "best_fid_path": best_path,
        })
        print(f"Result: schedule={sched}, lr={lr}, bs={bs} -> best_fid={best_fid} ({best_path})")

    # Sort and persist
    results.sort(key=lambda r: (float('inf') if r['best_fid'] is None else r['best_fid']))
    os.makedirs("grid_search_results", exist_ok=True)
    out_json = os.path.join("grid_search_results", "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")
    if results and results[0]['best_fid'] is not None:
        best = results[0]
        print("\nBest configuration:")
        print(f" schedule={best['schedule']} lr={best['lr']} bs={best['batch_size']}")
        print(f" best_fid={best['best_fid']} at {best['best_fid_path']}")


if __name__ == "__main__":
    main()
