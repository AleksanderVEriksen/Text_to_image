import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple

import torch
import shutil
import time

# Workspace-local training script
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")

# Run folder naming used by this grid search:
#   <bs>_<schedule>_lr<lr>_bs<bs>
# Note: <schedule> may contain underscores, so we match it non-greedily up to "_lr".
RUN_FOLDER_RE = re.compile(r"^(?P<bs>\d+)_(?P<sched>.+?)_lr(?P<lr>[^_]+)_bs(?P<bs2>\d+)$")


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


def find_best_fid(models_dir: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    if not os.path.isdir(models_dir):
        return None, None, None
    best_val = None
    best_path = None
    best_runtime = None
    for fname in os.listdir(models_dir):
        if not fname.endswith('.pth'):
            continue
        fpath = os.path.join(models_dir, fname)
        try:
            ckpt = torch.load(fpath, map_location='cpu', weights_only=False)
        except Exception:
            continue
        fid = ckpt.get('best_fid')
        run_time = ckpt.get('run_time')
        if isinstance(fid, (int, float)):
            if best_val is None or fid < best_val:
                best_val, best_path = float(fid), fpath
                best_runtime = float(run_time) if isinstance(run_time, (int, float)) else None
    return best_val, best_path, best_runtime


def discover_run_folders(models_root: str, dataset: str) -> List[str]:
    base = os.path.join(models_root, dataset)
    if not os.path.isdir(base):
        return []
    run_folders: List[str] = []
    for name in os.listdir(base):
        if name == "checkpoints":
            continue
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        if RUN_FOLDER_RE.match(name):
            run_folders.append(name)
    return run_folders


def parse_run_metadata(run_folder: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    m = RUN_FOLDER_RE.match(run_folder)
    if not m:
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


def _clean_dataset_folder(models_root: str, dataset: str) -> None:
    target_dir = os.path.join(models_root, dataset)
    if not os.path.isdir(target_dir):
        return
    print(f"Cleaning grid search dataset folder: {target_dir}")
    for name in os.listdir(target_dir):
        path = os.path.join(target_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")


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
    parser.add_argument("--clean", action="store_true", help="Delete contents of models_root/<dataset> before running")
    parser.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        help="Skip training if an existing run folder already has a checkpoint with best_fid (default)",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing runs; always (re)train",
    )
    parser.add_argument(
        "--efficiency_metric",
        type=str,
        default="fid_per_sec",
        choices=["fid_per_sec", "weighted"],
        help="Efficiency metric: fid_per_sec score (higher is better, in (0,1]) or weighted normalized score (higher is better, in [0,1])",
    )
    parser.add_argument("--efficiency_weight", type=float, default=0.6, help="Weight for FID in weighted efficiency (0-1); time weight is 1-w")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()


    batch_sizes = parse_list(args.batch_sizes, int)
    lrs = parse_list(args.lrs, float)
    schedules = [s.strip() for s in args.schedules.split(',') if s.strip()]

    results = []

    # Optional cleanup of dataset folder under models_root
    if args.clean:
        os.makedirs(args.models_root, exist_ok=True)
        _clean_dataset_folder(args.models_root, args.dataset)
    
    # In auto-discover mode, ignore provided grids and scan all run folders
    discovered = []
    if args.scan_only and args.auto_discover:
        discovered = discover_run_folders(args.models_root, args.dataset)

    combos = []
    if discovered:
        for run_folder in discovered:
            bs, lr, sched = parse_run_metadata(run_folder)
            if bs is None or lr is None or sched is None:
                # Ignore folders that don't match the expected grid-search naming
                continue
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
        # When models_root ends with 'models', route figures to its parent (e.g., 'grid_search_results')
        base_root = os.path.normpath(models_root)
        if os.path.basename(base_root) == "models":
            figure_root = os.path.dirname(base_root)
            if figure_root:
                extra.extend(["--figure_root", figure_root])

        # Resolve model folder for this run and optionally skip if it already has results.
        models_dir = os.path.join(models_root, args.dataset, run_folder)
        if (not args.scan_only) and args.skip_existing and os.path.isdir(models_dir):
            existing_best_fid, existing_best_path, existing_runtime_ckpt = find_best_fid(models_dir)
            if existing_best_fid is not None:
                print(
                    f"Skipping existing run (already has best_fid): {run_folder} -> best_fid={existing_best_fid} ({existing_best_path})"
                )
                runtime_val = existing_runtime_ckpt
                fid_per_sec_raw = (
                    (existing_best_fid / max(runtime_val, 1e-6))
                    if (existing_best_fid is not None and runtime_val is not None)
                    else None
                )
                # Convert raw ratio to a bounded score where closer to 1 is better
                eff_fid_per_sec = (1.0 / (1.0 + fid_per_sec_raw)) if fid_per_sec_raw is not None else None
                results.append({
                    "batch_size": bs,
                    "lr": lr,
                    "schedule": sched,
                    "run_name": run_name,
                    "models_dir": models_dir,
                    "best_fid": existing_best_fid,
                    "best_fid_path": existing_best_path,
                    "runtime_seconds": runtime_val,
                    "fid_per_sec_raw": fid_per_sec_raw,
                    "efficiency_fid_per_sec": eff_fid_per_sec,
                })
                continue
        if not args.scan_only:
            rc = run_training(
                args.dataset, args.model, bs, args.epochs, args.fid_epoch_calc, lr, sched, run_name,
                models_root=models_root, extra_args=extra, python_exe=args.python
            )
            if rc != 0:
                print(f"Run failed (exit {rc}) for {run_name}")
        # After run (or in scan-only), inspect model directory for best FID
        best_fid, best_path, runtime_ckpt = find_best_fid(models_dir)
        # Efficiency based on checkpoint runtime (preferred). If missing, remains None in scan-only mode.
        runtime_val = runtime_ckpt
        fid_per_sec_raw = (best_fid / max(runtime_val, 1e-6)) if (best_fid is not None and runtime_val is not None) else None
        # Convert raw ratio to a bounded score where closer to 1 is better
        eff_fid_per_sec = (1.0 / (1.0 + fid_per_sec_raw)) if fid_per_sec_raw is not None else None
        results.append({
            "batch_size": bs,
            "lr": lr,
            "schedule": sched,
            "run_name": run_name,
            "models_dir": models_dir,
            "best_fid": best_fid,
            "best_fid_path": best_path,
            "runtime_seconds": runtime_val,
            "fid_per_sec_raw": fid_per_sec_raw,
            "efficiency_fid_per_sec": eff_fid_per_sec,
        })
        time_str = f"{runtime_val:.2f}s" if isinstance(runtime_val, (int, float)) else "n/a"
        print(f"Result: schedule={sched}, lr={lr}, bs={bs}, time={time_str} -> best_fid={best_fid} ({best_path})")

    # Sort and persist
    results.sort(key=lambda r: (float('inf') if r['best_fid'] is None else r['best_fid']))
    os.makedirs("grid_search_results", exist_ok=True)
    out_json = os.path.join("grid_search_results", "results.json")
    # Compute weighted normalized efficiency if requested
    if args.efficiency_metric == "weighted":
        # Collect ranges
        fid_vals = [r["best_fid"] for r in results if r["best_fid"] is not None]
        time_vals = [r["runtime_seconds"] for r in results if r["runtime_seconds"] is not None]
        fid_min, fid_max = (min(fid_vals), max(fid_vals)) if fid_vals else (None, None)
        time_min, time_max = (min(time_vals), max(time_vals)) if time_vals else (None, None)
        def norm(val, vmin, vmax):
            if val is None or vmin is None or vmax is None or vmax == vmin:
                return None
            return (val - vmin) / (vmax - vmin)
        w = max(0.0, min(1.0, args.efficiency_weight))
        for r in results:
            fid_n = norm(r["best_fid"], fid_min, fid_max)
            time_n = norm(r["runtime_seconds"], time_min, time_max)
            if fid_n is not None and time_n is not None:
                # fid_n and time_n are in [0,1] where 0 is best. Convert to a score in [0,1] where 1 is best.
                weighted_loss = w * fid_n + (1.0 - w) * time_n
                r["efficiency_weighted"] = 1.0 - weighted_loss
            else:
                r["efficiency_weighted"] = None
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")
    if results and results[0]['best_fid'] is not None:
        best = results[0]
        print("\nBest configuration:")
        print(f" schedule={best['schedule']} lr={best['lr']} bs={best['batch_size']} time={best['runtime_seconds']:.2f}s, efficiency_fid_per_sec={best.get('efficiency_fid_per_sec', 'n/a'):.4f}")
        print(f" best_fid={best['best_fid']} at {best['best_fid_path']}")

    # Also print best by efficiency when available
    if any(r.get("efficiency_fid_per_sec") is not None for r in results):
        best_eff = sorted(
            [r for r in results if r.get("efficiency_fid_per_sec") is not None],
            key=lambda r: r["efficiency_fid_per_sec"],
            reverse=True,
        )[0]
        print("\nBest by efficiency (FID/sec score; higher is better):")
        print(f" schedule={best_eff['schedule']} lr={best_eff['lr']} bs={best_eff['batch_size']} time={best_eff['runtime_seconds']:.2f}s")
        raw = best_eff.get("fid_per_sec_raw")
        raw_str = f"{raw:.6f}" if isinstance(raw, (int, float)) else "n/a"
        print(f" fid={best_eff['best_fid']} fid_per_sec_raw={raw_str} efficiency={best_eff['efficiency_fid_per_sec']:.4f}")
    if any(r.get("efficiency_weighted") is not None for r in results):
        best_w = sorted(
            [r for r in results if r.get("efficiency_weighted") is not None],
            key=lambda r: r["efficiency_weighted"],
            reverse=True,
        )[0]
        print("\nBest by weighted efficiency (higher is better):")
        print(f" schedule={best_w['schedule']} lr={best_w['lr']} bs={best_w['batch_size']} time={best_w['runtime_seconds']:.2f}s")
        print(f" fid={best_w['best_fid']} efficiency={best_w['efficiency_weighted']:.4f} (w={args.efficiency_weight})")


if __name__ == "__main__":
    main()
