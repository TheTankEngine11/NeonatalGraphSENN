#!/usr/bin/env python3
"""
RunValidationSweep.py

Small Python wrapper that runs Validation.py once for every Saved_models_* folder
inside a model sweep directory.

Expected sweep layout:
    ModelSweeps/
        Saved_models_468340_MTbase_LR0.002_WD0.001_robloss0.0/
            GAT_CV_10_0/best_auprc.pt
            ...
        Saved_models_480652_MTSENNrawx_LR0.002_WD0.001_robloss3e-5/
            GAT_CV_10_0/best_auprc.pt
            ...

For every Saved_models_* folder it creates a matching Results_* folder.
"""

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys


# ============================================================
# USER SETTINGS
# ============================================================
MODEL_PATTERN = "Saved_models_*"
HISTORY_PREFIX = "History_"
RESULTS_PREFIX = "Results_"

N_FOLDS = 10
CHECKPOINT_NAME = "best_auprc.pt"
MODE = "recompute"        # "recompute" or "checkpoint"

# False: stop when a validation run fails.
# True: continue with the other models.
KEEP_GOING = False
# ============================================================


def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def suffix_from_saved_models_name(model_dir):
    name = os.path.basename(os.path.normpath(model_dir))
    if name.startswith("Saved_models_"):
        return name.replace("Saved_models_", "", 1)
    return name


def infer_history_dir(model_dir, history_root):
    suffix = suffix_from_saved_models_name(model_dir)
    return os.path.join(history_root, HISTORY_PREFIX + suffix)


def infer_results_dir(model_dir, results_root):
    suffix = suffix_from_saved_models_name(model_dir)
    return os.path.join(results_root, RESULTS_PREFIX + suffix)


def find_folds(model_dir):
    folds = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(model_dir, f"GAT_CV_10_{fold}", CHECKPOINT_NAME)
        if os.path.isfile(ckpt_path):
            folds.append(fold)
    return folds


def find_model_dirs(models_root):
    pattern = os.path.join(models_root, MODEL_PATTERN)
    model_dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    model_dirs = sorted(model_dirs, key=lambda p: natural_key(os.path.basename(os.path.normpath(p))))

    valid_model_dirs = []
    for model_dir in model_dirs:
        folds = find_folds(model_dir)
        if len(folds) == 0:
            print(f"[SKIP] {model_dir}: no {CHECKPOINT_NAME} files found")
            continue
        valid_model_dirs.append((model_dir, folds))

    return valid_model_dirs


def build_command(python_cmd, validation_script, data_folder, model_dir, history_dir, results_dir):
    cmd = [
        python_cmd,
        validation_script,
        "--mode", MODE,
        "--data_folder", data_folder,
        "--model_dir", model_dir,
        "--history_dir", history_dir,
        "--results_dir", results_dir,
    ]
    return cmd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", required=True, help="Folder containing Saved_models_* directories")
    parser.add_argument("--data_folder", required=True, help="Path to Datasets/CV_Folds")
    parser.add_argument("--validation_script", default="./Validation.py")
    parser.add_argument("--history_root", default="", help="Folder containing History_* directories. Default: models_root")
    parser.add_argument("--results_root", default="", help="Folder where Results_* directories are written. Default: models_root")
    parser.add_argument("--python_cmd", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    models_root = os.path.abspath(os.path.expanduser(args.models_root))
    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    validation_script = os.path.abspath(os.path.expanduser(args.validation_script))
    history_root = os.path.abspath(os.path.expanduser(args.history_root)) if args.history_root else models_root
    results_root = os.path.abspath(os.path.expanduser(args.results_root)) if args.results_root else models_root

    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"models_root does not exist: {models_root}")
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"data_folder does not exist: {data_folder}")
    if not os.path.isfile(validation_script):
        raise FileNotFoundError(f"validation_script does not exist: {validation_script}")

    model_dirs = find_model_dirs(models_root)
    if len(model_dirs) == 0:
        raise FileNotFoundError(f"No valid {MODEL_PATTERN} folders found in {models_root}")

    print(f"Found {len(model_dirs)} model run(s).")
    print(f"Models root      : {models_root}")
    print(f"Data folder      : {data_folder}")
    print(f"Validation script: {validation_script}")

    failures = []

    for idx, (model_dir, folds) in enumerate(model_dirs, start=1):
        history_dir = infer_history_dir(model_dir, history_root)
        results_dir = infer_results_dir(model_dir, results_root)
        os.makedirs(results_dir, exist_ok=True)

        cmd = build_command(
            python_cmd=args.python_cmd,
            validation_script=validation_script,
            data_folder=data_folder,
            model_dir=model_dir,
            history_dir=history_dir,
            results_dir=results_dir,
        )

        print("=" * 90)
        print(f"[{idx}/{len(model_dirs)}] {os.path.basename(os.path.normpath(model_dir))}")
        print(f"Folds found: {folds}")
        print(f"Results dir: {results_dir}")
        print("Command:")
        print(" ".join(shlex.quote(x) for x in cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            failures.append((model_dir, completed.returncode))
            print(f"[ERROR] {model_dir} failed with return code {completed.returncode}")
            if not KEEP_GOING:
                raise SystemExit(completed.returncode)
        else:
            print(f"[OK] {model_dir}")

    if len(failures) > 0:
        print("\nSome runs failed:")
        for model_dir, code in failures:
            print(f"  {model_dir}: return code {code}")
        raise SystemExit(1)

    print("\nAll validation runs completed successfully.")


if __name__ == "__main__":
    main()
