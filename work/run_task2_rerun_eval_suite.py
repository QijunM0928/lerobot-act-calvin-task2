#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full 2000-step Task2 eval suite.")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--offline-script", required=True)
    parser.add_argument("--visual-shift-script", required=True)
    parser.add_argument("--a-run-dir", required=True)
    parser.add_argument("--abc-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[500, 1000, 1500, 2000])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=6400)
    parser.add_argument("--samples-per-family", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splitA-root", required=True)
    parser.add_argument("--splitB-root", required=True)
    parser.add_argument("--splitC-root", required=True)
    parser.add_argument("--splitD-root", required=True)
    parser.add_argument("--splitA-repo", default="calvin_splitA")
    parser.add_argument("--splitB-repo", default="calvin_splitB")
    parser.add_argument("--splitC-repo", default="calvin_splitC")
    parser.add_argument("--splitD-repo", default="calvin_splitD")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_cmd(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ckpt_map(run_dir: Path, prefix: str, steps: list[int]) -> dict[str, str]:
    out = {}
    for step in steps:
        step_name = f"{step:06d}"
        ckpt_dir = run_dir / "checkpoints" / step_name / "pretrained_model"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir}")
        out[f"{prefix}{step:04d}"] = str(ckpt_dir)
    return out


def run_offline_eval(
    python_bin: str,
    script: str,
    dataset_root: str,
    repo_id: str,
    output_path: Path,
    checkpoints: dict[str, str],
    batch_size: int,
    max_samples: int,
    device: str,
    skip_existing: bool,
):
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return
    cmd = [
        python_bin,
        script,
        "--dataset-root",
        dataset_root,
        "--repo-id",
        repo_id,
        "--batch-size",
        str(batch_size),
        "--max-samples",
        str(max_samples),
        "--device",
        device,
        "--output",
        str(output_path),
        "--checkpoints",
    ]
    for name, path in checkpoints.items():
        cmd.append(f"{name}={path}")
    run_cmd(cmd)


def run_visual_shift_eval(
    python_bin: str,
    script: str,
    output_path: Path,
    checkpoints: dict[str, str],
    batch_size: int,
    samples_per_family: int,
    device: str,
    seed: int,
    splitA_root: str,
    splitD_root: str,
    splitA_repo: str,
    splitD_repo: str,
    skip_existing: bool,
):
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return
    cmd = [
        python_bin,
        script,
        "--split-root",
        f"splitA={splitA_root}",
        "--split-root",
        f"splitD={splitD_root}",
        "--repo-id",
        f"splitA={splitA_repo}",
        "--repo-id",
        f"splitD={splitD_repo}",
        "--checkpoints",
    ]
    for name, path in checkpoints.items():
        cmd.append(f"{name}={path}")
    cmd.extend(
        [
            "--samples-per-family",
            str(samples_per_family),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--seed",
            str(seed),
            "--output",
            str(output_path),
        ]
    )
    run_cmd(cmd)


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_run_dir = Path(args.a_run_dir)
    abc_run_dir = Path(args.abc_run_dir)

    a_ckpts = ckpt_map(a_run_dir, "A", args.steps)
    abc_ckpts = ckpt_map(abc_run_dir, "ABC", args.steps)

    paths = {
        "splitD_a": output_dir / "splitD_Aonly.json",
        "splitD_abc": output_dir / "splitD_ABCjoint.json",
        "visual_shift_a": output_dir / "visual_shift_Aonly_D_minus_A.json",
        "visual_shift_abc": output_dir / "visual_shift_ABCjoint_D_minus_A.json",
        "splitA_abc": output_dir / "splitA_ABCjoint.json",
        "splitB_abc": output_dir / "splitB_ABCjoint.json",
        "splitC_abc": output_dir / "splitC_ABCjoint.json",
    }

    run_offline_eval(
        args.python_bin,
        args.offline_script,
        args.splitD_root,
        args.splitD_repo,
        paths["splitD_a"],
        a_ckpts,
        args.batch_size,
        args.max_samples,
        args.device,
        args.skip_existing,
    )
    run_offline_eval(
        args.python_bin,
        args.offline_script,
        args.splitD_root,
        args.splitD_repo,
        paths["splitD_abc"],
        abc_ckpts,
        args.batch_size,
        args.max_samples,
        args.device,
        args.skip_existing,
    )
    run_visual_shift_eval(
        args.python_bin,
        args.visual_shift_script,
        paths["visual_shift_a"],
        a_ckpts,
        args.batch_size,
        args.samples_per_family,
        args.device,
        args.seed,
        args.splitA_root,
        args.splitD_root,
        args.splitA_repo,
        args.splitD_repo,
        args.skip_existing,
    )
    run_visual_shift_eval(
        args.python_bin,
        args.visual_shift_script,
        paths["visual_shift_abc"],
        abc_ckpts,
        args.batch_size,
        args.samples_per_family,
        args.device,
        args.seed,
        args.splitA_root,
        args.splitD_root,
        args.splitA_repo,
        args.splitD_repo,
        args.skip_existing,
    )
    run_offline_eval(
        args.python_bin,
        args.offline_script,
        args.splitA_root,
        args.splitA_repo,
        paths["splitA_abc"],
        abc_ckpts,
        args.batch_size,
        args.max_samples,
        args.device,
        args.skip_existing,
    )
    run_offline_eval(
        args.python_bin,
        args.offline_script,
        args.splitB_root,
        args.splitB_repo,
        paths["splitB_abc"],
        abc_ckpts,
        args.batch_size,
        args.max_samples,
        args.device,
        args.skip_existing,
    )
    run_offline_eval(
        args.python_bin,
        args.offline_script,
        args.splitC_root,
        args.splitC_repo,
        paths["splitC_abc"],
        abc_ckpts,
        args.batch_size,
        args.max_samples,
        args.device,
        args.skip_existing,
    )

    splitD_a = load_json(paths["splitD_a"])["results"]
    splitD_abc = load_json(paths["splitD_abc"])["results"]
    visual_shift_a = load_json(paths["visual_shift_a"])["results"]
    visual_shift_abc = load_json(paths["visual_shift_abc"])["results"]
    splitA_abc = load_json(paths["splitA_abc"])["results"]
    splitB_abc = load_json(paths["splitB_abc"])["results"]
    splitC_abc = load_json(paths["splitC_abc"])["results"]

    summary = {
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "samples_per_family": args.samples_per_family,
            "device": args.device,
            "seed": args.seed,
            "split_roots": {
                "splitA": args.splitA_root,
                "splitB": args.splitB_root,
                "splitC": args.splitC_root,
                "splitD": args.splitD_root,
            },
        },
        "runs": {
            "Aonly": {
                "run_dir": str(a_run_dir),
                "checkpoints": {},
            },
            "ABCjoint": {
                "run_dir": str(abc_run_dir),
                "checkpoints": {},
            },
        },
    }

    for step in args.steps:
        a_key = f"A{step:04d}"
        abc_key = f"ABC{step:04d}"

        summary["runs"]["Aonly"]["checkpoints"][str(step)] = {
            "checkpoint_dir": a_ckpts[a_key],
            "splitD_zero_shot": splitD_a[a_key],
            "visual_shift": visual_shift_a[a_key],
        }

        abc_mean = (
            splitA_abc[abc_key]["avg_l1_loss"]
            + splitB_abc[abc_key]["avg_l1_loss"]
            + splitC_abc[abc_key]["avg_l1_loss"]
        ) / 3.0
        d_minus_abc_mean = splitD_abc[abc_key]["avg_l1_loss"] - abc_mean

        summary["runs"]["ABCjoint"]["checkpoints"][str(step)] = {
            "checkpoint_dir": abc_ckpts[abc_key],
            "splitD_zero_shot": splitD_abc[abc_key],
            "visual_shift": visual_shift_abc[abc_key],
            "splitA_seen": splitA_abc[abc_key],
            "splitB_seen": splitB_abc[abc_key],
            "splitC_seen": splitC_abc[abc_key],
            "ABC_mean_avg_l1_loss": abc_mean,
            "D_minus_ABC_mean_delta_l1_loss": d_minus_abc_mean,
        }

    (output_dir / "task2_2000_rerun_eval_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
