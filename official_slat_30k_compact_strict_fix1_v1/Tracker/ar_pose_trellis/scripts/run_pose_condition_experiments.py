#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


TRACKER_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TRACKER_DATA_ROOT", "/data/zjr")).expanduser()
DEFAULT_PYTHON = "/home/zjr/anaconda3/envs/reconviagen/bin/python"
DEFAULT_DATA_ROOT = str(DATA_ROOT / "ar_pose_trellis" / "objaverse_pose_1000_meshrgb_s2")
DEFAULT_TESTSETS = (
    "/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/objaverse_meshrgb_val_testsets.json"
)
DEFAULT_RUN_ROOT = "/home/zjr/Tracker/ar_pose_trellis/outputs/training_runs/pose_condition_experiments"
DEFAULT_OUTPUT_ROOT = "/home/zjr/Tracker/ar_pose_trellis/outputs/benchmarks/pose_condition_experiments"


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_noise_settings(text: str) -> list[tuple[float, float]]:
    out = []
    for item in parse_csv(text):
        if ":" not in item:
            raise ValueError(f"Noise setting should be ROT_DEG:TRANS_STD, got {item!r}")
        rot, trans = item.split(":", 1)
        out.append((float(rot), float(trans)))
    return out


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def emit(progress_log: Path, message: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    append_text(progress_log, line + "\n")


def command_text(cmd: list[str]) -> str:
    return " ".join(cmd)


def should_show_child_output(mode: str) -> bool:
    if mode == "inherit":
        return True
    if mode == "quiet":
        return False
    if mode == "auto":
        return sys.stdout.isatty()
    raise ValueError(f"Unknown child output mode: {mode}")


def run_command(
    task: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
    *,
    child_output: str,
    save_raw_logs: bool,
) -> dict[str, Any]:
    log_path = Path(task["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path = log_path.with_name(f"{log_path.stem}.raw.log")
    result = {
        "name": task["name"],
        "kind": task["kind"],
        "status": "pending",
        "command": task["cmd"],
        "log_path": str(log_path),
        "output_root": task.get("output_root"),
        "checkpoint": task.get("checkpoint"),
    }
    if save_raw_logs:
        result["raw_log_path"] = str(raw_log_path)
    start_time = time.time()
    result["start_time"] = _dt.datetime.now().isoformat(timespec="seconds")

    if dry_run:
        result["status"] = "dry_run"
        log_path.write_text(command_text(task["cmd"]) + "\n", encoding="utf-8")
        return result

    log_path.write_text(
        "[command]\n"
        + command_text(task["cmd"])
        + "\n\n"
        + f"[start] {result['start_time']}\n",
        encoding="utf-8",
    )

    if child_output == "inherit" and not save_raw_logs:
        completed = subprocess.run(
            task["cmd"],
            cwd=str(TRACKER_ROOT),
            env=env,
            check=False,
        )
        returncode = completed.returncode
        output_tail: deque[str] = deque(maxlen=0)
    else:
        raw_log = raw_log_path.open("w", encoding="utf-8") if save_raw_logs else None
        output_tail = deque(maxlen=120)
        show_child = should_show_child_output(child_output)
        try:
            proc = subprocess.Popen(
                task["cmd"],
                cwd=str(TRACKER_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                output_tail.append(line.rstrip("\n"))
                if raw_log is not None:
                    raw_log.write(line)
                if show_child:
                    print(line, end="", flush=True)
            returncode = proc.wait()
        finally:
            if raw_log is not None:
                raw_log.close()

    result["returncode"] = int(returncode)
    result["elapsed_sec"] = round(time.time() - start_time, 2)
    result["end_time"] = _dt.datetime.now().isoformat(timespec="seconds")
    if returncode == 0:
        result["status"] = "ok"
    else:
        result["status"] = "failed"
        result["error_tail"] = "\n".join(output_tail)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[end] {result['end_time']}\n")
        log.write(f"[status] {result['status']}\n")
        log.write(f"[returncode] {result.get('returncode')}\n")
        log.write(f"[elapsed_sec] {result['elapsed_sec']}\n")
        if result["status"] == "failed":
            log.write("\n[error_tail]\n")
            log.write(str(result.get("error_tail", "")) + "\n")
    return result


def mean_numeric(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def summarize_eval_report(payload: dict[str, Any]) -> dict[str, Any]:
    wanted = [
        "voxel_iou",
        "chamfer_vox",
        "fscore_1vox",
        "fscore_2vox",
        "fscore_4vox",
        "pred_count",
        "ref_count",
    ]
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for item in payload.get("reports", []):
        mode = str(item.get("mode", "unknown"))
        metrics = item.get("metrics", {})
        for key in wanted:
            grouped[mode][key].append(metrics.get(key))

    summary: dict[str, Any] = {}
    for mode, values in grouped.items():
        mode_summary: dict[str, Any] = {"n": len(values.get("voxel_iou", []))}
        for key in wanted:
            mean_value = mean_numeric(values.get(key, []))
            if mean_value is not None:
                mode_summary[key] = round(mean_value, 6)
        summary[mode] = mode_summary
    return summary


def format_metric_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    parts = []
    for mode in ["correct", "identity", "shuffle", "noise"]:
        if mode not in summary:
            continue
        item = summary[mode]
        bits = [mode]
        if "voxel_iou" in item:
            bits.append(f"iou={item['voxel_iou']:.4f}")
        if "chamfer_vox" in item:
            bits.append(f"chamfer={item['chamfer_vox']:.3f}")
        if "fscore_2vox" in item:
            bits.append(f"f2={item['fscore_2vox']:.4f}")
        if "pred_count" in item:
            bits.append(f"pred={item['pred_count']:.0f}")
        parts.append(" ".join(bits))
    return "; ".join(parts)


def format_result_line(result: dict[str, Any]) -> str:
    name = result.get("name")
    status = result.get("status")
    elapsed = result.get("elapsed_sec")
    elapsed_text = f" elapsed={elapsed}s" if elapsed is not None else ""
    if status == "ok" and result.get("kind") == "eval":
        metric_text = format_metric_summary(result.get("metric_summary", {}))
        suffix = f" | {metric_text}" if metric_text else ""
        return f"[ok] {name}{elapsed_text}{suffix}"
    if status == "ok":
        checkpoint = result.get("checkpoint")
        suffix = f" checkpoint={checkpoint}" if checkpoint else ""
        return f"[ok] {name}{elapsed_text}{suffix}"
    if status == "dry_run":
        return f"[dry_run] {name} log={result.get('log_path')}"
    if status == "skipped":
        return f"[skip] {name}: {result.get('reason', '')}"
    if status == "report_failures":
        return f"[report_failures] {name}{elapsed_text} failures={result.get('num_report_failures', 0)}"
    return f"[failed] {name}{elapsed_text} log={result.get('log_path')}"


def append_task_summary_log(result: dict[str, Any]) -> None:
    log_path = result.get("log_path")
    if not log_path:
        return
    metric_text = format_metric_summary(result.get("metric_summary", {}))
    if metric_text:
        append_text(Path(log_path), "\n[metric_summary]\n" + metric_text + "\n")


def safe_run_command(
    task: dict[str, Any],
    env: dict[str, str],
    dry_run: bool,
    *,
    child_output: str,
    save_raw_logs: bool,
) -> dict[str, Any]:
    try:
        return run_command(
            task,
            env,
            dry_run,
            child_output=child_output,
            save_raw_logs=save_raw_logs,
        )
    except Exception as exc:
        return {
            "name": task["name"],
            "kind": task["kind"],
            "status": "failed",
            "command": task["cmd"],
            "log_path": task.get("log_path"),
            "output_root": task.get("output_root"),
            "checkpoint": task.get("checkpoint"),
            "error_tail": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def inspect_eval_report(result: dict[str, Any]) -> dict[str, Any]:
    output_root = result.get("output_root")
    if not output_root:
        return result
    report_path = Path(output_root) / "sparse_pose_ablation_report.json"
    result["report_path"] = str(report_path)
    if result["status"] != "ok" or not report_path.exists():
        return result
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["status"] = "failed"
        result["error_tail"] = f"Could not read report {report_path}: {type(exc).__name__}: {exc}"
        return result
    failures = payload.get("failures", [])
    result["num_reports"] = len(payload.get("reports", []))
    result["num_report_failures"] = len(failures)
    result["metric_summary"] = summarize_eval_report(payload)
    if failures:
        result["status"] = "report_failures"
        result["report_failures"] = failures
    return result


def variant_flags(variant: str) -> list[str]:
    if variant == "image_pose":
        return []
    if variant == "image_only":
        return ["--image_only"]
    if variant == "pose_only":
        return ["--pose_only"]
    raise ValueError(f"Unknown train variant: {variant}")


def train_task(args: argparse.Namespace, run_dir: Path, log_dir: Path, variant: str) -> dict[str, Any]:
    save_dir = run_dir / "checkpoints" / variant
    cmd = [
        args.python,
        "ar_pose_trellis/train_ss_ar_pose.py",
        "--dataset_format",
        "objaverse_pose",
        "--data_root",
        args.data_root,
        "--weights",
        args.weights,
        "--save_dir",
        str(save_dir),
        "--num_views",
        str(args.num_views),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--max_epochs",
        str(args.max_epochs),
        "--lr",
        str(args.lr),
        "--cfg_drop_prob",
        str(args.cfg_drop_prob),
        "--ckpt_every_n_steps",
        str(args.ckpt_every_n_steps),
    ]
    cmd.extend(variant_flags(variant))
    return {
        "name": f"train_{variant}",
        "kind": "train",
        "cmd": cmd,
        "log_path": str(log_dir / f"train_{variant}.log"),
        "checkpoint": str(save_dir / "last.ckpt"),
    }


def eval_task(
    args: argparse.Namespace,
    output_dir: Path,
    log_dir: Path,
    variant: str,
    checkpoint: Path,
    name: str,
    *,
    modes: str,
    max_frames: int,
    visual_hull_prior_weight: float | None = None,
    noise_rot_deg: float | None = None,
    noise_trans_std: float | None = None,
) -> dict[str, Any]:
    task_output = output_dir / variant / name
    cmd = [
        args.python,
        "ar_pose_trellis/benchmark/evaluate_sparse_pose_ablation.py",
        "--testsets",
        args.testsets,
        "--checkpoint",
        str(checkpoint),
        "--output_root",
        str(task_output),
        "--weights",
        args.weights,
        "--modes",
        modes,
        "--max_frames",
        str(max_frames),
        "--ss_steps",
        str(args.ss_steps),
        "--ss_min_coords",
        str(args.ss_min_coords),
        "--cond_fp16",
        "--inprocess",
        "--continue_on_error",
    ]
    cmd.extend(variant_flags(variant))
    if visual_hull_prior_weight is not None and float(visual_hull_prior_weight) != 0.0:
        cmd.extend(["--visual_hull_prior_weight", str(visual_hull_prior_weight)])
    if noise_rot_deg is not None:
        cmd.extend(["--noise_rot_deg", str(noise_rot_deg)])
    if noise_trans_std is not None:
        cmd.extend(["--noise_trans_std", str(noise_trans_std)])
    return {
        "name": f"eval_{variant}_{name}",
        "kind": "eval",
        "cmd": cmd,
        "log_path": str(log_dir / f"eval_{variant}_{name}.log"),
        "output_root": str(task_output),
        "checkpoint": str(checkpoint),
    }


def skipped_task(name: str, kind: str, reason: str, checkpoint: Path | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "status": "skipped",
        "reason": reason,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def failed_task(name: str, kind: str, error: str, checkpoint: Path | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "status": "failed",
        "error_tail": error,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def provided_checkpoint(args: argparse.Namespace, variant: str) -> Path | None:
    mapping = {
        "image_pose": args.checkpoint_image_pose,
        "image_only": args.checkpoint_image_only,
        "pose_only": args.checkpoint_pose_only,
    }
    value = mapping.get(variant)
    return Path(value).expanduser().resolve() if value else None


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    return env


def write_failure_report(path: Path, results: list[dict[str, Any]]) -> None:
    failures = [r for r in results if r.get("status") in {"failed", "report_failures"}]
    lines = []
    lines.append(f"total_tasks: {len(results)}")
    lines.append(f"failures: {len(failures)}")
    lines.append("")
    for item in results:
        lines.append(f"[{item.get('status')}] {item.get('name')} ({item.get('kind')})")
        if item.get("reason"):
            lines.append(f"  reason: {item['reason']}")
        if item.get("log_path"):
            lines.append(f"  log: {item['log_path']}")
        if item.get("report_path"):
            lines.append(f"  report: {item['report_path']}")
        if item.get("status") == "failed":
            lines.append("  error_tail:")
            for line in str(item.get("error_tail", "")).splitlines()[-30:]:
                lines.append(f"    {line}")
        if item.get("status") == "report_failures":
            lines.append("  report_failures:")
            for failure in item.get("report_failures", [])[:20]:
                lines.append(f"    {failure}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_result_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 位姿条件实验最终结果",
        "",
        f"- exp_name: `{summary['exp_name']}`",
        f"- run_dir: `{summary['run_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- total_tasks: {summary['num_tasks']}",
        f"- failed_or_report_failed: {summary['num_failed']}",
        f"- skipped: {summary['num_skipped']}",
        "",
        "| 状态 | 类型 | 任务 | 耗时(s) | 关键结果 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for item in summary["results"]:
        status = item.get("status", "")
        kind = item.get("kind", "")
        name = item.get("name", "")
        elapsed = item.get("elapsed_sec", "")
        metric_text = format_metric_summary(item.get("metric_summary", {}))
        if not metric_text:
            if item.get("checkpoint"):
                metric_text = f"checkpoint: `{item['checkpoint']}`"
            elif item.get("reason"):
                metric_text = str(item["reason"])
            elif item.get("output_root"):
                metric_text = f"output: `{item['output_root']}`"
        lines.append(f"| {status} | {kind} | {name} | {elapsed} | {metric_text} |")

    failures = [r for r in summary["results"] if r.get("status") in {"failed", "report_failures"}]
    if failures:
        lines.extend(["", "## 失败任务", ""])
        for item in failures:
            lines.append(f"### {item.get('name')}")
            lines.append("")
            if item.get("log_path"):
                lines.append(f"- log: `{item['log_path']}`")
            if item.get("report_path"):
                lines.append(f"- report: `{item['report_path']}`")
            if item.get("error_tail"):
                lines.append("")
                lines.append("```text")
                lines.extend(str(item["error_tail"]).splitlines()[-40:])
                lines.append("```")
            if item.get("report_failures"):
                lines.append("")
                lines.append("```text")
                for failure in item.get("report_failures", [])[:20]:
                    lines.append(str(failure))
                lines.append("```")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_final_results(summary: dict[str, Any], final_report: Path, failure_report: Path) -> None:
    print("\n========== 位姿条件实验最终结果 ==========", flush=True)
    for item in summary["results"]:
        print(format_result_line(item), flush=True)
    print("========================================", flush=True)
    print(f"summary: {summary['run_dir']}/summary.json", flush=True)
    print(f"final report: {final_report}", flush=True)
    print(f"failure report: {failure_report}", flush=True)
    print(f"output root: {summary['output_dir']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AR-pose condition training variants and sparse-pose evaluation sweeps."
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--testsets", default=DEFAULT_TESTSETS)
    parser.add_argument("--run_root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--variants", default="image_pose,image_only,pose_only")
    parser.add_argument(
        "--checkpoint_image_pose",
        default="",
        help="Use an existing image+pose checkpoint and skip training this variant.",
    )
    parser.add_argument(
        "--checkpoint_image_only",
        default="",
        help="Use an existing image-only checkpoint and skip training this variant.",
    )
    parser.add_argument(
        "--checkpoint_pose_only",
        default="",
        help="Use an existing pose-only checkpoint and skip training this variant.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--child_output",
        choices=["auto", "inherit", "quiet"],
        default="auto",
        help="auto shows child stdout only in an interactive terminal; nohup/redirected runs stay concise.",
    )
    parser.add_argument(
        "--save_raw_logs",
        action="store_true",
        help="Also save full child stdout/stderr to *.raw.log for debugging.",
    )

    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--ckpt_every_n_steps", type=int, default=1000)

    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_min_coords", type=int, default=0)
    parser.add_argument("--eval_max_frames", type=int, default=8)
    parser.add_argument("--visual_hull_weights", default="0,5,10,20,40,80")
    parser.add_argument("--noise_settings", default="5:0.02,15:0.05,30:0.10,60:0.20")
    parser.add_argument("--view_counts", default="2,4,6,8,12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = parse_csv(args.variants)
    allowed_variants = {"image_pose", "image_only", "pose_only"}
    bad_variants = [v for v in variants if v not in allowed_variants]
    if bad_variants:
        raise ValueError(f"Unsupported variants {bad_variants}; allowed={sorted(allowed_variants)}")

    visual_hull_weights = parse_csv(args.visual_hull_weights, float)
    noise_settings = parse_noise_settings(args.noise_settings)
    view_counts = parse_csv(args.view_counts, int)

    exp_name = args.exp_name.strip() or f"pose_condition_{now_stamp()}"
    run_dir = Path(args.run_root) / exp_name
    output_dir = Path(args.output_root) / exp_name
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    results: list[dict[str, Any]] = []
    checkpoint_by_variant: dict[str, Path] = {}

    config = {
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "variants": variants,
        "visual_hull_weights": visual_hull_weights,
        "noise_settings": noise_settings,
        "view_counts": view_counts,
        "args": vars(args),
    }
    write_json(run_dir / "config.json", config)
    progress_log = run_dir / "实验进度.log"
    progress_log.write_text("", encoding="utf-8")
    emit(progress_log, f"[start] exp={exp_name} run_dir={run_dir} output_dir={output_dir}")

    if not args.skip_train:
        for variant in variants:
            provided_ckpt = provided_checkpoint(args, variant)
            if provided_ckpt is not None:
                if provided_ckpt.exists():
                    checkpoint_by_variant[variant] = provided_ckpt
                    results.append(
                        skipped_task(
                            f"train_{variant}",
                            "train",
                            f"using provided checkpoint: {provided_ckpt}",
                            provided_ckpt,
                        )
                    )
                    emit(progress_log, format_result_line(results[-1]))
                else:
                    results.append(
                        failed_task(
                            f"train_{variant}",
                            "train",
                            f"provided checkpoint does not exist: {provided_ckpt}",
                            provided_ckpt,
                        )
                    )
                    emit(progress_log, format_result_line(results[-1]))
                continue

            task = train_task(args, run_dir, log_dir, variant)
            emit(progress_log, f"[start] {task['name']}")
            result = safe_run_command(
                task,
                env,
                args.dry_run,
                child_output=args.child_output,
                save_raw_logs=args.save_raw_logs,
            )
            results.append(result)
            emit(progress_log, format_result_line(result))
            ckpt = Path(task["checkpoint"])
            if result["status"] == "ok" and ckpt.exists():
                checkpoint_by_variant[variant] = ckpt
    else:
        for variant in variants:
            ckpt = provided_checkpoint(args, variant) or (run_dir / "checkpoints" / variant / "last.ckpt")
            if ckpt.exists():
                checkpoint_by_variant[variant] = ckpt
            results.append(skipped_task(f"train_{variant}", "train", "--skip_train was set", ckpt))
            emit(progress_log, format_result_line(results[-1]))

    if not args.skip_eval:
        for variant in variants:
            ckpt = checkpoint_by_variant.get(variant)
            if ckpt is None or not ckpt.exists():
                reason = f"missing checkpoint for {variant}"
                results.append(skipped_task(f"eval_{variant}_all", "eval", reason))
                emit(progress_log, format_result_line(results[-1]))
                continue

            eval_tasks = []
            eval_tasks.append(
                eval_task(
                    args,
                    output_dir,
                    log_dir,
                    variant,
                    ckpt,
                    "raw",
                    modes="correct,identity,shuffle,noise",
                    max_frames=args.eval_max_frames,
                )
            )
            for weight in visual_hull_weights:
                if float(weight) == 0.0:
                    continue
                eval_tasks.append(
                    eval_task(
                        args,
                        output_dir,
                        log_dir,
                        variant,
                        ckpt,
                        f"vh_w{weight:g}",
                        modes="correct,identity,shuffle,noise",
                        max_frames=args.eval_max_frames,
                        visual_hull_prior_weight=weight,
                    )
                )
            for rot_deg, trans_std in noise_settings:
                eval_tasks.append(
                    eval_task(
                        args,
                        output_dir,
                        log_dir,
                        variant,
                        ckpt,
                        f"noise_r{rot_deg:g}_t{trans_std:g}",
                        modes="correct,noise",
                        max_frames=args.eval_max_frames,
                        noise_rot_deg=rot_deg,
                        noise_trans_std=trans_std,
                    )
                )
            for view_count in view_counts:
                eval_tasks.append(
                    eval_task(
                        args,
                        output_dir,
                        log_dir,
                        variant,
                        ckpt,
                        f"views_{view_count}",
                        modes="correct,identity,shuffle,noise",
                        max_frames=view_count,
                    )
                )

            for task in eval_tasks:
                emit(progress_log, f"[start] {task['name']}")
                result = safe_run_command(
                    task,
                    env,
                    args.dry_run,
                    child_output=args.child_output,
                    save_raw_logs=args.save_raw_logs,
                )
                result = inspect_eval_report(result)
                append_task_summary_log(result)
                results.append(result)
                emit(progress_log, format_result_line(result))
    else:
        results.append(skipped_task("eval_all", "eval", "--skip_eval was set"))
        emit(progress_log, format_result_line(results[-1]))

    summary = {
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "num_tasks": len(results),
        "num_failed": sum(1 for r in results if r.get("status") in {"failed", "report_failures"}),
        "num_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "results": results,
    }
    write_json(run_dir / "summary.json", summary)
    final_report = run_dir / "最终结果.md"
    failure_report = run_dir / "失败汇总.txt"
    write_final_result_report(final_report, summary)
    write_failure_report(failure_report, results)
    emit(progress_log, f"[done] summary: {run_dir / 'summary.json'}")
    emit(progress_log, f"[done] final report: {final_report}")
    emit(progress_log, f"[done] failure report: {failure_report}")
    emit(progress_log, f"[done] output root: {output_dir}")
    print_final_results(summary, final_report, failure_report)


if __name__ == "__main__":
    main()
