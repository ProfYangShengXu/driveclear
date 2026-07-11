"""
消融实验框架 (Ablation Study) — M9

通过逐一 disable 管线模块，量化每个模块的贡献。

对每组配置：
  1. 使用 M6 PipelineOrchestrator + 对应 PipelineConfig 处理视频
  2. 使用 M7 evaluate_video 评分
  3. 输出消融对比表

固定的消融配置集：
  - full:      全部开启
  - no_fog:    关闭去雾
  - no_glare:  关闭眩光
  - no_ll:     关闭低光
  - no_fusion: 关闭自适应融合（DCP only）
  - no_enhance: 关闭后处理增强
  - baseline:  全部关闭（直接输出原片）
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.processing_service import PipelineConfig, PipelineOrchestrator
from scripts.evaluate import evaluate_video, evaluate_frame_pair


# ─── 预定义的消融配置集 ──────────────────────────────────────────────────


ABLATION_CONFIGS = {
    "full": {
        "label": "完整流水线",
        "config": {},
    },
    "no_fog": {
        "label": "去掉去雾",
        "config": {
            "enable_fog": False,
            "enable_glare": True,
            "enable_low_light": True,
            "enable_fusion": True,
            "enable_enhance": True,
        },
    },
    "no_glare": {
        "label": "去掉眩光",
        "config": {
            "enable_fog": True,
            "enable_glare": False,
            "enable_low_light": True,
            "enable_fusion": True,
            "enable_enhance": True,
        },
    },
    "no_ll": {
        "label": "去掉低光增强",
        "config": {
            "enable_fog": True,
            "enable_glare": True,
            "enable_low_light": False,
            "enable_fusion": True,
            "enable_enhance": True,
        },
    },
    "no_fusion": {
        "label": "去掉自适应融合",
        "config": {
            "enable_fog": True,
            "enable_glare": True,
            "enable_low_light": True,
            "enable_fusion": False,
            "enable_enhance": True,
        },
    },
    "no_enhance": {
        "label": "去掉后处理增强",
        "config": {
            "enable_fog": True,
            "enable_glare": True,
            "enable_low_light": True,
            "enable_fusion": True,
            "enable_enhance": False,
        },
    },
    "dcp_only": {
        "label": "仅 DCP 去雾",
        "config": {
            "enable_fog": True,
            "enable_glare": False,
            "enable_low_light": False,
            "enable_fusion": False,
            "enable_enhance": False,
            "auto_detect": False,
        },
    },
    "none": {
        "label": "无处理（原片）",
        "config": {
            "enable_fog": False,
            "enable_glare": False,
            "enable_low_light": False,
            "enable_fusion": False,
            "enable_enhance": False,
            "auto_detect": False,
        },
    },
}


def process_with_config(
    input_path: str,
    output_path: str,
    config: dict,
) -> str:
    """使用指定配置处理视频

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        config: PipelineConfig 字段

    Returns:
        output_path
    """
    cfg = PipelineConfig.from_dict(config)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    orchestrator = PipelineOrchestrator(cfg, h, w)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = orchestrator.process_frame(frame)
        writer.write(result)

    cap.release()
    writer.release()
    return output_path


def run_ablation(
    input_video: str,
    output_dir: str,
    configs: list[dict] | None = None,
    original_video: str | None = None,
    max_eval_frames: int = 200,
) -> dict:
    """运行消融实验

    Args:
        input_video: 输入退化视频路径
        output_dir: 输出目录
        configs: 自定义配置列表，None=使用预定义 ABLATION_CONFIGS
        original_video: 原始（无退化）视频，用于全参考评估
        max_eval_frames: 评估帧上限

    Returns:
        { "ablation_table": [...], "details": {...} }
    """
    os.makedirs(output_dir, exist_ok=True)

    if configs is None:
        configs_to_run = ABLATION_CONFIGS
    else:
        configs_to_run = {f"custom_{i}": {"label": f"配置{i}", "config": c}
                          for i, c in enumerate(configs)}

    ablation_table = []
    details = {}

    for key, spec in configs_to_run.items():
        label = spec["label"]
        config = spec["config"]
        out_path = os.path.join(output_dir, f"{key}.mp4")

        print(f"[{key}] {label}...")
        try:
            process_with_config(input_video, out_path, config)
            print(f"  -> {out_path}")

            entry = {
                "id": key,
                "label": label,
                "disabled_modules": _describe_disabled(config),
            }

            if original_video and os.path.exists(original_video):
                eval_result = evaluate_video(original_video, out_path, max_frames=max_eval_frames)
                entry["psnr"] = eval_result["psnr_mean"]
                entry["ssim"] = eval_result["ssim_mean"]
                entry["psnr_std"] = eval_result["psnr_std"]
                entry["ssim_std"] = eval_result["ssim_std"]
                print(f"  PSNR={entry['psnr']:.2f}, SSIM={entry['ssim']:.4f}")
            else:
                entry["psnr"] = None
                entry["ssim"] = None

            ablation_table.append(entry)
            details[key] = {"output": out_path, "config": config}

        except Exception as e:
            print(f"  FAILED: {e}")
            ablation_table.append({
                "id": key,
                "label": label,
                "error": str(e),
            })

    # 按 PSNR 降序排序（如果存在）
    valid = [e for e in ablation_table if e.get("psnr") is not None]
    valid.sort(key=lambda x: x["psnr"], reverse=True)
    for rank, entry in enumerate(valid, 1):
        entry["rank"] = rank

    result = {
        "ablation_table": ablation_table,
        "details": details,
    }

    summary_path = os.path.join(output_dir, "ablation_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSummary -> {summary_path}")

    return result


def _describe_disabled(config: dict) -> list[str]:
    """human-readable 描述哪些模块被禁用"""
    disabled = []
    mapping = {
        "enable_fog": "去雾",
        "enable_glare": "眩光抑制",
        "enable_low_light": "低光增强",
        "enable_fusion": "自适应融合",
        "enable_enhance": "后处理增强",
    }
    for key, label in mapping.items():
        if not config.get(key, True):
            disabled.append(label)
    if not config.get("auto_detect", True):
        disabled.append("自动检测")
    return disabled if disabled else ["无（全开）"]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/ablation.py <input.mp4> <output_dir> [original.mp4]")
        sys.exit(1)

    input_video = sys.argv[1]
    output_dir = sys.argv[2]
    original_video = sys.argv[3] if len(sys.argv) > 3 else None
    run_ablation(input_video, output_dir, original_video=original_video)
