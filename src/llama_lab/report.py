from __future__ import annotations

import csv
import json
from pathlib import Path


def _fmt(value: str | float, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_report(results_dir: Path, output_path: Path) -> None:
    with (results_dir / "baseline.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline = list(csv.DictReader(handle))
    with (results_dir / "server_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        server = list(csv.DictReader(handle))
    with (results_dir / "quality_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        quality = list(csv.DictReader(handle))
    engine_ab_path = results_dir / "engine_ab.csv"
    if engine_ab_path.exists():
        with engine_ab_path.open(encoding="utf-8-sig", newline="") as handle:
            engine_ab = list(csv.DictReader(handle))
    else:
        engine_ab = []
    environment = json.loads(
        (results_dir / "environment.json").read_text(encoding="utf-8")
    )
    build_commit = baseline[0].get("build_commit", "unknown") if baseline else "unknown"
    build_number = baseline[0].get("build_number", "unknown") if baseline else "unknown"

    def find_baseline(
        case: str, quantization: str, test: str
    ) -> dict[str, str] | None:
        return next(
            (
                row
                for row in baseline
                if row["case"] == case
                and row["quantization"] == quantization
                and row["test"] == test
            ),
            None,
        )

    q4_gpu_tg = find_baseline("gpu-quantization", "Q4_K_M", "tg64")
    f16_gpu_tg = find_baseline("gpu-quantization", "F16", "tg64")
    q4_cpu_tg = find_baseline("cpu-thread-12", "Q4_K_M", "tg64")
    observations: list[str] = []
    if q4_gpu_tg and f16_gpu_tg:
        q4_speedup_over_f16 = float(q4_gpu_tg["avg_ts"]) / float(
            f16_gpu_tg["avg_ts"]
        )
        q4_size_ratio = float(q4_gpu_tg["model_size"]) / float(
            f16_gpu_tg["model_size"]
        )
        observations.append(
            f"- Q4_K_M 权重大小是 F16 的 {q4_size_ratio:.1%}，CUDA decode 速度是 F16 的 {q4_speedup_over_f16:.2f} 倍。"
        )
    if q4_gpu_tg and q4_cpu_tg:
        gpu_speedup_over_cpu = float(q4_gpu_tg["avg_ts"]) / float(
            q4_cpu_tg["avg_ts"]
        )
        observations.append(
            f"- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 {gpu_speedup_over_cpu:.2f} 倍。"
        )
    server_by_concurrency = {int(row["concurrency"]): row for row in server}
    lowest_concurrency = min(server_by_concurrency)
    highest_concurrency = max(server_by_concurrency)
    aggregate_scaling = float(
        server_by_concurrency[highest_concurrency]["aggregate_output_tps"]
    ) / float(server_by_concurrency[lowest_concurrency]["aggregate_output_tps"])
    observations.append(
        f"- 并发从 {lowest_concurrency} 增至 {highest_concurrency} 时，聚合输出吞吐提高到 {aggregate_scaling:.2f} 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。"
    )
    if len(engine_ab) >= 2:
        upstream = min(engine_ab, key=lambda row: float(row["eviction_penalty"]))
        cache_aware = max(engine_ab, key=lambda row: float(row["eviction_penalty"]))
        sequence_speedup = float(upstream["sequence_wall_ms_median"]) / float(
            cache_aware["sequence_wall_ms_median"]
        )
        prefill_reduction = 1.0 - float(
            cache_aware["sequence_prompt_processed_tokens_median"]
        ) / float(upstream["sequence_prompt_processed_tokens_median"])
        eviction_reduction = 1.0 - float(
            cache_aware["selection_evicted_tokens_median"]
        ) / float(upstream["selection_evicted_tokens_median"])
        observations.append(
            f"- Cache-aware 调度在冲突序列中减少 {prefill_reduction:.1%} 的重复 prefill token、减少 {eviction_reduction:.1%} 的缓存淘汰，中位序列延迟提升 {sequence_speedup:.2f} 倍。"
        )
    quality_scores = [float(row["accuracy"]) for row in quality]
    quality_total = max((int(row["total"]) for row in quality), default=0)

    lines = [
        "# llama.cpp 可复现基线报告",
        "",
        "> 该报告由实验脚本从原始 JSON/CSV 自动生成；速度只代表当前机器、固定版本和固定配置。",
        "",
        "## 环境",
        "",
        f"- 平台：`{environment.get('platform', 'unknown')}`",
        f"- Python：`{environment.get('python', 'unknown')}`",
        f"- GPU：`{environment.get('gpu', 'unknown')}`",
        f"- llama.cpp：`b{build_number}` / `{build_commit}`",
        "",
        "## 离线算子基线",
        "",
        "| Case | 量化 | 后端 | 测试 | tokens/s | 标准差 | 模型 MiB | Run 峰值 VRAM MiB | Run 增量 VRAM MiB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        model_mib = float(row.get("model_size", 0)) / (1024 * 1024)
        lines.append(
            "| {case} | {quant} | {backend} | {test} | {avg} | {std} | {size:.1f} | {peak} | {increment} |".format(
                case=row.get("case", ""),
                quant=row.get("quantization", ""),
                backend=row.get("execution", row.get("backends", "")),
                test=row.get("test", ""),
                avg=_fmt(row.get("avg_ts", "")),
                std=_fmt(row.get("stddev_ts", "")),
                size=model_mib,
                peak=_fmt(row.get("gpu_memory_peak_mib", ""), 0),
                increment=_fmt(row.get("gpu_memory_increment_mib", ""), 0),
            )
        )
    lines.extend(
        [
            "",
            "`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。",
            "",
            "## 在线流式服务",
            "",
            "| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in server:
        lines.append(
            "| {c} | {n} | {p50} | {p95} | {tpot} | {total} | {mean} | {aggregate} | {peak} | {increment} |".format(
                c=row["concurrency"],
                n=_fmt(row["requests"], 0),
                p50=_fmt(row["ttft_p50_ms"]),
                p95=_fmt(row["ttft_p95_ms"]),
                tpot=_fmt(row["tpot_p95_ms"]),
                total=_fmt(row["total_p95_ms"]),
                mean=_fmt(row["mean_output_tps"]),
                aggregate=_fmt(row["aggregate_output_tps"]),
                peak=_fmt(row.get("gpu_memory_peak_mib", ""), 0),
                increment=_fmt(row.get("gpu_memory_increment_mib", ""), 0),
            )
        )
    if engine_ab:
        lines.extend(
            [
                "",
                "## C++ KV Cache 调度 A/B",
                "",
                "| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in engine_ab:
            lines.append(
                "| {penalty} | {trials} | {target} | {followup} | {sequence} | {tokens} | {evicted} |".format(
                    penalty=_fmt(row["eviction_penalty"]),
                    trials=row["trials"],
                    target=_fmt(row["target_wall_ms_median"]),
                    followup=_fmt(row["followup_wall_ms_median"]),
                    sequence=_fmt(row["sequence_wall_ms_median"]),
                    tokens=_fmt(row["sequence_prompt_processed_tokens_median"], 0),
                    evicted=_fmt(row["selection_evicted_tokens_median"], 0),
                )
            )
        lines.extend(
            [
                "",
                "惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。",
            ]
        )
    lines.extend(
        [
            "",
            "## 最小质量护栏",
            "",
            "| 量化 | 通过题数 | 总题数 | 规则准确率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in quality:
        lines.append(
            f"| {row['quantization']} | {row['passed']} | {row['total']} | {float(row['accuracy']):.0%} |"
        )
    lines.extend(
        [
            "",
            "## 自动计算观察",
            "",
            *observations,
            f"- {len(quality)} 种精度在 {quality_total} 题 smoke set 上的规则准确率为 {min(quality_scores):.0%}–{max(quality_scores):.0%}；样本过少，不能据此比较量化精度。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。",
            "- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。",
            f"- {quality_total} 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。",
            "- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。",
            "- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。",
            "- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
