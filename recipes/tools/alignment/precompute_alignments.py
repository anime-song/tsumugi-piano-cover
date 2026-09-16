from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from tqdm.auto import tqdm

from recipes.data.alignment import (
    compute_pair_audio_alignment,
    load_alignment_cache,
    resolve_alignment_cache_path,
    save_alignment_cache,
)
from recipes.data.index import PairEntry, build_pair_index, split_pairs_by_song
from tsumugi_piano_cover.config import AlignmentConfig, ExperimentConfig, load_experiment_config


def parse_args() -> argparse.Namespace:
    # alignment 前計算用の CLI 引数を受け取る
    parser = argparse.ArgumentParser(description="Precompute source-target alignments.")
    parser.add_argument("--config", type=str, default="configs/piano_cover_v2.yaml")
    parser.add_argument("--split", type=str, choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def resolve_alignment_cache_dir(config: ExperimentConfig) -> Path:
    # 新しい alignment 設定を優先し、未指定なら既存 dataset 設定と既定値へフォールバックする
    if config.alignment.cache_dir:
        return Path(config.alignment.cache_dir)
    if config.dataset.alignment_cache_dir:
        return Path(config.dataset.alignment_cache_dir)
    return Path("artifacts/piano_cover_v2_alignments")


def build_pairs(config: ExperimentConfig, split: str) -> list[PairEntry]:
    # dataset 定義から対象 split のペア一覧を構築する
    entries = build_pair_index(
        dataset_json_path=config.dataset.dataset_json,
        piano_midi_dir=config.dataset.piano_midi_dir,
        piano_to_performer_json=config.dataset.piano_to_performer_json,
        original_audio_dir=config.dataset.original_audio_dir,
        piano_audio_dir=config.dataset.piano_audio_dir,
    )
    if split == "all":
        return entries

    split_entries = split_pairs_by_song(
        entries=entries,
        train_fraction=config.dataset.train_fraction,
        val_fraction=config.dataset.val_fraction,
        test_fraction=config.dataset.test_fraction,
        seed=config.seed,
    )
    return split_entries[split]


def _process_single_pair(
    entry: PairEntry,
    alignment_config: AlignmentConfig,
    cache_dir: str,
    force: bool,
) -> tuple[str, float | None, float | None, float | None]:
    # 1. 既存 cache があれば再計算をスキップする
    cache_path = resolve_alignment_cache_path(cache_dir, entry)
    if cache_path.exists() and not force and _cache_matches_requested_method(cache_path, alignment_config):
        return ("skipped", None, None, None)

    # 2. 設定された方式で alignment を計算して保存する
    result = compute_pair_audio_alignment(entry, alignment_config)
    save_alignment_cache(cache_dir, entry, alignment_config, result)

    # 3. 親プロセスで summary を集計しやすい形だけ返す
    average_match_cost = result.average_match_cost if math.isfinite(result.average_match_cost) else None
    return ("saved", average_match_cost, result.gap_ratio, result.max_absolute_offset_seconds)


def _cache_matches_requested_method(cache_path: Path, alignment_config: AlignmentConfig) -> bool:
    # 異なる方式で作った cache は使い回さず、現在の設定で作り直す
    payload = load_alignment_cache(cache_path)
    if payload.get("version") != 3:
        return False
    payload_config = payload.get("alignment_config")
    payload_method = payload_config.get("method") if isinstance(payload_config, dict) else None
    payload_kind = payload.get("alignment_kind")
    return payload_method == alignment_config.method or payload_kind == alignment_config.method


def _process_pair_group(
    entries: list[PairEntry],
    alignment_config: AlignmentConfig,
    cache_dir: str,
    force: bool,
) -> list[tuple[str, float | None, float | None, float | None]]:
    # 同じ source を共有する pair 群を同一 worker でまとめて処理し、source feature cache を効かせる
    results: list[tuple[str, float | None, float | None, float | None]] = []
    for entry in entries:
        try:
            results.append(_process_single_pair(entry, alignment_config, cache_dir, force))
        except Exception as exc:
            raise RuntimeError(
                f"failed to precompute alignment for original_id={entry.original_id} piano_id={entry.piano_id}"
            ) from exc
    return results


def _accumulate_summary(
    status: str,
    average_match_cost: float | None,
    gap_ratio: float | None,
    max_offset_seconds: float | None,
    summary_costs: list[float],
    summary_gap_ratios: list[float],
    summary_offsets: list[float],
) -> tuple[int, int]:
    # saved / skipped 数と summary 集計を 1 箇所に寄せる
    if status == "skipped":
        return 0, 1

    if average_match_cost is not None:
        summary_costs.append(average_match_cost)
    if gap_ratio is not None:
        summary_gap_ratios.append(gap_ratio)
    if max_offset_seconds is not None:
        summary_offsets.append(max_offset_seconds)
    return 1, 0


def _resolve_num_workers(args: argparse.Namespace, config: ExperimentConfig) -> int:
    # CLI 未指定時は実験設定の runtime.num_workers を使い、最低 1 worker は確保する
    if args.num_workers is not None:
        return max(1, int(args.num_workers))
    return max(1, int(config.runtime.num_workers))


def _group_pairs_for_processing(pairs: list[PairEntry]) -> list[list[PairEntry]]:
    # 同じ original をまとめ、worker 内の source feature cache を活かす
    groups_by_original: dict[str, list[PairEntry]] = {}
    for entry in pairs:
        groups_by_original.setdefault(entry.original_id, []).append(entry)
    return list(groups_by_original.values())


def _run_single_process(
    pairs: list[PairEntry],
    alignment_config: AlignmentConfig,
    cache_dir: str,
    force: bool,
) -> tuple[int, int, list[float], list[float], list[float]]:
    # 逐次処理はデバッグしやすく、少数件の smoke test に向いている
    num_saved = 0
    num_skipped = 0
    summary_costs: list[float] = []
    summary_gap_ratios: list[float] = []
    summary_offsets: list[float] = []

    iterator = (_process_single_pair(entry, alignment_config, cache_dir, force) for entry in pairs)
    progress = tqdm(iterator, total=len(pairs), desc="Precompute alignments", dynamic_ncols=True)
    for status, average_match_cost, gap_ratio, max_offset_seconds in progress:
        saved_inc, skipped_inc = _accumulate_summary(
            status=status,
            average_match_cost=average_match_cost,
            gap_ratio=gap_ratio,
            max_offset_seconds=max_offset_seconds,
            summary_costs=summary_costs,
            summary_gap_ratios=summary_gap_ratios,
            summary_offsets=summary_offsets,
        )
        num_saved += saved_inc
        num_skipped += skipped_inc

    return num_saved, num_skipped, summary_costs, summary_gap_ratios, summary_offsets


def _run_multi_process(
    pair_groups: list[list[PairEntry]],
    alignment_config: AlignmentConfig,
    cache_dir: str,
    force: bool,
    num_workers: int,
    total_pairs: int,
) -> tuple[int, int, list[float], list[float], list[float]]:
    # original 単位の task にして、同じ source を複数 worker で踏み直しにくくする
    num_saved = 0
    num_skipped = 0
    summary_costs: list[float] = []
    summary_gap_ratios: list[float] = []
    summary_offsets: list[float] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_group = {
            executor.submit(_process_pair_group, group, alignment_config, cache_dir, force): group
            for group in pair_groups
        }
        progress = tqdm(total=total_pairs, desc="Precompute alignments", dynamic_ncols=True)
        for future in as_completed(future_to_group):
            group = future_to_group[future]
            try:
                group_results = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"failed to precompute alignment group for original_id={group[0].original_id}"
                ) from exc

            for status, average_match_cost, gap_ratio, max_offset_seconds in group_results:
                saved_inc, skipped_inc = _accumulate_summary(
                    status=status,
                    average_match_cost=average_match_cost,
                    gap_ratio=gap_ratio,
                    max_offset_seconds=max_offset_seconds,
                    summary_costs=summary_costs,
                    summary_gap_ratios=summary_gap_ratios,
                    summary_offsets=summary_offsets,
                )
                num_saved += saved_inc
                num_skipped += skipped_inc
                progress.update(1)
        progress.close()

    return num_saved, num_skipped, summary_costs, summary_gap_ratios, summary_offsets


def main() -> None:
    # 1. 設定と対象ペアを読み込む
    args = parse_args()
    config = load_experiment_config(args.config)
    pairs = build_pairs(config, args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    num_workers = _resolve_num_workers(args, config)
    pair_groups = _group_pairs_for_processing(pairs)

    cache_dir = resolve_alignment_cache_dir(config)
    print(f"split={args.split}")
    print(f"pairs={len(pairs)}")
    print(f"pair_groups={len(pair_groups)}")
    print(f"cache_dir={cache_dir}")
    print(f"alignment_frame_seconds={config.alignment.frame_seconds}")
    print(f"alignment_method={config.alignment.method}")
    print(f"num_workers={num_workers}")
    if args.dry_run:
        return

    # 2. pair ごとに alignment を前計算して保存する
    if num_workers <= 1:
        num_saved, num_skipped, summary_costs, summary_gap_ratios, summary_offsets = _run_single_process(
            pairs=pairs,
            alignment_config=config.alignment,
            cache_dir=str(cache_dir),
            force=args.force,
        )
    else:
        num_saved, num_skipped, summary_costs, summary_gap_ratios, summary_offsets = _run_multi_process(
            pair_groups=pair_groups,
            alignment_config=config.alignment,
            cache_dir=str(cache_dir),
            force=args.force,
            num_workers=num_workers,
            total_pairs=len(pairs),
        )

    # 3. ざっくりした統計を出して、前計算の異常を見つけやすくする
    print(f"saved={num_saved}")
    print(f"skipped={num_skipped}")
    if summary_costs:
        print(f"average_match_cost={mean(summary_costs):.4f}")
    if summary_gap_ratios:
        print(f"average_gap_ratio={mean(summary_gap_ratios):.4f}")
    if summary_offsets:
        print(f"average_max_offset_seconds={mean(summary_offsets):.2f}")


if __name__ == "__main__":
    main()
