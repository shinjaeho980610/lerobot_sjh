#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rewrite_with_latent.py   ★ UPDATED ★

소스 데이터셋.
├── data
│   └── chunk-000/episode_000000.parquet
└── latent_first_output        ← latent 벡터가 들어있는 추가 디렉터리
    └── chunk-000/episode_000000.npy

● 모든 프레임을 add_frame()/save_episode() 로 재녹화하며
  latent_feature 열을 삽입
● meta/{episodes_stats/*.json , stats.json} 자동 생성

--------------------------------------------------------------------
python rewrite_with_latent.py \
       --src-repo robocasa_single \
       --src-root ~/datasets/robocasa_single \
       --dst-repo robocasa_single_latent \
       --dst-root ~/datasets/robocasa_single_latent \
       --latent-dir latent_first_output   # (src-root 하위·기본값) \
       --latent-col latent_feature
--------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import write_json

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_new_features(src_ds: LeRobotDataset, latent_dim: int, col: str) -> dict:
    feats = {k: dict(v) for k, v in src_ds.features.items()}
    if col in feats:
        logging.info(f"'{col}' already exists in source dataset.")
    else:
        feats[col] = {
            "dtype": "float32",
            "shape": (latent_dim,),
            "names": [f"{col}_{i}" for i in range(latent_dim)],
        }
    return feats


def write_global_stats(dst_ds: LeRobotDataset) -> None:
    stats = aggregate_stats(list(dst_ds.meta.episodes_stats.values()))
    write_json(stats, dst_ds.root / "meta" / "stats.json")
    dst_ds.meta.stats = stats
    logging.info("stats.json updated.")


def get_npy_path(latent_root: Path, meta, ep_idx: int) -> Path:
    """
    latent_root/chunk-XXX/episode_YYYYYY.npy
    where chunk-XXX and stem are taken from meta.get_data_file_path().
    """
    rel = Path(meta.get_data_file_path(ep_idx))          # data/chunk-000/episode_000000.parquet
    chunk_dir = rel.parts[-2]                            # chunk-000
    filename = rel.stem + ".npy"                         # episode_000000.npy
    return latent_root / chunk_dir / filename


# ─────────────────────────────────────────────────────────────────────────────
# Replay routine
# ─────────────────────────────────────────────────────────────────────────────
def replay_dataset(
    src_ds: LeRobotDataset,
    dst_ds: LeRobotDataset,
    latent_root: Path,
    latent_col: str,
) -> None:
    epi_map = src_ds.episode_data_index
    for ep in tqdm(range(src_ds.meta.total_episodes), desc="Episodes"):
        npy_path = get_npy_path(latent_root, src_ds.meta, ep)
        if not npy_path.exists():
            logging.warning(f"[skip] latent npy missing: {npy_path}")
            continue
        latents = np.load(npy_path).astype(np.float32)
        if latents.ndim == 1:
            latents = latents[None, :]

        start = epi_map["from"][ep].item()
        end = epi_map["to"][ep].item()
        if latents.shape[0] != end - start:
            logging.warning(
                f"[skip] Episode {ep:06d}: latent rows({latents.shape[0]}) != frames({end-start})"
            )
            continue

        for local_i, src_idx in enumerate(range(start, end)):
            s = src_ds[src_idx]
            print(s)
            frame = {
                "timestamp": s["timestamp"],
                "task": s["task"],
                latent_col: torch.from_numpy(latents[local_i]),
            }
            for k in src_ds.features:
                if k in {
                    "timestamp",
                    "index",
                    "frame_index",
                    "episode_index",
                    "task_index",
                    "latent"
                }:
                    continue
                frame[k] = s[k]
            dst_ds.add_frame(frame, task = s["task"], timestamp = s["timestamp"])
        dst_ds.save_episode()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(
    src_repo: str,
    src_root: Path,
    dst_repo: str,
    dst_root: Path,
    latent_dir_name: str,
    latent_col: str,
    writer_threads: int,
) -> None:
    logging.info("Loading source dataset…")
    src_ds = LeRobotDataset(repo_id=src_repo, root=src_root, download_videos=True)

    latent_root = (src_root / latent_dir_name).resolve()
    if not latent_root.is_dir():
        raise FileNotFoundError(f"latent root not found: {latent_root}")

    first_npy = next(latent_root.rglob("episode_*.npy"), None)
    if first_npy is None:
        raise FileNotFoundError("No *.npy under latent root.")
    latent_dim = np.load(first_npy, mmap_mode="r").shape[-1]
    logging.info(f"Detected latent_dim = {latent_dim}")

    new_feats = build_new_features(src_ds, latent_dim, latent_col)

    logging.info("Creating destination dataset…")
    dst_ds = LeRobotDataset.create(
        repo_id=dst_repo,
        fps=src_ds.fps,
        root=dst_root,
        robot_type=src_ds.meta.robot_type,
        features=new_feats,
        use_videos=True,
        image_writer_threads=writer_threads,
    )

    logging.info("Re‑recording with latent features…")
    replay_dataset(src_ds, dst_ds, latent_root, latent_col)

    dst_ds.stop_image_writer()
    write_global_stats(dst_ds)
    logging.info("✅  Completed. New dataset at %s", dst_root)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-repo", default='lerobot/custom_dataset', help="source repo_id")
    parser.add_argument("--src-root", default='/mnt/d/lerobot_dataset/panda_robot_dataset_lerobot/v1', help="source dataset root")
    parser.add_argument("--dst-repo", default='lerobot/custom_dataset_latent', help="destination repo_id")
    parser.add_argument("--dst-root", default='/mnt/d/lerobot_dataset/panda_robot_dataset_lerobot/v1_latent', help="destination root")
    parser.add_argument(
        "--latent-dir",
        default="latent",
        help="directory (under src-root) containing chunk-***/episode_*.npy",
    )
    parser.add_argument("--latent-col", default="latent", help="new column name")
    parser.add_argument("--writer-threads", type=int, default=8)
    args = parser.parse_args()

    main(
        src_repo=args.src_repo,
        src_root=Path(args.src_root).expanduser().resolve(),
        dst_repo=args.dst_repo,
        dst_root=Path(args.dst_root).expanduser().resolve(),
        latent_dir_name=args.latent_dir,
        latent_col=args.latent_col,
        writer_threads=args.writer_threads,
    )