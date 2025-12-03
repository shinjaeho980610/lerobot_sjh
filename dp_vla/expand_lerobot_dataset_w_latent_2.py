#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    """
    → 원래 코드대로 latent에 names 포함
    """
    feats = {k: dict(v) for k, v in src_ds.features.items()}
    if col in feats:
        logging.info(f"'{col}' already exists in source dataset.")
    else:
        feats[col] = {
            "dtype": "float32",
            "shape": (latent_dim,),
            "names": [f"{col}_{i}" for i in range(latent_dim)],  # ← names 유지!
        }
    return feats


def write_global_stats(dst_ds: LeRobotDataset) -> None:
    stats = aggregate_stats(list(dst_ds.meta.episodes_stats.values()))
    write_json(stats, dst_ds.root / "meta" / "stats.json")
    dst_ds.meta.stats = stats
    logging.info("stats.json updated.")


def get_npy_path(latent_root: Path, meta, ep_idx: int) -> Path:
    rel = Path(meta.get_data_file_path(ep_idx))          # data/chunk-000/episode_000000.parquet
    chunk_dir = rel.parts[-2]                            # chunk-000
    filename = rel.stem + ".npy"                         # episode_000000.npy
    return latent_root / chunk_dir / filename


def _to_numpy(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v


def _is_image_key(k: str) -> bool:
    return "image" in k  # 필요하면 'rgb', 'cam' 등 추가


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    (C,H,W) 또는 (H,W,C) float(0~1 or 0~255) -> (H,W,C) uint8
    """
    assert img.ndim == 3
    if img.shape[0] in (1, 3):                 # CHW
        img = np.transpose(img, (1, 2, 0))     # -> HWC
    if np.issubdtype(img.dtype, np.floating):  # 0~1 → 0~255
        vmin = float(img.min()) if img.size else 0.0
        vmax = float(img.max()) if img.size else 1.0
        if vmin >= 0.0 and vmax <= 1.0:
            img = img * 255.0
        img = np.rint(img).clip(0, 255).astype(np.uint8, copy=False)
    elif img.dtype != np.uint8:
        img = np.rint(img).clip(0, 255).astype(np.uint8, copy=False)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img


def _cast_scalar_to_shape_dtype(v, shape, dtype_str):
    """
    스칼라/숫자 배열을 spec의 shape/dtype에 맞춤.
    - shape == (1,) 이면 (1,)로 래핑
    - dtype은 float32/int32 등 숫자형만 캐스팅
    """
    v = _to_numpy(v)
    exp_shape = tuple(shape) if isinstance(shape, (list, tuple)) else tuple()
    if exp_shape == (1,):
        if not (isinstance(v, np.ndarray) and v.shape == (1,)):
            v = np.array([v])
    if dtype_str in {"float32", "float64", "int32", "int64"}:
        v = v.astype(dtype_str, copy=False)
    return v


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
        n_frames = end - start

        if latents.shape[0] != n_frames:
            logging.warning(
                f"[skip] Episode {ep:06d}: latent rows({latents.shape[0]}) != frames({n_frames})"
            )
            continue

        for local_i, src_idx in enumerate(range(start, end)):
            s = src_ds[src_idx]

            # (A) frame dict: 저장 피처만 넣는다 (task/timestamp는 절대 넣지 않음)
            frame = {}

            # (B) latent: numpy float32 1D (names는 features에만 존재)
            frame[latent_col] = latents[local_i]  # shape: (latent_dim,), dtype float32

            # (C) 나머지 피처를 저장 스키마에 맞춰 변환
            for k, spec in src_ds.features.items():
                if k in {"timestamp", "index", "frame_index", "episode_index", "task_index"}:
                    continue
                if k == "task" or k == latent_col:
                    continue

                v = s[k]
                if _is_image_key(k) and getattr(v, "ndim", None) in (3,):
                    v = _to_numpy(v)
                    v = _to_hwc_uint8(v)
                    frame[k] = v
                    continue

                # 수치형: 스칼라/벡터를 shape/dtype에 맞춤
                exp_shape = spec.get("shape", ())
                exp_dtype = spec.get("dtype", None)
                frame[k] = _cast_scalar_to_shape_dtype(v, exp_shape, exp_dtype if isinstance(exp_dtype, str) else str(v.dtype))

            # (D) add_frame: frame dict에 task/timestamp를 넣지 말고 인자로만 전달
            dst_ds.add_frame(frame, task=s["task"], timestamp=float(_to_numpy(s["timestamp"])))

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

    logging.info("Re-recording with latent features…")
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
