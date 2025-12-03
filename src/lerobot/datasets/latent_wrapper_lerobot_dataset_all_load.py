import numpy as np
import random
import torch
from pathlib import Path
from collections import OrderedDict

class LatentWrapperDatasetAllLoad(torch.utils.data.Dataset):
    def __init__(self, base_dataset, latent_root, window_s=1.0,
                 cache_name: str = "latent_cache.pt"):
        """
        Args:
            base_dataset: LeRobotDataset instance
            latent_root : 폴더 이름(예: 'latent_maxpool')
            window_s    : 과거 latent 샘플링 창(초)
            cache_name  : 캐시 파일명 (같은 root 아래 저장)
        """
        self.base_dataset = base_dataset
        self.latent_root  = base_dataset.root / latent_root
        self.window_s     = window_s
        self.fps          = base_dataset.fps

        cache_file = self.latent_root / cache_name       # <-- 캐시 위치
        if cache_file.exists():
            print(f"[latent‑cache] load from {cache_file}")
            # ep_idx(int) → np.ndarray  dict 그대로 저장했으니 바로 torch.load
            self.latent_data = torch.load(cache_file, map_location="cpu")
            return

        # 캐시가 없으면 원본 npz 전부 로드
        self.latent_data = {}          # ep_idx -> np.ndarray
        print("[latent‑cache] building…")
        for i, ep_idx in enumerate(base_dataset.meta.episodes):
            print(f"  [{i}/{len(base_dataset)}] episode {ep_idx}")
            cidx = ep_idx // base_dataset.meta.chunks_size
            npz_path = self.latent_root / f"chunk-{cidx:03d}" / f"episode_{ep_idx:06d}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(npz_path)
            with np.load(npz_path) as f:
                self.latent_data[ep_idx] = f[f.files[0]]

        # # 한 번만 저장
        # cache_file.parent.mkdir(parents=True, exist_ok=True)
        # torch.save(self.latent_data, cache_file)
        # print(f"[latent‑cache] saved to {cache_file}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        # 미리 로드한 latent 데이터에서 가져오기
        latent_array = self.latent_data[ep_idx]

        # compute candidate indices
        window_frames = int(self.window_s * self.fps)
        start_idx = max(0, frame_idx - window_frames)
        end_idx = frame_idx

        if end_idx >= start_idx:
            candidate_indices = list(range(start_idx, end_idx + 1))
            selected_idx = random.choice(candidate_indices)
        else:
            selected_idx = frame_idx

        latent_vector = latent_array[selected_idx]

        # add to item dict
        item["latent"] = torch.from_numpy(latent_vector).float()
        item["latent_frame_index"] = torch.tensor(selected_idx, dtype=torch.long)

        return item