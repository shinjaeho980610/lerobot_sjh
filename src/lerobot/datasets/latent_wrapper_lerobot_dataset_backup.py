import numpy as np
import random
import torch
from pathlib import Path
from collections import OrderedDict

### DON'T USE

class LatentWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, latent_root, window_s=1.0, cache_size=10000):
        """
        Args:
            base_dataset: LeRobotDataset instance
            latent_root: Path to the 'latent' directory
            window_s: Time window in seconds to sample latent vectors
            cache_size: Number of latent files to keep in memory
        """
        self.base_dataset = base_dataset
        self.latent_root = base_dataset.root / latent_root
        self.window_s = window_s
        self.fps = base_dataset.fps
        self.cache_size = cache_size

        self.meta = self.base_dataset.meta
        self.num_frames = self.base_dataset.num_frames
        self.num_episodes = self.base_dataset.num_episodes
        self.episode_data_index = self.base_dataset.episode_data_index

        # LRU Cache (최신 순으로 유지)
        self.latent_cache = OrderedDict()

    def _load_latent_npz(self, latent_path):
        # 캐시에 있으면 사용
        #print('load_latent...')
        if latent_path in self.latent_cache:
            latent_array = self.latent_cache.pop(latent_path)
            self.latent_cache[latent_path] = latent_array  # move to end (most recently used)
            return latent_array

        # 없으면 파일에서 로드
        with np.load(latent_path) as latent_data:
            latent_key = list(latent_data.files)[0]
            latent_array = latent_data[latent_key]

        # 캐시에 추가
        self.latent_cache[latent_path] = latent_array
        #print('cache_size=', len(self.latent_cache))
        if len(self.latent_cache) > self.cache_size:
            self.latent_cache.popitem(last=False)  # remove least recently used

        #print('latent loading completed')

        return latent_array

    def _load_latent(self, latent_path):
        # 캐시에 있으면 사용
        if latent_path in self.latent_cache:
            latent_array = self.latent_cache.pop(latent_path)
            self.latent_cache[latent_path] = latent_array
            return latent_array

        # ↓↓↓ 여기부터 수정 ↓↓↓
        latent_array = np.load(latent_path)#, mmap_mode="r")  # ndarray 직접 반환
        # ↑↑↑ 여기까지 수정 ↑↑↑

        # 캐시에 추가
        self.latent_cache[latent_path] = latent_array
        if len(self.latent_cache) > self.cache_size:
            self.latent_cache.popitem(last=False)

        return latent_array

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        # latent npz path
        chunk_idx = ep_idx // self.base_dataset.meta.chunks_size
        latent_path = self.latent_root / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.npy"

        # load latent array (with caching)
        # print(latent_path)
        latent_array = self._load_latent(latent_path)

        # compute candidate indices
        window_frames = int(self.window_s * self.fps)

        #start_idx = max(0, frame_idx - window_frames)
        #end_idx = frame_idx
        # if end_idx >= start_idx:
        #     candidate_indices = list(range(start_idx, end_idx + 1))
        #     selected_idx = random.choice(candidate_indices)
        # else:
        #     selected_idx = frame_idx
        block = frame_idx // window_frames  # 몇 번째 윈도우인지
        selected_idx = min(block * window_frames, frame_idx)

        latent_vector = latent_array[selected_idx]

        # add to item dict
        item["latent"] = torch.from_numpy(latent_vector).float() * 0.1
        item["latent_frame_index"] = torch.tensor(selected_idx, dtype=torch.long)

        return item
