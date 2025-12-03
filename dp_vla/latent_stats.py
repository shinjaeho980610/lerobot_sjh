import json
import numpy as np
from pathlib import Path
from lerobot.datasets.compute_stats import compute_episode_stats

# === 경로 설정 ===
dataset_root = Path("/mnt/d/lerobot_dataset/panda_robot_dataset_lerobot/v1_latent")
feature_dir = f"{dataset_root}/latent"  # .npy 파일들이 있는 디렉토리
in_stats_path = f"{dataset_root}/meta/episodes_stats.jsonl"  # 기존 JSONL
out_stats_path = f"{dataset_root}/meta/episodes_stats_w_latent_last.jsonl"  # 출력 JSONL

feature_name = "latent"
latent_dim = 3584

# === 결과 저장용 리스트 ===
updated_lines = []
CHUNK_SIZE = 1000  # 에피소드 0~999 -> chunk-000, 1000~1999 -> chunk-001, ...

# === JSONL 한 줄씩 읽어서 처리 ===
with open(in_stats_path, "r") as f:
    for line in f:
        record = json.loads(line)
        ep_index = record["episode_index"]
        print(ep_index, '...')

        chunk_id = ep_index // CHUNK_SIZE
        npy_path = Path(f"{feature_dir}/chunk-{chunk_id:03d}/episode_{ep_index:06d}.npy")

        if not npy_path.exists():
            print(f"⚠️ {npy_path} not found. Skipping.")
            updated_lines.append(record)
            continue

        # Load latent feature
        data = np.load(npy_path).astype(np.float64)  # shape: [T, 3584]
        assert data.ndim == 2 and data.shape[1] == latent_dim, f"{npy_path.name} has invalid shape {data.shape}"

        # 통계 계산
        ep_dict = {feature_name: data}
        features = {
            feature_name: {
                "dtype": "float",
                "shape": [latent_dim],
            }
        }
        ep_stats = compute_episode_stats(ep_dict, features)

        # 에피소드의 stats에 latent 추가
        # ✅ ndarray → list로 변환해서 JSON 직렬화 가능하게 수정
        record["stats"][feature_name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in ep_stats[feature_name].items()
        }

        updated_lines.append(record)

# === JSONL 저장 ===
with open(out_stats_path, "w") as f:
    for record in updated_lines:
        json.dump(record, f)
        f.write("\n")

print(f"✅ Updated stats with '{feature_name}' written to: {out_stats_path}")
