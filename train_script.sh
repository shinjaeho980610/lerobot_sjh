lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=lerobot/custom_dataset \
    --dataset.root=/mnt/d/panda_robot_dataset_lerobot/v1 \
    --steps=1000000 \
    --save_freq=50000 \
    --output_dir=outputs/train/smolvla_panda_robot_v1 \
    --batch_size=64 \
    --policy.repo_id=false
