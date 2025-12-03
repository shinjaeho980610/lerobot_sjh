import jason
import h5py
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, required = True, help = 'input_dir(e.g. /mnt/data_disk/sjh/dataset)')
parser.add_argument('--output_dir', type = str, required = True, help = 'output_dir(e.g. /mnt/data_disk/pick_up_marker)')
option_args = parser.parse_args()

def convert_custom_to_lerobot(input_dir, output_dir, fps = 30):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    features = {
        'timestamp':{'dtype': 'float32', 'shape': (1,)},
        'action':{'dtype':'float32', 'shape':(49,)},
        'observation.state':{'dtype':'float32','shape':(50,)},
        'observation.head_camera_rgb':{'dtype':'video','shape':(480,640,3),
                                 'names':('height','width','channel')},
        'reward':{'dtype':'float32','shape':(1,)},
        'done':{'dtype':'int32','shape':(1,)},
    }

    dataset = LeRobotDataset.create(
        repo_id = 'lerobot/bi_custom_dataset',
        fps = fps,
        root = output_dir,
        #robot = None,
        #robot_type = 'panda',
        features = features,
        use_videos = True,
        image_writer_threads = 5,
        image_writer_processes = 10,
    )

    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        instr = str(task_dir).split('/')[-1]

        for i, h5_path in enumerate(sorted(task_dir.glob('*.hdf5'))):
            print(f'# Processing: [{i}] {h5_path}')
            with h5py.File(h5_path, 'r') as f:
                states = f['observations']['qpos'][:]
                actions = f['action'][:]

                if states.shape[0] != actions.shape[0]:
                    print('shape 불일치 : fail', states.shape[0], actions.shape[0])
                    continue
                print('shape 일치 : pass')

                images = {
                    'observations.head_camera_rgb':np.array(f['observations']['images']['head_camera_rgb'])
                }
                n_frames = states.shape[0]

                rewards = np.zeros((n_frames,), dtype = np.float32)
                dones = np.zeros((n_frames,), dtype = np.int32)
                dones[-1] = 1

                for n in range(n_frames):
                    frame = {
                        #"timestamp" : np.array([n/fps], dtype = np.float32),
                        "action" : np.array(actions[n], dtype = np.float32),
                        "observation.state" : np.array(states[n], dtype = np.float32),
                        "observation.head_camera_rgb" : images[ 'observations.head_camera_rgb'][n],
                        "reward" : np.array([int(rewards[n])], dtype = np.float32),
                        "done" : np.array([int(dones[n])], dtype = np.int32)
                    }
                    dataset.add_frame(frame, task = instr, timestamp=float(n)/fps)
                dataset.save_episode()

def main(option_args = option_args):
    convert_custom_to_lerobot(
        input_dir = option_args.input_dir,
        output_dir = option_args.output_dir,
        fps = 30
    )

if __name__ == '__main__':
    main()