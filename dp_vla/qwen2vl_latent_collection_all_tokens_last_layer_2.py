from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import numpy as np
import PIL
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def latent_collect(model, processor, dataloader, output_dir, num_decoder_layer):

    CurrentEpisode = None
    Latent = []
    with torch.no_grad():
        for batch in dataloader:
            images_e = batch['observation.mid_image']
            images_r = batch['observation.right_image']
            images_l = batch['observation.left_image']
            tasks = batch['task']
            print(batch['episode_index'])
            print(batch['frame_index'])

            messages = []
            ImageSet = []

            # lerobot batch to qwen2.5vl batch
            for Idx in range(len(tasks)):
                pil_img_e = PIL.Image.fromarray(np.transpose((images_e[Idx].numpy() * 255).astype('uint8'), [1, 2, 0]))
                pil_img_r = PIL.Image.fromarray(np.transpose((images_r[Idx].numpy() * 255).astype('uint8'), [1, 2, 0]))
                pil_img_l = PIL.Image.fromarray(np.transpose((images_l[Idx].numpy() * 255).astype('uint8'), [1, 2, 0]))
                ImageSet.append(pil_img_e)
                ImageSet.append(pil_img_r)
                ImageSet.append(pil_img_l)

            texts = [(
                '<|im_start|>system\n'
                'You are a single-arm gripper-type robot. You have just received the following images from three cameras. '
                'Picture 1: <|vision_start|><|image_pad|><|vision_end|> - Captured by the wrist-mounted camera. '
                'Picture 2: <|vision_start|><|image_pad|><|vision_end|> - Captured by the rear-right camera, showing both the robot and its environment. '
                'Picture 3: <|vision_start|><|image_pad|><|vision_end|> - Captured by the front-left camera, showing both the robot and its environment.<|im_end|>\n'
                '<|im_start|>user\n'
                f'How should you move when you need to {task}?'
                '<|im_start|>assistant\n'
            )
                for task in tasks
            ]

            inputs = processor(
                text=texts,
                images=ImageSet,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # bug --. try to use generate function
            #output1 = model(**inputs, output_hidden_states=True, return_dict=True)
            output1 = model.generate(**inputs,
                                     max_new_tokens=1,
                                     do_sample=False,
                                     output_hidden_states=True,
                                     return_dict_in_generate=True)

            print('tuple size=', len(output1['hidden_states']))
            # len(output1['hidden_states']) = 346 --> # of output tokens
            print('len(output1[\'hidden_states\'][0])', len(output1['hidden_states'][0]))
            # len(output1['hidden_states'][0]) = 29 --> # of layers (depth)
            print('len(output1[\'hidden_states\'][0][num_decoder_layer].shape)', output1['hidden_states'][0][num_decoder_layer].shape)
            # output1['hidden_states'][-1][-1].shape = [64,   1, 3584] --> # (batch_size, seq_len, hidden_size)
            # output1['hidden_states'][ 0][-1].shape = [64, 184, 3584]
            last_hidden_state_of_1st_token = output1['hidden_states'][0][num_decoder_layer]


            #pooled_max = last_hidden_state.max(dim=1).values.cpu().float().numpy().astype('float16')
            #latent_extracted = pooled_max

            #prompt_len = inputs["input_ids"].shape[1]  # 모두 같은 길이(패딩 left)면 OK
            #print('prompt_length', prompt_len)
            #print(len(last_hidden_state))
            first_token_latent = last_hidden_state_of_1st_token[:, -1, :]  # (batch, hidden)
            latent_extracted = first_token_latent.cpu().float().numpy().astype('float16')

            print(latent_extracted.shape)

            if CurrentEpisode == int(batch['episode_index'][0]) and CurrentEpisode == int(batch['episode_index'][-1]):
                #Latent.append(output1['hidden_states'][-1][:, :, :].cpu().float().numpy().astype('float16'))
                Latent.append(latent_extracted)
            else:
                for Idx in range(len(tasks)):
                    if CurrentEpisode != int(batch['episode_index'][Idx]):
                        if CurrentEpisode is not None:
                            output_sub_dir = os.path.join(output_dir, 'chunk-' + str(CurrentEpisode // 1000).zfill(3))
                            if not os.path.exists(output_sub_dir):
                                os.makedirs(output_sub_dir)
                            np.save(os.path.join(output_sub_dir, 'episode_' + str(CurrentEpisode).zfill(6)),
                                     np.concatenate(Latent, 0))
                            print('Episode_Index: ', batch['episode_index'][Idx], ' Saved!')
                        Latent = []
                        CurrentEpisode = int(batch['episode_index'][Idx])
                    #Latent.append(
                    #    output1['hidden_states'][-1][Idx:Idx + 1, :, :].cpu().float().numpy().astype('float16'))
                    Latent.append(latent_extracted[Idx:Idx+1, :])

    # <<<< 추가 부분: 모든 batch 처리가 끝난 뒤 남아있는 마지막 에피소드 Latent 저장
    if len(Latent) > 0 and CurrentEpisode is not None:  # <<<< NEW
        output_sub_dir = os.path.join(output_dir, 'chunk-' + str(CurrentEpisode // 1000).zfill(3))  # <<<< NEW
        if not os.path.exists(output_sub_dir):  # <<<< NEW
            os.makedirs(output_sub_dir)         # <<<< NEW
        np.save(                               # <<<< NEW
            os.path.join(output_sub_dir, 'episode_' + str(CurrentEpisode).zfill(6)),
            np.concatenate(Latent, 0)
        )
        print('Episode_Index: ', CurrentEpisode, ' Saved (final)!')  # <<<< NEW

def main():

    parser = argparse.ArgumentParser(description="latent collection")

    parser.add_argument("--db_root", default='/mnt/data_disk4/dbs/robocasa/datasets/lerobot_small_single',type=str, help="Path to the input file")
    parser.add_argument("--output_dir", default='latent', type=str, help="Path to the input file")
    parser.add_argument("--start_ep_idx", type=int, default=0, help="Number of epochs to run")
    parser.add_argument("--end_ep_idx", type=int, default=1144, help="Number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of epochs to run")
    parser.add_argument("--num_decoder_layer", type=int, default=10, help="Number of epochs to run")

    # 파싱
    args = parser.parse_args()

    # 1. 모델 & 프로세서 불러오기
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    processor.tokenizer.padding_side = "left"

    device = torch.device("cuda")

    target_episodes= list(range(args.start_ep_idx, args.end_ep_idx+1))
    print(f'target_episodes = {target_episodes}')
    dataset = LeRobotDataset("lerobot/custom_dataset",
                             root=args.db_root,
                             episodes = target_episodes
                             )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    output_dir = os.path.join(args.db_root, args.output_dir)
    latent_collect(model, processor, dataloader, output_dir, args.num_decoder_layer)

if __name__ == "__main__":
    main()






