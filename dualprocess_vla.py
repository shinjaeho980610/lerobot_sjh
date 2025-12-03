# dpvla.py
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from lerobot.policies.factory import make_policy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.train import TrainPipelineConfig
import threading
import queue
import time
from draccus import load
import matplotlib.pyplot as plt

class DualProcess_VLA:
    def __init__(self, system1_cfg_path, system2_model_path, system1_hz=25, system2_hz=1, mode="real"):
        self.system1_hz = system1_hz
        self.system2_hz = system2_hz
        self.step_interval = int(system1_hz / system2_hz)
        self.current_step = 0
        self.mode = mode

        self.system2_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            system2_model_path, 
            #torch_dtype="auto",
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(system2_model_path, use_fast = True)
        processor.tokenizer.padding_side = "left"
        self.processor = processor

        # clip initialization
        # model_name = "openai/clip-vit-base-patch32"
        # self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # self.clip_text_w_proj_model = CLIPTextModelWithProjection.from_pretrained(model_name)

        #cfg = load(TrainPipelineConfig, system1_cfg_path)
        #dataset_metadata = LeRobotDatasetMetadata(repo_id='lerobot/robocasa_small',
        #                                          root='/mnt/data_disk4/dbs/robocasa/datasets/lerobot_single_human_demo')
        #self.system1_model = make_policy(cfg=cfg.policy, ds_meta=dataset_metadata)
        self.system1_model = ACTPolicy.from_pretrained(
            "/mnt/data_disk2/work/lerobot/outputs/train/act_accel_10dec/checkpoints/last/pretrained_model/", strict=False)
        self.num_decoder_layer = 10
        # self.system1_model = ACTPolicy.from_pretrained(
        #     "/mnt/data_disk2/work/lerobot/outputs/train/act_vae2_prefilm_accel/checkpoints/1290000/pretrained_model/",
        #     strict=False)

        self.system1_model.eval()

        self.current_latent = None
        if self.mode == "thread":
            self.latent_queue = queue.Queue(maxsize=1)
            self.thread_running = False

    def system2_thread(self, obs_dict):
        while self.thread_running:
            latent = self.extract_latent(obs_dict)
            if not self.latent_queue.empty():
                _ = self.latent_queue.get()
            self.latent_queue.put(latent)

    def extract_latent(self, obs_dict):

        #print("LATENT___", goal_dict["lang"].lower())
        images = [
            #to_pil_image(obs_dict['robot0_eye_in_hand_image'].squeeze(0)).convert("RGB"),
            #to_pil_image(obs_dict['robot0_agentview_right_image'].squeeze(0)).convert("RGB"),
            #to_pil_image(obs_dict['robot0_agentview_left_image'].squeeze(0)).convert("RGB"),
            pil_img_m = Image.fromarray(np.array(obs_dict['observation.mid_image'] * 255.).astype('uint8')),
            pil_img_r = Image.fromarray(np.array(obs_dict['observation.right_image'] * 255.).astype('uint8')),
            pil_img_l = Image.fromarray(np.array(obs_dict['observation.left_image'] * 255.).astype('uint8')),
        ]
        #
        # fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # (행, 열)
        # for ax, img in zip(axes, images):
        #     ax.imshow(img)
        #     ax.axis('off')  # 테두리·눈금 제거
        # plt.tight_layout()
        # plt.show()

        query = (
            '<|im_start|>system\n'
            'You are a single-arm gripper-type robot. You have just received the following images from three cameras. '
            'Picture 1: <|vision_start|><|image_pad|><|vision_end|> - Captured by the wrist-mounted camera. '
            'Picture 2: <|vision_start|><|image_pad|><|vision_end|> - Captured by the rear-right camera, showing both the robot and its environment. '
            'Picture 3: <|vision_start|><|image_pad|><|vision_end|> - Captured by the front-left camera, showing both the robot and its environment.<|im_end|>\n'
            '<|im_start|>user\n'
            f'How should you move when you need to {obs_dict["task"]}?'
            '<|im_start|>assistant\n'
        )
        #print(goal_dict["lang"].lower())
        inputs = self.processor(text=query, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.system2_model.device)

        output = self.system2_model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True
        )

        # remove 0.1 scaling on 250830
        #latent = output.hidden_states[0][-1][:, -1, :].float() * 0.1
        latent = output.hidden_states[0][self.num_decoder_layer][:, -1, :].float()

        return latent

    # def extract_clip_feature(self, goal_dict):
    #
    #     inputs = self.clip_tokenizer(goal_dict["lang"].lower(), padding=True, return_tensors="pt")
    #     inputs = inputs.to(self.clip_text_w_proj_model.device)
    #     with torch.no_grad():
    #         inst_text_features = self.clip_text_w_proj_model(**inputs).text_embeds.detach()act_vae8
    #
    #     return inst_text_features
    #     # print(clip_feature_embed.shape)

    def start_system2_thread(self, obs_dict):
        if self.mode == "thread" and not self.thread_running:
            self.thread_running = True
            thread = threading.Thread(target=self.system2_thread, args=(obs_dict))
            thread.daemon = True
            thread.start()

    def stop_system2_thread(self):
        self.thread_running = False

    def reset(self):
        self.system1_model.reset()
        self.current_step=0

    def forward_system1(self, obs_dict):
        #print(self.current_step, self.step_interval)
        if self.mode == "simulation":
            if self.current_step % self.step_interval == 0:
                self.current_latent = self.extract_latent(sys2_obs)
                #print("+++LATENT FEATURE extracted: ", self.current_latent)
            self.current_step += 1
        else:
            self.current_latent = self.latent_queue.get() if not self.latent_queue.empty() else self.current_latent

        #print(obs_dict.keys())
        # state = torch.cat([
        #     obs_dict["robot0_joint_pos_cos"],
        #     obs_dict["robot0_joint_vel"],
        #     obs_dict["robot0_eef_pos"],
        #     obs_dict["robot0_eef_quat"],
        # ], dim=1)


        #state = torch.cat([
        #    obs_dict["robot0_base_pos"], #
        #    obs_dict["robot0_base_quat"], #
        #    obs_dict["robot0_base_to_eef_pos"], #
        #    obs_dict["robot0_base_to_eef_quat"], #
        #    obs_dict["robot0_eef_pos"], #
        #    obs_dict["robot0_eef_quat"], #
        #    obs_dict["robot0_gripper_qpos"], #
        #    obs_dict["robot0_gripper_qvel"], #
        #    obs_dict["robot0_joint_pos_cos"],#
        #    obs_dict["robot0_joint_pos_sin"], #
        #    obs_dict["robot0_joint_vel"] #
        #], dim=1)
        
        state = torch.from_numpy(obs_dict['observation.state']).type(torch.float32).unsqueeze(0).to(self.system1_model.device)
        img_l = torch.from_numpy(obs_dict['observation.left_image']).permute(2,0,1).unsqueeze(0).type(torch.float32).to(self.system2_model.device)
        img_r = torch.from_numpy(obs_dict['observation.right_image']).permute(2,0,1).unsqueeze(0).type(torch.float32).to(self.system2_model.device)
        img_m = torch.from_numpy(obs_dict['observation.mid_image']).permute(2,0,1).unsqueeze(0).type(torch.float32).to(self.system2_model.device)
        #print(type(img_e), img_e.shape)

        #── ★ 디버그용 이미지 시각화 블록 ──────────────────────
        # if self.current_step % self.step_interval == 0:
        #     imgs   = [img_e, img_r, img_l]
        #     labels = ['eye', 'right', 'left']
        #
        #     # ① 파일로 저장
        #     for lab, t in zip(labels, imgs):
        #         to_pil_image(t.squeeze(0).cpu()).save(f"/tmp/{lab}_{self.current_step}.png")
        #
        #     # ② 즉시 화면에 띄우기 (Jupyter/GUI 세션용)
        #     fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        #     for ax, t, lab in zip(axes, imgs, labels):
        #         ax.imshow(to_pil_image(t.squeeze(0).cpu()))
        #         ax.set_title(lab)
        #         ax.axis('off')
        #     plt.tight_layout()
        #     plt.show()

        #print(img_e)
        observation = {
            "observation.mid_image": img_m,
            "observation.right_image": img_r,
            "observation.left_image": img_l,
            "observation.state": state,
            "latent": self.current_latent,
        }

        if self.current_latent == None:
            print('latent is not prepared!!!')
            break

        with torch.inference_mode():
            action = self.system1_model.select_action(observation)
            #print("++++++Action extracted: ", action)

        return action
