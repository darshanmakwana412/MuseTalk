import os
os.environ["PULSE_SERVER"] = "tcp:localhost:4713"

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from transformers.audio_utils import mel_filter_bank
from transformers import WhisperModel
from musetalk.utils.utils import datagen
from musetalk.utils.utils import load_all_model
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

import shutil
import threading
import queue
import time
import subprocess

SAMPLE_RATE   = 16000
BLOCK_MS      = 40
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
CHANNELS      = 1
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float32
AUDIO_PATH    = "data/audio/yongen.wav"

sr = 16000
n_fft = 400
hop_length = 160
feature_size = 80
num_samples = 30 * SAMPLE_RATE

# ~2.5s buffering for both audio and frame queues
AUDIO_Q_MAX = 64
FRAME_Q_MAX = 64

# This will give a fixed latency of 200ms
LOOKAHEAD = 4
WINDOW = LOOKAHEAD + 1

window = torch.hann_window(n_fft, device=DEVICE)
mel_filters = torch.from_numpy(
    mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=feature_size,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    )
).to(DEVICE, DTYPE)

vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=DEVICE
)
timesteps = torch.tensor([0], device=DEVICE)

pe = pe.half().to(DEVICE)
vae.vae = vae.vae.half().to(DEVICE)
unet.model = unet.model.half().to(DEVICE)

whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=DEVICE, dtype=DTYPE).eval()
whisper.requires_grad_ = False

audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_Q_MAX)
frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=FRAME_Q_MAX)
stop_flag = threading.Event()

fp = FaceParsing(
    left_cheek_width=90,
    right_cheek_width=90
)

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        # 根据版本设置不同的基础路径
        self.base_path = f"./results/v15/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": "v15"
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                # response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                response = "y" # For testing purposes, always re-create
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2 = y2 + 10
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = "jaw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

avatar_id = "dragon"
video_path = "data/video/yongen.mp4"
avatar = Avatar(
    avatar_id=avatar_id,
    video_path=video_path,
    bbox_shift=0,
    batch_size=1,
    preparation=True
)

def rec_loop():
    wav, sr = torchaudio.load(AUDIO_PATH)
    wav = wav.mean(dim=0)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    num_samples = wav.numel()

    while True:
        for i in range(0, num_samples, BLOCK_SAMPLES):
            block = wav[i : i + BLOCK_SAMPLES]
            if block.numel() < BLOCK_SAMPLES:
                block = torch.nn.functional.pad(
                    block, (0, BLOCK_SAMPLES - block.numel()))
            try:
                audio_q.put(block.unsqueeze(1).cpu().numpy(), timeout=1)
            except queue.Full:
                pass
            time.sleep(BLOCK_MS / 1000)

    # stop_flag.set()

@torch.no_grad()
def infer_frame():

    while not stop_flag.is_set():
        try:
            block = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        audio = block[:, 0]

        if audio.shape[0] >= num_samples:
            padded_audio = audio[:num_samples]
        else:
            padded_audio = np.pad(audio, (0, num_samples - audio.shape[0]))
        audio_tensor = torch.from_numpy(padded_audio).to(DEVICE, DTYPE)
        
        stft = torch.stft(
            audio_tensor,
            n_fft,
            hop_length,
            window=window,
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = mel_filters.T @ magnitudes
        
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        hidden_states = whisper.encoder(
            log_spec.unsqueeze(0),
            output_hidden_states=True
        ).hidden_states
        hidden_states = torch.stack(hidden_states, dim=2)

        chunks = hidden_states[:, :10, :, :].flatten(start_dim=1, end_dim=2)

        gen = datagen(
            chunks,
            avatar.input_latent_list_cycle,
            1
        )
        start_time = time.time()
        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(gen):
            audio_feature_batch = pe(whisper_batch.to(DEVICE, unet.model.dtype))
            latent_batch = latent_batch.to(device=DEVICE, dtype=unet.model.dtype)

            # print(unet.model.dtype, latent_batch.dtype, timesteps.dtype, audio_feature_batch)
            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=DEVICE, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)

        frame_np = np.array(recon[0], dtype=np.uint8)

        try:
            frame_q.put_nowait(frame_np)
        except queue.Full:
            pass

def display_loop():
    # cv2.namedWindow("Realtime video", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Realtime video", 640, 480)

    last_time = time.perf_counter()
    while not stop_flag.is_set():
        try:
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.perf_counter()
        fps = 1 / (now - last_time)
        last_time = now
        print(fps)
        # cv2.putText(
        #     frame.astype(np.uint8),
        #     f"{fps:5.1f} FPS",
        #     (10, 25),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        # )

    #     cv2.imshow("Realtime video", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         stop_flag.set()
    #         break

    # cv2.destroyAllWindows()

def main():

    threading.Thread(target=rec_loop, daemon=True).start()
    threading.Thread(target=infer_frame, daemon=True).start()
    threading.Thread(target=display_loop, daemon=True).start()

    while not stop_flag.is_set():
        time.sleep(0.1)
    stop_flag.set()

if __name__ == "__main__":
    main()
