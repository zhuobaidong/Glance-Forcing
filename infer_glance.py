import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Subset
import json

from pipeline.slowfast_chunk import CausalInferencePipeline

from pipeline import (
    CausalDiffusionInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/causal_forcing_dmd_chunkwise.yaml")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/chunkwise/ar_diffusion.pt")
parser.add_argument("--lora_path_1", type=str, default="final_logs/glance_slow_ode/slow_continue/checkpoint_model_002000/model.pt")
parser.add_argument("--lora_path_2", type=str, default="final_logs/glance_fast_ode/fast_continue/checkpoint_model_002000/model.pt")
parser.add_argument("--data_path", type=str, default="prompts/evaluate.txt")
parser.add_argument("--output_folder", type=str, default="output/0724/slow2fast2")
parser.add_argument(
    "--output_name_path",
    type=str,
    default=None,
    help="Optional line-aligned file of collision-safe output filename stems",
)
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--seed_by_prompt",
    action="store_true",
    help="Generate each prompt with seed + its original prompt index",
)
parser.add_argument(
    "--sample_index", type=int, default=0,
    help="Sample suffix used in output filenames",
)
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true",
                    help="Only tested on A800, for the Causal Forcing++ latency. Not make claims for other hardware like H100. For the result on H100, refer to the reported results in the Self Forcing paper.")
parser.add_argument("--steps", type=int, choices=[4, 8], default=4, help="Inference steps: 4 or 8")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    global_rank = 0

set_seed(args.seed)

print(f'Free VRAM on {device}: {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules="all-linear",
    )

    print(f"Loading LoRA 1 from {args.lora_path_1}...")
    pipeline.generator = get_peft_model(pipeline.generator, lora_config, adapter_name="lora_1")
    lora_sd_1 = torch.load(args.lora_path_1, map_location="cpu")
    set_peft_model_state_dict(pipeline.generator, lora_sd_1, adapter_name="lora_1")

    print(f"Loading LoRA 2 from {args.lora_path_2}...")
    pipeline.generator.add_adapter("lora_2", lora_config)
    lora_sd_2 = torch.load(args.lora_path_2, map_location="cpu")
    set_peft_model_state_dict(pipeline.generator, lora_sd_2, adapter_name="lora_2")
    pipeline.generator.set_adapter("lora_1")

# ==================== 根据 --steps 自动映射 denoising list 及切换点 ====================
if args.steps == 4:
    full_raw_list = [1000, 750, 500, 250]
    switch_step = 2  # 第 2 个 step 开始切 LoRA 2
elif args.steps == 8:
    full_raw_list = [1000, 875, 750, 625, 500, 375, 250, 125]
    switch_step = 4  # 第 4 个 step 开始切 LoRA 2

timesteps_map = torch.cat((pipeline.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
full_denoising_list = timesteps_map[1000 - torch.tensor(full_raw_list, dtype=torch.long)].to(device=device)
pipeline.denoising_step_list = full_denoising_list

current_active_lora = "lora_1"

def switch_lora_by_step_callback(step_index):
    global current_active_lora
    
    target_lora = "lora_1" if step_index < switch_step else "lora_2"
    
    if current_active_lora != target_lora:
        pipeline.generator.set_adapter(target_lora)
        current_active_lora = target_lora
        print(f" -> [Step {step_index}] Switched to {target_lora.upper()} (Pointer Switch).")

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

output_name_list = None
if args.output_name_path is not None:
    with open(args.output_name_path, encoding="utf-8") as f:
        output_name_list = [line.rstrip("\n") for line in f]
    if len(output_name_list) != num_prompts:
        raise ValueError(
            f"Output-name count ({len(output_name_list)}) does not match "
            f"prompt count ({num_prompts})"
        )
    if len(set(output_name_list)) != len(output_name_list):
        raise ValueError("Output names must be unique")
    for output_name in output_name_list:
        if (
            not output_name
            or output_name in {".", ".."}
            or os.path.basename(output_name) != output_name
        ):
            raise ValueError(f"Unsafe output filename stem: {output_name!r}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    if output_name_list is not None:
        unique_indices = list(range(num_prompts))
    else:
        unique_indices = []
        seen_prompts = set()
        for dataset_idx, prompt in enumerate(dataset.prompt_list):
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                unique_indices.append(dataset_idx)
    rank_indices = unique_indices[global_rank::world_size]
    dataset = Subset(dataset, rank_indices)
    if global_rank == 0:
        print(
            f"Distributed inference: {len(unique_indices)} unique prompts "
            f"across {world_size} GPUs"
        )
    print(f"Rank {global_rank}: {len(rank_indices)} prompts")

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier(device_ids=[local_rank])

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def get_output_path(prompt: str, prompt_idx: int) -> str:
    if output_name_list is None:
        # Preserve the repository's original demo/inference filename format.
        return os.path.join(args.output_folder, f"{prompt[:100]}.mp4")

    return os.path.join(
        args.output_folder,
        f"{output_name_list[prompt_idx]}-{args.sample_index}.mp4",
    )


def sample_noise(shape: list[int], prompt_idx: int) -> torch.Tensor:
    generator = None
    if args.seed_by_prompt:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + prompt_idx)
    return torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = get_output_path(prompt, idx)
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = sample_noise(
            [1, args.num_output_frames - 1, 16, 60, 104],
            idx,
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = get_output_path(prompt, idx)
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = sample_noise(
            [1, args.num_output_frames, 16, 60, 104],
            idx,
        )

    sample_report_timing = args.report_timing and i >= 1
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        report_timing=sample_report_timing,
        before_block_callback=switch_lora_by_step_callback
    )
    if sample_report_timing:
        latency = pipeline.first_chunk_time
        elapsed = pipeline.last_generation_time
        num_pixel_frames = video.shape[1]
        fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
        print(f"[Sample {i}] {num_pixel_frames} frames, "
              f"latency ↓ {latency:.2f}s, FPS ↑ {fps:.2f}")
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    write_video(output_path, video[0], fps=16)

       
