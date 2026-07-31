#!/usr/bin/env python3
"""Evaluate Self-Forcing-style extended prompts with original VBench.

This wrapper leaves the VBench checkout untouched. It maps collision-safe
video basenames to the full extended prompts and applies the human-action
matching rule published by the Causal Forcing authors.
"""

import argparse
import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--full_json_dir", type=Path, required=True)
    parser.add_argument("--videos_path", type=Path, required=True)
    parser.add_argument("--dimension", nargs="+", required=True)
    parser.add_argument("--load_ckpt_from_local", type=bool, default=True)
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        default="longer",
    )
    return parser.parse_args()


def words(value):
    return set(re.findall(r"\b[\w']+\b", value.lower()))


def main():
    args = parse_args()
    sys.path.insert(0, str(args.vbench_root.resolve()))

    import torch
    from tqdm import tqdm
    from vbench import VBench
    from vbench.distributed import dist_init, get_rank, print0
    from vbench.utils import load_json, save_json
    import vbench.distributed as distributed_module
    import vbench.human_action as human_action_module

    with args.full_json_dir.open("r", encoding="utf-8") as handle:
        extended_info = json.load(handle)
    prompt_by_basename = {
        entry["video_basename"]: entry["prompt_en"]
        for entry in extended_info
    }
    original_prompt_by_basename = {
        entry["video_basename"]: entry["original_prompt_en"]
        for entry in extended_info
    }
    if len(prompt_by_basename) != len(extended_info):
        raise ValueError("video_basename values in metadata are not unique")

    def safe_all_gather(data):
        """Gather Python objects without VBench's unsafe size-tensor code."""
        world_size = distributed_module.get_world_size()
        if world_size == 1:
            return [data]
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, data)
        return gathered

    def safe_gather_list_of_dict(results):
        return [item for rank_results in safe_all_gather(results)
                for item in rank_results]

    # VBench's custom arbitrary-object all_gather can misread a serialized
    # byte count as a huge tensor length on some PyTorch/NCCL combinations.
    # Patch the shared module before dimension modules are imported. The human
    # action module was imported above, so update its bound references too.
    distributed_module.all_gather = safe_all_gather
    distributed_module.gather_list_of_dict = safe_gather_list_of_dict
    human_action_module.all_gather = safe_all_gather
    human_action_module.gather_list_of_dict = safe_gather_list_of_dict

    if "color" in args.dimension:
        import vbench.color as color_module

        original_color = color_module.color

        def color_with_original_target(model, video_dict, device):
            # VBench's color parser expects a short prompt such as
            # "a red bicycle" and derives the object name by deleting the
            # color word. Feeding the LLM-expanded paragraph makes the whole
            # paragraph the object key, so no detection can ever match.
            patched_video_dict = []
            for source_info in video_dict:
                info = copy.deepcopy(source_info)
                if info["video_list"]:
                    stem = Path(info["video_list"][0]).stem
                    basename, separator, sample_index = stem.rpartition("-")
                    if not separator or not sample_index.isdigit():
                        raise ValueError(
                            f"Unexpected VBench video name: {stem}"
                        )
                    info["prompt"] = original_prompt_by_basename[basename]
                patched_video_dict.append(info)
            try:
                return original_color(model, patched_video_dict, device)
            except ZeroDivisionError:
                # A rank may legitimately have no GRiT object matches. Its
                # empty contribution is handled after gathering all ranks.
                return 0.0, []

        def compute_color_extended(
            json_dir,
            device,
            submodules_dict,
            **kwargs,
        ):
            dense_caption_model = color_module.DenseCaptioning(device)
            dense_caption_model.initialize_model(**submodules_dict)
            color_module.logger.info("Initialize detection model success")
            _, prompt_dict_ls = color_module.load_dimension_info(
                json_dir,
                dimension="color",
                lang="en",
            )
            prompt_dict_ls = color_module.distribute_list_to_rank(
                prompt_dict_ls
            )
            _, video_results = color_with_original_target(
                dense_caption_model,
                prompt_dict_ls,
                device,
            )
            if color_module.get_world_size() > 1:
                video_results = safe_gather_list_of_dict(video_results)
            if not video_results:
                return 0.0, []
            score = sum(
                result["cur_success_frame_rate"]
                for result in video_results
            ) / len(video_results)
            return score, video_results

        color_module.color = color_with_original_target
        color_module.compute_color = compute_color_extended

    def build_extended_full_info(
        self,
        videos_path,
        name,
        dimension_list,
        prompt_list=None,
        special_str="",
        verbose=False,
        custom_image_folder=None,
        mode="vbench_standard",
        **kwargs,
    ):
        if mode != "vbench_standard":
            raise ValueError(
                "The extended-prompt wrapper only supports vbench_standard"
            )

        video_names = set(os.listdir(videos_path))
        selected = []
        for source_entry in load_json(self.full_info_dir):
            if not set(dimension_list) & set(source_entry["dimension"]):
                continue

            entry = copy.deepcopy(source_entry)
            basename = entry["video_basename"]
            entry["video_list"] = []
            for sample_index in range(5):
                intended_name = (
                    f"{basename}{special_str}-{sample_index}.mp4"
                )
                if intended_name in video_names:
                    entry["video_list"].append(
                        os.path.join(videos_path, intended_name)
                    )
                    if verbose:
                        print0(f"Successfully found video: {intended_name}")
                else:
                    print0(
                        "WARNING!!! Missing benchmark video: "
                        f"{intended_name}"
                    )
            selected.append(entry)

        output_path = os.path.join(self.output_path, name + "_full_info.json")
        save_json(selected, output_path)
        print0(f"Evaluation meta data saved to {output_path}")
        return output_path

    def extended_human_action(umt_path, video_list, device):
        state_dict = torch.load(umt_path, map_location="cpu")
        model = human_action_module.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=400,
            all_frames=16,
            tubelet_size=1,
            use_learnable_pos_emb=False,
            fc_drop_rate=0.0,
            drop_rate=0.0,
            drop_path_rate=0.2,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=False,
            checkpoint_num=16,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        data_transform = human_action_module.Compose(
            [
                human_action_module.Resize(256, interpolation="bilinear"),
                human_action_module.CenterCrop(size=(224, 224)),
                human_action_module.ClipToTensor(),
                human_action_module.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        model = model.to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        category_dict = human_action_module.build_dict()

        correct = 0
        video_results = []
        for video_path in tqdm(video_list, disable=get_rank() > 0):
            stem = Path(video_path).stem
            basename, separator, sample_index = stem.rpartition("-")
            if not separator or not sample_index.isdigit():
                raise ValueError(f"Unexpected VBench video name: {stem}")
            if basename not in prompt_by_basename:
                raise KeyError(
                    f"No extended prompt mapped for basename: {basename}"
                )
            extended_prompt = prompt_by_basename[basename]

            images = human_action_module.load_video(
                video_path,
                data_transform,
                num_frames=16,
            )
            images = images.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = torch.sigmoid(model(images))
                scores, indices = torch.topk(logits, 5, dim=1)

            predicted_categories = []
            for score, index in zip(
                scores.squeeze().tolist(),
                indices.squeeze().tolist(),
            ):
                if round(score, 4) >= 0.85:
                    predicted_categories.append(category_dict[str(index)])

            matched = any(
                words(category) & words(extended_prompt)
                for category in predicted_categories
            )
            correct += int(matched)
            video_results.append(
                {
                    "video_path": video_path,
                    "video_results": matched,
                    "cor_num_per_video": int(matched),
                }
            )

        accuracy = correct / len(video_list) if video_list else 0.0
        return accuracy, video_results

    VBench.build_full_info_json = build_extended_full_info
    human_action_module.human_action = extended_human_action

    dist_init()
    print0(f"args: {args}")
    device = torch.device("cuda")
    evaluator = VBench(device, str(args.full_json_dir), str(args.output_path))
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S-%f")
    evaluator.evaluate(
        videos_path=str(args.videos_path),
        name=f"results_{current_time}",
        prompt_list=[],
        dimension_list=args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=False,
        mode="vbench_standard",
        imaging_quality_preprocessing_mode=(
            args.imaging_quality_preprocessing_mode
        ),
    )
    print0("done")

    # Some VBench third-party native libraries (notably MUSIQ/pyiqa in this
    # environment) can corrupt the allocator while Python tears down objects
    # after a completely successful evaluation. Synchronize after rank 0 has
    # saved the result, close NCCL explicitly, flush logs, then bypass native
    # library destructors. At this point no required work remains in-process.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
