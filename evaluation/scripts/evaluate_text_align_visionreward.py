#!/usr/bin/env python3
"""Batch VisionReward and Instruction evaluation for text_align videos."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, bridge, cpu
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from validate_text_align import load_manifest


INSTRUCTION_TEMPLATE = (
    'Does the video meet some of the requirements stated in the text '
    '"[[prompt]]"?'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def load_video(video_path: Path):
    bridge.set_bridge("torch")
    video_data = video_path.read_bytes()
    reader = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        raise ValueError(f"Video contains no frames: {video_path}")
    timestamps = reader.get_frame_timestamp(np.arange(total_frames))
    timestamps = [item[0] for item in timestamps]
    max_second = round(max(timestamps)) + 1
    frame_ids = []
    for second in range(max_second):
        closest = min(timestamps, key=lambda value: abs(value - second))
        frame_ids.append(timestamps.index(closest))
        if len(frame_ids) >= 24:
            break
    video = reader.get_batch(frame_ids)
    return video.permute(3, 0, 1, 2)


def ask(model, tokenizer, video, query, device, torch_type):
    conversation = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=[],
        template_version="chat",
    )
    inputs = {
        "input_ids": conversation["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": (
            conversation["token_type_ids"].unsqueeze(0).to(device)
        ),
        "attention_mask": (
            conversation["attention_mask"].unsqueeze(0).to(device)
        ),
        "images": [[conversation["images"][0].to(device).to(torch_type)]],
    }
    generation_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": 0.1,
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
        answer_token = outputs[:, inputs["input_ids"].shape[1]]
    return tokenizer.decode(answer_token[0])


def load_completed(path: Path):
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
            index = record.get("index")
            if index in completed:
                raise ValueError(f"Duplicate index {index} in {path}")
            completed[index] = record
    return completed


def main():
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    manifest = load_manifest(args.manifest, args.expected_count)
    records = manifest["records"]
    selected = [
        record
        for record in records
        if record["index"] % args.num_shards == args.shard_index
    ]
    for record in selected:
        video_path = args.videos_dir / record["video_filename"]
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing video: {video_path}")

    with args.questions_path.open(encoding="utf-8") as handle:
        questions = handle.readlines()
    weights = np.asarray(
        json.loads(args.weights_path.read_text(encoding="utf-8")),
        dtype=np.float64,
    )
    if len(questions) != 29 or weights.shape != (29,):
        raise ValueError(
            f"Expected 29 questions and weights, got "
            f"{len(questions)} and {weights.shape}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official VisionReward evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch_type = (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )

    print(
        f"[VisionReward] Loading {args.model_path} on {device} "
        f"with dtype={torch_type}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch_type,
        trust_remote_code=True,
    ).eval().to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"visionreward-part-{args.shard_index:03d}"
        f"-of-{args.num_shards:03d}.jsonl"
    )
    completed = load_completed(output_path)
    selected_indices = {record["index"] for record in selected}
    stale = sorted(set(completed) - selected_indices)
    if stale:
        raise ValueError(
            f"{output_path} contains indices outside this shard: {stale}"
        )

    remaining = [
        record for record in selected
        if record["index"] not in completed
    ]
    print(
        f"[VisionReward] shard {args.shard_index}/{args.num_shards}: "
        f"{len(completed)} complete, {len(remaining)} remaining"
    )
    with output_path.open("a", encoding="utf-8") as output:
        for record in tqdm(remaining, desc="VisionReward videos"):
            video_path = args.videos_dir / record["video_filename"]
            video = load_video(video_path)
            prompt = record["prompt"]
            qa_results = []
            answer_values = []
            for question_index, (question, weight) in enumerate(
                zip(questions, weights)
            ):
                query = question.replace("[[prompt]]", prompt)
                answer = ask(
                    model, tokenizer, video, query, device, torch_type
                )
                value = 1 if answer == "yes" else -1
                answer_values.append(value)
                qa_results.append(
                    {
                        "question_index": question_index,
                        "question": question.rstrip("\n"),
                        "answer": answer,
                        "value": value,
                        "weight": float(weight),
                    }
                )

            answer_array = np.asarray(answer_values, dtype=np.float64)
            visionreward_score = float(
                np.mean(answer_array * weights)
            )
            instruction_query = INSTRUCTION_TEMPLATE.replace(
                "[[prompt]]", prompt
            )
            instruction_answer = ask(
                model,
                tokenizer,
                video,
                instruction_query,
                device,
                torch_type,
            )
            instruction_value = (
                1 if instruction_answer == "yes" else -1
            )
            result = {
                "index": record["index"],
                "video_filename": record["video_filename"],
                "prompt": prompt,
                "visionreward_score": visionreward_score,
                "visionreward_questions": qa_results,
                "instruction_query": instruction_query,
                "instruction_answer": instruction_answer,
                "instruction_value": instruction_value,
            }
            output.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output.flush()

    print(f"[VisionReward] Results: {output_path}")


if __name__ == "__main__":
    main()
