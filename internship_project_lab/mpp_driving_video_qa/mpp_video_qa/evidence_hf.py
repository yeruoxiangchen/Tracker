"""Modern Qwen3-VL adapter; import only in the external inference environment."""
import copy
import time

from scripts.generate_vlm_teacher import build_processor


class HFSearchBackend:
    def __init__(self, model_path, config):
        import torch
        from transformers import AutoModelForImageTextToText
        if not torch.cuda.is_available():
            raise RuntimeError("external evidence search needs a CUDA-enabled modern environment")
        self.config = config
        self.processor = build_processor(model_path, config["min_pixels"], config["max_pixels"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa").eval().requires_grad_(False)
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.top_p = None
        self.generation_config.top_k = None
        self.generation_config.temperature = None

    def prepare(self, messages):
        images = [c["image"] for m in messages if isinstance(m["content"], list)
                  for c in m["content"] if c["type"] == "image"]
        if not images:
            raise ValueError("visual evidence search requires actual image observations")
        # Bound the WHOLE growing history, not just each individual image.
        p = self.processor.image_processor
        factor = p.patch_size * p.merge_size
        per_image = min(self.config["max_pixels"], self.config["total_pixels"] // len(images))
        per_image = per_image // factor**2 * factor**2
        if per_image < self.config["min_pixels"]:
            raise ValueError("observation history exceeds the total visual budget")
        p.size = {"shortest_edge": self.config["min_pixels"], "longest_edge": per_image}
        p.min_pixels, p.max_pixels = self.config["min_pixels"], per_image
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        spatial_pixels = int((grid[:, 1] * grid[:, 2]).sum()) * p.patch_size**2
        if spatial_pixels > self.config["total_pixels"]:
            raise ValueError("processor exceeded the fixed total pixel budget")
        if inputs.input_ids.shape[1] > self.config["max_input_tokens"]:
            raise ValueError("history exceeds the fixed text+vision token budget; it was not truncated")
        return inputs, {"images": len(images), "per_image_max_pixels": per_image,
                        "actual_spatial_pixels": spatial_pixels, "grid_thw": grid.tolist(),
                        "visual_tokens": int(grid.prod(dim=-1).sum()) // p.merge_size**2,
                        "input_tokens": inputs.input_ids.shape[1]}

    def complete(self, messages):
        import torch
        inputs, metadata = self.prepare(messages)
        inputs = inputs.to(self.model.device)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation_config,
                                         max_new_tokens=self.config["max_new_tokens"], use_cache=True)
        torch.cuda.synchronize()
        completion = output[0, inputs.input_ids.shape[1]:]
        eos = self.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        metadata.update(generate_seconds=time.monotonic() - started, completion_tokens=len(completion),
                        finished=any(x in eos_ids for x in completion.tolist()),
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        return {"text": self.processor.decode(completion, skip_special_tokens=True).strip(),
                "observation": metadata}
