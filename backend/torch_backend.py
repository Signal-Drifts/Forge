"""PyTorch execution backend for Windows and Linux."""

import gc
import json
import shutil
from pathlib import Path

import numpy as np


def require_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ValueError(
            'Install the Windows/Linux PyTorch backend dependencies from backend/requirements.lock.txt.'
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def device_for(engine_id):
    torch, _, _ = require_runtime()
    if engine_id == 'torch-cuda':
        return torch.device('cuda')
    if engine_id == 'torch-xpu':
        return torch.device('xpu')
    if engine_id == 'torch-directml':
        try:
            import torch_directml
        except ImportError as error:
            raise ValueError('Install torch-directml to use the detected DirectML device.') from error
        return torch_directml.device()
    return torch.device('cpu')


def clear_cache(engine_id):
    torch, _, _ = require_runtime()
    gc.collect()
    if engine_id == 'torch-cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif engine_id == 'torch-xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _load(source, engine_id, training=False):
    torch, AutoModelForCausalLM, AutoTokenizer = require_runtime()
    device = device_for(engine_id)
    dtype = torch.float16 if device.type in {'cuda', 'xpu'} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.train(training)
    return model, tokenizer, device


def convert_checkpoint(source, destination, engine_id, dtype=None, progress=None):
    torch, AutoModelForCausalLM, AutoTokenizer = require_runtime()
    destination.mkdir(parents=True, exist_ok=False)
    target_dtype = dtype or torch.float32
    if progress:
        progress('Loading PyTorch checkpoint')
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        local_files_only=True,
        trust_remote_code=False,
        dtype=target_dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    del model, tokenizer
    clear_cache(engine_id)
    return destination


def _copy_non_weights(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.is_file() and not item.name.endswith(('.safetensors', '.bin', '.pt', '.pth')):
            shutil.copy2(item, destination / item.name)
        elif item.is_dir() and item.name != '.cache':
            shutil.copytree(item, destination / item.name)


def _calibration_inputs(source, dataset_path, engine_id, progress=None, max_rows=16, max_tokens=128):
    torch, _, _ = require_runtime()
    model, tokenizer, device = _load(source, engine_id)
    captured = {}
    handles = []

    def capture(name):
        def hook(module, args):
            if not args:
                return
            values = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu()
            previous = sum(chunk.shape[0] for chunk in captured.get(name, []))
            if previous < max_rows:
                captured.setdefault(name, []).append(values[: max_rows - previous])

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(capture(f'{name}.weight')))
    rows = [
        json.loads(line)['text']
        for line in (Path(dataset_path) / 'train.jsonl').read_text().splitlines()
    ]
    try:
        with torch.inference_mode():
            for index, text in enumerate(rows[:8], 1):
                batch = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
                model(**{key: value.to(device) for key, value in batch.items()})
                if progress:
                    progress(f'Calibrated sample {index}/{min(8, len(rows))}')
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        clear_cache(engine_id)
    return {
        name: torch.cat(chunks).numpy()
        for name, chunks in captured.items()
        if chunks
    }


def _sparsegpt(weight, inputs, ratio, block_size=128):
    values = weight.detach().float().cpu().numpy().copy()
    _, columns = values.shape
    for start in range(0, columns, block_size):
        end = min(columns, start + block_size)
        x = inputs[:, start:end].astype(np.float64, copy=False)
        hessian = (x.T @ x) / max(1, x.shape[0])
        damping = max(float(np.diag(hessian).mean()) * 0.01, 1e-5)
        inverse = np.linalg.inv(hessian + np.eye(end - start) * damping)
        block = values[:, start:end].astype(np.float64, copy=True)
        scores = block * block / np.maximum(np.diag(inverse)[None, :] ** 2, 1e-12)
        count = max(1, min(block.shape[1] - 1, int(block.shape[1] * ratio)))
        mask = np.zeros_like(block, dtype=bool)
        selected = np.argpartition(scores, count - 1, axis=1)[:, :count]
        np.put_along_axis(mask, selected, True, axis=1)
        output = block.copy()
        for column in range(block.shape[1]):
            kept = np.where(mask[:, column], 0.0, block[:, column])
            error = (block[:, column] - kept) / max(inverse[column, column], 1e-12)
            block[:, column:] -= error[:, None] * inverse[column, column:][None, :]
            output[:, column] = kept
        values[:, start:end] = output.astype(np.float32)
    return values


def prune_model(source, destination, ratio, method, dataset_path, engine_id, progress=None):
    if not 0 < ratio <= 0.8:
        raise ValueError('Pruning target must be between 1 and 80 percent.')
    torch, _, _ = require_runtime()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    _copy_non_weights(source, destination)
    calibration = {}
    if method in {'wanda', 'sparsegpt'}:
        if not dataset_path:
            raise ValueError(f'{method} requires a calibration dataset.')
        calibration = _calibration_inputs(source, dataset_path, engine_id, progress)
    shards = sorted(source.glob('*.safetensors'))
    if not shards:
        raise ValueError('No Safetensors weights were found for pruning.')
    for shard_index, shard in enumerate(shards, 1):
        tensors = load_file(shard)
        with safe_open(shard, framework='pt', device='cpu') as opened:
            metadata = opened.metadata()
        pruned = {}
        for name, tensor in tensors.items():
            array = tensor
            if tensor.numel() and tensor.ndim == 2 and tensor.is_floating_point():
                values = tensor.float()
                if method == 'magnitude':
                    flat = values.abs().reshape(-1)
                    count = max(1, min(flat.numel() - 1, int(flat.numel() * ratio)))
                    cutoff = torch.kthvalue(flat, count).values
                    array = torch.where(values.abs() <= cutoff, 0, values).to(tensor.dtype)
                elif method == 'structured':
                    scores = values.square().sum(dim=1)
                    count = max(1, min(scores.numel() - 1, int(scores.numel() * ratio)))
                    selected = torch.topk(scores, count, largest=False).indices
                    array = tensor.clone()
                    array[selected] = 0
                elif method == 'nm':
                    complete = tensor.shape[-1] // 4 * 4
                    head = values[:, :complete].reshape(values.shape[0], -1, 4)
                    selected = torch.topk(head.abs(), 2, dim=-1, largest=False).indices
                    mask = torch.zeros_like(head, dtype=torch.bool).scatter_(-1, selected, True)
                    head = torch.where(mask, 0, head).reshape(values.shape[0], complete)
                    array = torch.cat((head, values[:, complete:]), dim=1).to(tensor.dtype)
                else:
                    inputs = calibration.get(name)
                    if inputs is None:
                        pruned[name] = tensor
                        continue
                    if inputs.shape[1] != tensor.shape[1]:
                        raise ValueError(f'Calibration dimensions do not match {name}.')
                    if method == 'wanda':
                        scale = torch.from_numpy(
                            np.sqrt(np.mean(inputs.astype(np.float64) ** 2, axis=0) + 1e-12)
                        )
                        scores = values.abs() * scale
                        count = max(1, min(scores.shape[-1] - 1, int(scores.shape[-1] * ratio)))
                        selected = torch.topk(scores, count, dim=-1, largest=False).indices
                        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)
                        array = torch.where(mask, 0, values).to(tensor.dtype)
                    else:
                        array = torch.from_numpy(_sparsegpt(tensor, inputs, ratio)).to(tensor.dtype)
            pruned[name] = array.contiguous()
        save_file(pruned, destination / shard.name, metadata=metadata)
        if progress:
            progress(f'Pruned weight file {shard_index}/{len(shards)}')
    config_path = destination / 'config.json'
    config = json.loads(config_path.read_text())
    config['forge_pruning'] = {
        'method': method,
        'target_ratio': ratio,
        **({'pattern': '2:4'} if method == 'nm' else {}),
    }
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    return destination


def fine_tune(source, destination, training_path, dataset_path, iterations, engine_id, method='lora', progress=None):
    torch, AutoModelForCausalLM, AutoTokenizer = require_runtime()
    if method not in {'lora', 'qlora', 'full'}:
        raise ValueError('Select LoRA, QLoRA, or Full Fine-Tuning.')
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as error:
        raise ValueError('Install PEFT to fine-tune models on Windows or Linux.') from error
    device = device_for(engine_id)
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if method == 'qlora':
        if engine_id != 'torch-cuda':
            raise ValueError('QLoRA on Windows or Linux requires a CUDA or ROCm PyTorch GPU.')
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = (
                torch.bfloat16
                if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(source),
                local_files_only=True,
                trust_remote_code=False,
                quantization_config=quantization,
                device_map={'': device.index or 0},
                low_cpu_mem_usage=True,
            )
        except ImportError as error:
            raise ValueError('Install bitsandbytes and Accelerate to run QLoRA.') from error
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model, _, device = _load(source, engine_id, training=True)
    if method != 'full':
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_LM',
                target_modules='all-linear',
            ),
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4
    )
    rows = [
        json.loads(line)['text']
        for line in (Path(dataset_path) / 'train.jsonl').read_text().splitlines()
    ]
    if not rows:
        raise ValueError('The training dataset is empty.')
    model.train()
    for step in range(iterations):
        batch = tokenizer(
            rows[step % len(rows)],
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if progress and (step == 0 or (step + 1) % 10 == 0 or step + 1 == iterations):
            progress(f'Training step {step + 1}/{iterations}')
    if method == 'full':
        destination.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        del model, tokenizer
    else:
        model.save_pretrained(training_path, safe_serialization=True)
        del model
        clear_cache(engine_id)
        merge_dtype = torch.float16 if engine_id == 'torch-cuda' else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            str(source),
            local_files_only=True,
            trust_remote_code=False,
            dtype=merge_dtype,
            low_cpu_mem_usage=True,
        )
        merged = PeftModel.from_pretrained(base, training_path).merge_and_unload()
        destination.mkdir(parents=True, exist_ok=False)
        merged.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        del base, merged, tokenizer
    clear_cache(engine_id)
    return destination


def quantize_model(source, destination, bits, engine_id, progress=None):
    torch, AutoModelForCausalLM, AutoTokenizer = require_runtime()
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    if bits in {16, 24, 32}:
        dtype = torch.float16 if bits == 16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(source),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        destination.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        config_path = destination / 'config.json'
        config = json.loads(config_path.read_text())
        if bits == 24:
            config['forge_precision'] = {
                'bits': 24,
                'storage_dtype': 'float32',
                'significand_bits': 24,
            }
        else:
            config.pop('forge_precision', None)
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        del model, tokenizer
        clear_cache(engine_id)
        return destination
    try:
        from transformers import QuantoConfig
    except ImportError as error:
        raise ValueError('Install optimum-quanto to create 4-bit or 8-bit models.') from error
    quantization = QuantoConfig(weights=f'int{bits}')
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        quantization_config=quantization,
        device_map='cpu',
    )
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    del model, tokenizer
    clear_cache(engine_id)
    return destination


def load_for_inference(source, engine_id):
    return _load(source, engine_id)


def generate(model, tokenizer, device, prompt, max_tokens):
    torch, _, _ = require_runtime()
    batch = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
