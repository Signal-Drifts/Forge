import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import storage
from .models import hardware, inspect_local


def validate_plan(model, payload):
    if not model.get('downloaded') or not model.get('path'):
        raise ValueError('Download or open the model before optimization.')
    if not model.get('compatible'):
        raise ValueError(model.get('compatibility_error') or 'This model is not compatible.')
    if model.get('model_format') not in {'safetensors', 'mlx', 'pytorch'}:
        raise ValueError('Optimization requires Safetensors weights; inference-only formats cannot be trained directly.')
    engine_id = hardware().get('engine_id')
    if engine_id != 'mlx' and not str(engine_id).startswith('torch-'):
        raise ValueError('Install the supported MLX or PyTorch execution backend before changing model weights.')
    if model.get('model_format') == 'mlx' and engine_id != 'mlx':
        raise ValueError('MLX weights require Apple silicon with the MLX runtime.')
    if model.get('model_format') == 'pytorch' and not str(engine_id).startswith('torch-'):
        raise ValueError('PyTorch checkpoints require the Windows or Linux PyTorch backend.')
    if not any(payload.get(key) for key in ('fine_tune', 'prune', 'quantize')):
        raise ValueError('Select at least one optimization operation.')
    if payload.get('fine_tune') and not payload.get('dataset_id'):
        raise ValueError('Select a JSONL dataset for fine-tuning.')
    fine_tune_method = payload.get('fine_tune_method', 'lora')
    if payload.get('fine_tune') and fine_tune_method not in {'lora', 'qlora', 'full'}:
        raise ValueError('Select LoRA, QLoRA, or Full Fine-Tuning.')
    if payload.get('fine_tune') and fine_tune_method == 'full' and model.get('source_bits') not in (16, 24, 32):
        raise ValueError('Full Fine-Tuning requires a 16-bit, 24-bit, or 32-bit model.')
    if payload.get('fine_tune') and fine_tune_method == 'qlora' and str(engine_id).startswith('torch-') and engine_id != 'torch-cuda':
        raise ValueError('QLoRA on Windows or Linux requires a CUDA or ROCm PyTorch GPU.')
    method = payload.get('prune_method', 'magnitude')
    if payload.get('prune') and method not in {'magnitude', 'wanda', 'sparsegpt', 'structured', 'nm'}:
        raise ValueError('Select a supported pruning method.')
    if payload.get('prune') and method in {'wanda', 'sparsegpt'} and not payload.get('dataset_id'):
        raise ValueError(f'{method.upper() if method == "wanda" else "SparseGPT"} requires a JSONL calibration dataset.')
    if payload.get('prune') and model.get('source_bits') not in (16, 24, 32):
        raise ValueError('Pruning requires a 16-bit, 24-bit, or 32-bit model.')
    bits = payload.get('bits', 4)
    if payload.get('quantize') and bits not in (4, 8, 16, 24, 32):
        raise ValueError('Output precision must be 4, 8, 16, 24, or 32 bits.')
    if payload.get('quantize') and bits == model.get('source_bits'):
        raise ValueError('Select an output precision different from the model precision.')


def _copy_model(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if destination.is_relative_to(item):
            continue
        if item.is_file() and not item.name.endswith('.safetensors'):
            shutil.copy2(item, destination / item.name)
        elif item.is_dir():
            shutil.copytree(item, destination / item.name)


def _row_mask(scores, ratio):
    import mlx.core as mx
    count = max(1, min(scores.shape[-1] - 1, int(scores.shape[-1] * ratio)))
    order = mx.argsort(scores, axis=-1)
    ranks = mx.argsort(order, axis=-1)
    return ranks < count


def _calibration_inputs(source, dataset_path, progress=None, max_rows=16, max_tokens=128):
    """Capture real Linear inputs from representative dataset examples."""
    import mlx.core as mx
    from mlx import nn
    from mlx_lm import load

    model, tokenizer = load(str(source), tokenizer_config={'trust_remote_code': False})
    names = {id(module): name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    captured = {name: [] for name in names.values()}
    original = nn.Linear.__call__

    def recording_call(module, x):
        name = names.get(id(module))
        if name and sum(chunk.shape[0] for chunk in captured[name]) < max_rows:
            flat = x.reshape(-1, x.shape[-1])
            remaining = max_rows - sum(chunk.shape[0] for chunk in captured[name])
            sample = flat[:remaining].astype(mx.float32)
            mx.eval(sample)
            captured[name].append(np.asarray(sample))
        return original(module, x)

    rows = [json.loads(line)['text'] for line in (Path(dataset_path) / 'train.jsonl').read_text().splitlines()]
    try:
        nn.Linear.__call__ = recording_call
        for index, text in enumerate(rows[:8], 1):
            tokens = tokenizer.encode(text)[:max_tokens]
            if not tokens:
                continue
            output = model(mx.array([tokens]))
            mx.eval(output)
            if progress:
                progress(f'Calibrated sample {index}/{min(8, len(rows))}')
    finally:
        nn.Linear.__call__ = original
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    return {f'{name}.weight': np.concatenate(chunks, axis=0) for name, chunks in captured.items() if chunks}


def _sparsegpt(weight, inputs, ratio, block_size=128):
    """Blockwise SparseGPT update using calibration-input Hessians."""
    import mlx.core as mx
    values = np.asarray(weight.astype(mx.float32)).copy()
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
    return mx.array(values).astype(weight.dtype)


def prune_model(source, destination, ratio, method='magnitude', dataset_path=None, progress=None):
    import mlx.core as mx

    if not 0 < ratio <= 0.8:
        raise ValueError('Pruning target must be between 1 and 80 percent.')
    _copy_model(source, destination)
    calibration = {}
    if method in {'wanda', 'sparsegpt'}:
        if not dataset_path:
            raise ValueError(f'{method} requires a calibration dataset.')
        calibration = _calibration_inputs(source, dataset_path, progress)
    shards = sorted(source.glob('*.safetensors'))
    if not shards:
        raise ValueError('No Safetensors weights were found for pruning.')
    for shard_index, shard in enumerate(shards, 1):
        arrays, metadata = mx.load(str(shard), return_metadata=True)
        pruned = {}
        for name, array in arrays.items():
            if array.size and array.ndim == 2 and mx.issubdtype(array.dtype, mx.floating):
                if method == 'magnitude':
                    absolute = mx.abs(array).reshape(-1)
                    cutoff_index = min(absolute.size - 1, max(0, int(absolute.size * ratio) - 1))
                    cutoff = mx.partition(absolute, cutoff_index)[cutoff_index]
                    array = mx.where(mx.abs(array) <= cutoff, mx.zeros_like(array), array)
                elif method == 'structured':
                    row_scores = mx.sum(mx.square(array.astype(mx.float32)), axis=1)
                    row_count = max(1, min(array.shape[0] - 1, int(array.shape[0] * ratio)))
                    row_order = mx.argsort(row_scores)
                    row_ranks = mx.argsort(row_order)
                    array = mx.where((row_ranks < row_count)[:, None], mx.zeros_like(array), array)
                elif method == 'nm':
                    group = 4
                    prune_count = 2
                    complete = array.shape[-1] // group * group
                    head = array[:, :complete].reshape(array.shape[0], -1, group)
                    mask = _row_mask(mx.abs(head), prune_count / group)
                    head = mx.where(mask, mx.zeros_like(head), head).reshape(array.shape[0], complete)
                    array = mx.concatenate([head, array[:, complete:]], axis=1) if complete < array.shape[-1] else head
                else:
                    inputs = calibration.get(name)
                    if inputs is None:
                        pruned[name] = array
                        continue
                    if inputs.shape[1] != array.shape[1]:
                        raise ValueError(f'Calibration dimensions do not match {name}.')
                    if method == 'wanda':
                        scale = mx.array(np.sqrt(np.mean(inputs.astype(np.float64) ** 2, axis=0) + 1e-12))
                        scores = mx.abs(array) * scale[None, :]
                        array = mx.where(_row_mask(scores, ratio), mx.zeros_like(array), array)
                    elif method == 'sparsegpt':
                        array = _sparsegpt(array, inputs, ratio)
            pruned[name] = array
        mx.save_safetensors(str(destination / shard.name), pruned, metadata)
        if progress:
            progress(f'Pruned weight file {shard_index}/{len(shards)}')
    config_path = destination / 'config.json'
    config = json.loads(config_path.read_text())
    config['forge_pruning'] = {'method': method, 'target_ratio': ratio, **({'pattern': '2:4'} if method == 'nm' else {})}
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    return destination


def magnitude_prune(source, destination, ratio, progress=None):
    return prune_model(source, destination, ratio, 'magnitude', progress=progress)


def _run(command, progress):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.strip()
        if clean:
            tail.append(clean)
            tail = tail[-20:]
            display = clean.removeprefix('[INFO] ').removeprefix('INFO: ')
            progress(display[:300])
    code = process.wait()
    if code:
        reason = next((line for line in reversed(tail) if line.startswith(('ValueError:', 'RuntimeError:', 'MemoryError:'))), None)
        raise ValueError((reason or (tail[-1] if tail else 'Model operation failed.'))[:1000])


def run_optimization(job_id, model, payload):
    validate_plan(model, payload)
    source = Path(model['path']).resolve(strict=True)
    if payload.get('output_path'):
        run_directory = Path(payload['output_path']).resolve(strict=True)
        marker = run_directory / '.forge-operation'
        if not marker.is_file():
            raise ValueError('The selected output folder is no longer available.')
    else:
        run_directory = storage.ROOT / 'outputs' / job_id
        if run_directory.exists():
            shutil.rmtree(run_directory, ignore_errors=True)
        if run_directory.exists():
            raise ValueError('Temporary files from this operation could not be removed.')
        run_directory.mkdir(parents=True, exist_ok=False)
        marker = None
    current = source

    def progress(message):
        storage.update_job(job_id, 'running', message)

    engine_id = hardware()['engine_id']
    torch_engine = str(engine_id).startswith('torch-')
    if torch_engine:
        from . import torch_backend
        if model.get('model_format') == 'pytorch':
            current = torch_backend.convert_checkpoint(
                current,
                run_directory / 'source-safetensors',
                engine_id,
                progress=progress,
            )

    if payload.get('prune'):
        progress('Pruning weights')
        dataset = storage.get(payload['dataset_id'], 'dataset') if payload.get('dataset_id') else None
        if torch_engine:
            current = torch_backend.prune_model(
                current,
                run_directory / 'pruned',
                payload['prune_ratio'],
                payload['prune_method'],
                dataset['path'] if dataset else None,
                engine_id,
                progress,
            )
        else:
            current = prune_model(current, run_directory / 'pruned', payload['prune_ratio'], payload['prune_method'], dataset['path'] if dataset else None, progress)

    adapter_path = None
    if payload.get('fine_tune'):
        dataset = storage.get(payload['dataset_id'], 'dataset')
        fine_tune_method = payload.get('fine_tune_method', 'lora')
        training_path = run_directory / ('full-weights' if fine_tune_method == 'full' else 'adapters')
        adapter_path = None if fine_tune_method == 'full' else training_path
        fused = run_directory / 'fine-tuned'
        progress({
            'lora': 'Training LoRA adapters',
            'qlora': 'Training QLoRA adapters',
            'full': 'Running Full Fine-Tuning',
        }[fine_tune_method])
        if torch_engine:
            current = torch_backend.fine_tune(
                current,
                fused,
                training_path,
                dataset['path'],
                payload['iterations'],
                engine_id,
                fine_tune_method,
                progress,
            )
        else:
            config = json.loads((current / 'config.json').read_text())
            training_source = current
            if fine_tune_method == 'qlora' and not (config.get('quantization') or config.get('quantization_config')):
                training_source = run_directory / 'qlora-base'
                progress('Preparing 4-bit QLoRA base model')
                _run([
                    sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', str(current),
                    '--mlx-path', str(training_source), '--quantize', '--q-bits', '4',
                ], progress)
            layer_count = -1 if fine_tune_method == 'full' else max(1, min(16, int(config.get('num_hidden_layers') or config.get('num_layers') or 16)))
            _run([
                sys.executable, '-m', 'mlx_lm', 'lora', '--model', str(training_source), '--train', '--data', dataset['path'],
                '--fine-tune-type', 'full' if fine_tune_method == 'full' else 'lora', '--adapter-path', str(training_path), '--iters', str(payload['iterations']),
                '--batch-size', '1', '--max-seq-length', '512', '--num-layers', str(layer_count), '--grad-checkpoint',
            ], progress)
            progress('Saving fine-tuned model' if fine_tune_method == 'full' else 'Merging fine-tuning adapters')
            _run([
                sys.executable, '-m', 'mlx_lm', 'fuse', '--model', str(training_source), '--adapter-path', str(training_path),
                '--save-path', str(fused),
            ], progress)
            current = fused

    if payload.get('quantize'):
        output = run_directory / f"{payload['bits']}-bit"
        progress(f"Creating {payload['bits']}-bit model")
        if torch_engine:
            current = torch_backend.quantize_model(
                current, output, payload['bits'], engine_id, progress
            )
        else:
            config = json.loads((current / 'config.json').read_text())
            quantized_input = bool(config.get('quantization') or config.get('quantization_config'))
            quantization_source = current
            if quantized_input and payload['bits'] in (4, 8):
                quantization_source = run_directory / 'dequantized'
                progress('Preparing full-precision weights')
                _run([
                    sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', str(current),
                    '--mlx-path', str(quantization_source), '--dequantize', '--dtype', 'float16',
                ], progress)
            command = [sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', str(quantization_source), '--mlx-path', str(output)]
            if payload['bits'] in (4, 8):
                command += ['--quantize', '--q-bits', str(payload['bits'])]
            else:
                if quantized_input:
                    command += ['--dequantize']
                command += ['--dtype', 'float16' if payload['bits'] == 16 else 'float32']
            _run(command, progress)
            if payload['bits'] == 24:
                config_path = output / 'config.json'
                output_config = json.loads(config_path.read_text())
                output_config['forge_precision'] = {
                    'bits': 24,
                    'storage_dtype': 'float32',
                    'significand_bits': 24,
                }
                config_path.write_text(json.dumps(output_config, indent=2) + '\n')
            current = output

    result_model = inspect_local(str(current))
    if marker:
        marker.unlink(missing_ok=True)
    return {
        'model': result_model,
        'output_path': str(current),
        'adapter_path': str(adapter_path) if adapter_path else None,
        'operations': [name for name in ('prune', 'fine_tune', 'quantize') if payload.get(name)],
        'saved': False,
    }
