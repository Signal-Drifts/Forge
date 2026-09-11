import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import psutil
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id

from .storage import ROOT, get, save

GB = 1_000_000_000


def _run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def pick_folder(prompt='Choose Model Folder'):
    """Open the operating system's folder chooser and return its selected path."""
    system = platform.system()
    if system == 'Darwin':
        command = [
            'osascript',
            '-e',
            f'POSIX path of (choose folder with prompt "{prompt}")',
        ]
    elif system == 'Windows':
        script = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; '
            f'$dialog.Description = "{prompt}"; '
            '$dialog.UseDescriptionForTitle = $true; '
            'if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) '
            '{ Write-Output $dialog.SelectedPath }'
        )
        command = ['powershell.exe', '-NoProfile', '-STA', '-Command', script]
    elif shutil.which('zenity'):
        command = ['zenity', '--file-selection', '--directory', f'--title={prompt}']
    elif shutil.which('kdialog'):
        command = ['kdialog', '--getexistingdirectory', str(Path.home())]
    else:
        raise ValueError('No system folder chooser is available. Enter the model folder path directly.')
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ValueError('The system folder chooser could not be opened. Enter the model folder path directly.') from error
    if result.returncode != 0 or not result.stdout.strip():
        return None
    path = Path(result.stdout.strip()).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError('Select a folder.')
    return str(path)


def _windows_gpu():
    nvidia = _run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'])
    if nvidia:
        name, _, memory = nvidia.splitlines()[0].partition(',')
        memory_bytes = int(float(memory.strip()) * 1024 * 1024) if memory.strip() else None
        return name.strip(), memory_bytes, 'nvidia'
    script = "$g=Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM; $g | ConvertTo-Json -Compress"
    raw = _run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script])
    try:
        devices = json.loads(raw) if raw else []
        devices = devices if isinstance(devices, list) else [devices]
        devices = [device for device in devices if isinstance(device, dict) and device.get('Name')]
        if devices:
            devices.sort(key=lambda device: ('microsoft basic' in device['Name'].lower(), 'intel' in device['Name'].lower()))
            device = devices[0]
            name = str(device['Name']).strip()
            vendor = 'nvidia' if 'nvidia' in name.lower() else 'amd' if any(value in name.lower() for value in ('amd', 'radeon')) else 'intel' if 'intel' in name.lower() else 'other'
            memory = device.get('AdapterRAM')
            return name, int(memory) if isinstance(memory, (int, float)) and memory > 0 else None, vendor
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return 'Not detected', None, None


def _windows_engine(vendor):
    if importlib.util.find_spec('torch'):
        try:
            import torch
            if torch.cuda.is_available():
                return 'torch-cuda', 'PyTorch CUDA available'
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return 'torch-xpu', 'PyTorch Intel XPU available'
        except (ImportError, RuntimeError):
            pass
    if vendor in ('nvidia', 'amd', 'intel') and importlib.util.find_spec('torch_directml'):
        return 'torch-directml', 'PyTorch DirectML available'
    if importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'):
        return 'torch-cpu', 'PyTorch CPU available'
    return None, 'Unavailable'


def _linux_gpu():
    nvidia = _run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'])
    if nvidia:
        name, _, memory = nvidia.splitlines()[0].partition(',')
        memory_bytes = int(float(memory.strip()) * 1024 * 1024) if memory.strip() else None
        return name.strip(), memory_bytes, 'nvidia'
    if importlib.util.find_spec('torch'):
        try:
            import torch
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return str(torch.xpu.get_device_name(0)), None, 'intel'
            if getattr(torch.version, 'hip', None):
                return 'AMD ROCm GPU', None, 'amd'
        except (ImportError, RuntimeError):
            pass
    return 'Not detected', None, None


def _linux_engine(vendor):
    if not (importlib.util.find_spec('torch') and importlib.util.find_spec('transformers')):
        return None, 'Unavailable'
    try:
        import torch
        if torch.cuda.is_available():
            label = 'PyTorch ROCm available' if getattr(torch.version, 'hip', None) else 'PyTorch CUDA available'
            return 'torch-cuda', label
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return 'torch-xpu', 'PyTorch Intel XPU available'
    except (ImportError, RuntimeError):
        pass
    return 'torch-cpu', 'PyTorch CPU available'


def hardware():
    mem = psutil.virtual_memory()
    ROOT.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    architecture = platform.machine()
    apple = system == 'Darwin' and architecture == 'arm64'
    chip = platform.processor() or platform.machine()
    gpu, gpu_memory, vendor = 'Not detected', None, None
    engine_id, engine = None, 'Unavailable'
    graphics_requirement = 'Supported local GPU'
    if apple:
        chip = _run(['sysctl', '-n', 'machdep.cpu.brand_string']) or chip
        gpu = f'{chip} integrated GPU'
        vendor = 'apple'
        graphics_requirement = 'Apple silicon GPU, M1 or newer'
        if importlib.util.find_spec('mlx_lm'):
            engine_id, engine = 'mlx', 'MLX available'
    elif system == 'Darwin':
        engine_id, engine = _linux_engine(None)
        graphics_requirement = 'PyTorch-compatible CPU'
    elif system == 'Windows':
        gpu, gpu_memory, vendor = _windows_gpu()
        engine_id, engine = _windows_engine(vendor)
        graphics_requirement = 'PyTorch-compatible GPU or CPU'
    elif system == 'Linux':
        gpu, gpu_memory, vendor = _linux_gpu()
        engine_id, engine = _linux_engine(vendor)
        graphics_requirement = 'PyTorch-compatible GPU or CPU'
    return {'chip': chip, 'architecture': architecture, 'os': system, 'gpu': gpu, 'gpu_memory_bytes': gpu_memory, 'gpu_vendor': vendor, 'graphics_requirement': graphics_requirement, 'engine_id': engine_id, 'engine': engine, 'ram_total_bytes': mem.total, 'ram_available_bytes': mem.available, 'disk_free_bytes': shutil.disk_usage(ROOT).free, 'model_directory': str(ROOT / 'models')}


def repository_id(value):
    value = value.strip()
    if value.lower().endswith('.gguf'):
        raise ValueError('Enter the repository ID, not a GGUF filename. Example: unsloth/Qwen3.8-27B-GGUF.')
    if value.startswith(('https://', 'http://')):
        url = urlparse(value)
        if url.scheme != 'https' or url.hostname != 'huggingface.co' or url.username or url.query or url.fragment:
            raise ValueError('Use a Hugging Face model ID or an https://huggingface.co model link.')
        pieces = url.path.strip('/').split('/')
        if len(pieces) != 2:
            raise ValueError('Use the model page link, not a file or dataset link.')
        value = '/'.join(pieces)
    if value.count('/') != 1:
        raise ValueError('Enter the full Hugging Face model ID as owner/model-name, or paste its model-page link.')
    validate_repo_id(value)
    return value


def read_config(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError('Model configuration is too large.')
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Invalid model configuration.')  # noqa: TRY004
    return config


def supported(config):
    model_type = config.get('model_type', '')
    if not re.fullmatch(r'[a-zA-Z0-9_]+', model_type):
        return False, 'Missing or invalid model architecture.'
    quant = config.get('quantization_config') or {}
    engine_id = hardware()['engine_id']
    if engine_id == 'mlx':
        aliases = {'mistral': 'llama'}
        if importlib.util.find_spec('mlx_lm.models.' + aliases.get(model_type, model_type)) is None:
            return False, f'Model architecture {model_type} is not supported by the installed MLX engine.'
        if quant and quant.get('quant_method') not in (None, 'mlx'):
            return False, 'This quantization format is not supported by the MLX loader; use an MLX or unquantized Safetensors model.'
        return True, None
    if engine_id and engine_id.startswith('torch-'):
        if config.get('quantization') or quant.get('quant_method') == 'mlx':
            return False, 'MLX-quantized weights require Apple silicon; choose standard Hugging Face Safetensors weights.'
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            if model_type not in CONFIG_MAPPING_NAMES:
                return False, f'Model architecture {model_type} is not supported by the installed Transformers runtime.'
        except ImportError:
            return False, 'Install the PyTorch backend dependencies before running this model.'
        return True, None
    return False, 'Install the model execution dependencies for this operating system.'


def source_bits(config):
    forge_precision = config.get('forge_precision') or {}
    if forge_precision.get('bits') == 24:
        return 24
    quant = config.get('quantization') or config.get('quantization_config') or {}
    bits = quant.get('bits')
    if isinstance(bits, int) and 1 <= bits <= 32:
        return bits
    dtype = str(config.get('torch_dtype', config.get('dtype', '')))
    return {'float32': 32, 'float16': 16, 'bfloat16': 16}.get(dtype)


FORMAT_NAMES = {
    'safetensors': 'Safetensors',
    'mlx': 'MLX Safetensors',
    'pytorch': 'PyTorch checkpoint',
    'gguf': 'GGUF',
    'onnx': 'ONNX',
    'tensorflow': 'TensorFlow SavedModel',
    'tensorrt': 'TensorRT engine',
    'unknown': 'Unknown',
}


def detect_model_format(config, names):
    names = [name.lower() for name in names]
    has_safetensors = any(name.endswith('.safetensors') for name in names)
    quantization = config.get('quantization') or config.get('quantization_config') or {}
    if has_safetensors and (config.get('quantization') or quantization.get('quant_method') == 'mlx'):
        format_id = 'mlx'
    elif has_safetensors:
        format_id = 'safetensors'
    elif any(name.endswith('.gguf') for name in names):
        format_id = 'gguf'
    elif any(name.endswith(('.bin', '.pt', '.pth')) for name in names):
        format_id = 'pytorch'
    elif any(name.endswith('.onnx') for name in names):
        format_id = 'onnx'
    elif any(name == 'saved_model.pb' or name.endswith('/saved_model.pb') for name in names):
        format_id = 'tensorflow'
    elif any(name.endswith(('.engine', '.plan')) for name in names):
        format_id = 'tensorrt'
    else:
        format_id = 'unknown'
    return format_id, FORMAT_NAMES[format_id]


def metadata(config, files, origin, revision=None, parameter_count=None, format_names=None):
    parameters_estimated = parameter_count is None
    weights = [f for f in files if f['name'].endswith(('.safetensors', '.bin', '.pt', '.pth'))]
    format_id, format_name = detect_model_format(config, format_names or [f['name'] for f in files])
    ok, error = supported(config)
    if not weights:
        errors = {
            'gguf': 'GGUF optimization is not supported. Choose a supported Safetensors model.',
            'pytorch': 'No downloadable root-level PyTorch weights were found. Choose a repository with complete model weights.',
            'onnx': 'ONNX optimization is not supported. Choose a supported Safetensors model.',
            'tensorflow': 'TensorFlow SavedModel optimization is not supported. Choose a supported Safetensors model.',
            'tensorrt': 'TensorRT optimization is not supported. Choose a supported Safetensors model.',
        }
        ok, error = False, errors.get(format_id, 'No supported model weights were found. Choose a complete Safetensors model folder.')
    if weights and format_id == 'pytorch' and hardware()['engine_id'] == 'mlx':
        ok, error = False, 'PyTorch checkpoints require the Windows or Linux PyTorch backend; choose Safetensors for MLX.'
    tokenizer_names = {'tokenizer.json', 'tokenizer.model', 'vocab.json', 'vocab.txt'}
    if weights and not any(f['name'] in tokenizer_names for f in files):
        ok, error = False, 'Tokenizer files are missing from this model.'
    weight_bytes = sum(f['size'] for f in weights)
    bits = source_bits(config)
    if not parameter_count and bits and bits >= 16:
        parameter_count = int(weight_bytes * 8 / bits)
    return {'name': origin, 'revision': revision, 'model_type': config.get('model_type'), 'model_format': format_id, 'model_format_name': format_name, 'source_bits': bits, 'parameter_count': parameter_count, 'parameters_estimated': parameters_estimated, 'weight_bytes': weight_bytes, 'largest_shard_bytes': max((f['size'] for f in weights), default=0), 'download_bytes': sum(f['size'] for f in files), 'files': files, 'compatible': ok, 'compatibility_error': error, 'memory_estimates_are_approximate': True}


def weight_header(path):
    with path.open('rb') as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError('Invalid Safetensors file: ' + path.name)
        length = struct.unpack('<Q', raw)[0]
        if length > 32 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError('Invalid Safetensors header: ' + path.name)
        header = json.loads(file.read(length))
    tensors = [v for k, v in header.items() if k != '__metadata__']
    if not tensors:
        raise ValueError('Empty model weights: ' + path.name)
    parameters = 0
    for tensor in tensors:
        shape = tensor.get('shape', [])
        offsets = tensor.get('data_offsets', [])
        if not isinstance(shape, list) or any(not isinstance(n, int) or n < 0 for n in shape) or len(offsets) != 2 or not all(isinstance(n, int) for n in offsets) or not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - length - 8:
            raise ValueError('Invalid model weight shape or offsets.')
        parameters += math.prod(shape)
    return parameters


def inspect_local(value):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError('Select a model folder.')
    # Model repositories keep their configuration and primary weights at the root.
    # Avoid recursively traversing a broad user-selected directory such as $HOME.
    entries = list(path.iterdir())
    format_names = [file.name for file in entries if file.is_file()]
    saved_model = path / 'saved_model.pb'
    if saved_model.is_file() and saved_model.name not in format_names:
        format_names.append(saved_model.name)
    format_id, _ = detect_model_format({}, format_names)
    if not (path / 'config.json').is_file():
        if format_id == 'unknown':
            raise ValueError('Select the model folder containing config.json and model weights.')
        info = metadata({}, [], path.name, format_names=format_names)
        info.update(path=str(path), source='local', downloaded=True)
        return save('model', info, hashlib.sha256(str(path).encode()).hexdigest()[:24])
    files = [{'name': p.name, 'size': p.stat().st_size} for p in path.iterdir() if p.is_file() and allowed_file(p.name)]
    index = path / 'model.safetensors.index.json'
    if index.exists():
        config_index = read_config(index)
        for shard in set(config_index.get('weight_map', {}).values()):
            if Path(shard).name != shard or not (path / shard).is_file():
                raise ValueError('One or more model weight shards are missing or invalid.')
    config = read_config(path / 'config.json')
    count = sum(weight_header(path / f['name']) for f in files if f['name'].endswith('.safetensors'))
    info = metadata(config, files, path.name, parameter_count=count if (source_bits(config) or 0) >= 16 else None, format_names=format_names)
    info.update(path=str(path), source='local', downloaded=True)
    return save('model', info, hashlib.sha256(str(path).encode()).hexdigest()[:24])


def allowed_file(name):
    return '/' not in name and name.endswith(('.safetensors', '.bin', '.pt', '.pth', '.json', '.txt', '.model', '.tiktoken'))


def inspect_hub(value):
    repo = repository_id(value)
    info = HfApi().model_info(repo, files_metadata=True)
    format_names = [f.rfilename for f in info.siblings]
    files = [{'name': f.rfilename, 'size': f.size or 0} for f in info.siblings if allowed_file(f.rfilename)]
    preliminary_format, _ = detect_model_format({}, format_names)
    config = {} if preliminary_format in {'gguf', 'onnx', 'tensorflow', 'tensorrt'} else read_config(Path(hf_hub_download(repo, 'config.json', revision=info.sha)))
    count = info.safetensors.total if info.safetensors else None
    data = metadata(config, files, repo, info.sha, count, format_names=format_names)
    data.update(source='huggingface', repo_id=repo, path=None, downloaded=False)
    record_id = hashlib.sha256((repo + info.sha).encode()).hexdigest()[:24]
    try:
        previous = get(record_id, 'model')
        if previous.get('path') and all((Path(previous['path']) / f['name']).is_file() for f in files):
            data.update(path=previous['path'], downloaded=True)
    except ValueError:
        pass
    return save('model', data, record_id)


def estimate(model, operations, bits=4, available_bytes=None, fine_tune_method='lora'):
    hw = hardware()
    available = hw['ram_available_bytes'] if available_bytes is None else available_bytes
    source = model['weight_bytes']
    shard = model['largest_shard_bytes']
    count = model.get('parameter_count')
    output = count * bits / 8 if count else source * bits / (model.get('source_bits') or 16)
    # Conservative planning heuristics, not measured architecture-specific peaks.
    # Quantization budgets source weights and conversion copies, not inference fit.
    training_estimate = {
        'lora': 2 * source + shard + 2 * GB,
        'qlora': math.ceil(1.5 * source) + shard + 2 * GB,
        'full': 6 * source + shard + 2 * GB,
    }.get(fine_tune_method, 2 * source + shard + 2 * GB)
    estimates = {'quantize': source + shard + output + 0.5 * GB,
                 'prune': 2 * source + shard + GB,
                 'retrain': training_estimate,
                 'load': source + shard + 0.5 * GB,
                 'evaluate': source + shard + GB}
    selected = {op: math.ceil(estimates[op]) for op in operations}
    peak = max(selected.values(), default=0)
    insufficient = peak > available
    required_gb = math.ceil(peak / GB * 10) / 10
    message = 'Available memory is not enough for optimization. Choose a smaller model or pre-quantized version.' if insufficient else None
    return {'available_bytes': available, 'required_bytes': peak, 'required_gb': required_gb, 'output_weight_bytes_estimate': math.ceil(output), 'by_operation': selected, 'insufficient_memory': insufficient, 'message': message, 'estimate_only': True, 'basis': 'Conservative source-weight, largest-shard, output-copy and workspace estimates; actual peaks depend on batch size, sequence length, architecture and engine.'}


def prepare_download_destination(model, directory=None):
    root = Path(directory).expanduser().resolve(strict=True) if directory else ROOT / 'models'
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError('Select a download destination folder.')
    if not os.access(root, os.W_OK):
        raise PermissionError('The selected download destination is not writable.')
    if directory:
        name = re.sub(r'[^a-zA-Z0-9._-]+', '-', model.get('repo_id', '').split('/')[-1]).strip('.-') or 'model'
        destination = root / f'{name}-{model["id"][:8]}'
    else:
        destination = root / model['id']
    destination = destination.resolve()
    if destination.parent != root.resolve():
        raise ValueError('Invalid model download destination.')
    marker = destination / '.forge-download'
    if destination.exists() and any(destination.iterdir()) and (not marker.is_file() or marker.read_text().strip() != model['id']):
        raise ValueError('The model download folder already exists and is not managed by Forge.')
    destination.mkdir(parents=False, exist_ok=True)
    marker.write_text(model['id'])
    return str(destination)


def discard_partial_download(model):
    value = model.get('download_path')
    destination = Path(value).expanduser().resolve() if value else (ROOT / 'models' / model['id']).resolve()
    managed_root = (ROOT / 'models').resolve()
    if not value and destination.parent != managed_root:
        raise ValueError('Invalid model download path.')
    marker = destination / '.forge-download'
    managed_custom = marker.is_file() and marker.read_text().strip() == model['id']
    if destination.is_dir() and (not value or managed_custom):
        shutil.rmtree(destination, ignore_errors=True)


def download_progress(model, complete=False):
    total = max(0, int(model.get('download_bytes') or 0))
    destination_value = model.get('download_path') or model.get('path')
    downloaded = total if complete or model.get('downloaded') else 0
    if destination_value and not downloaded:
        destination = Path(destination_value).expanduser()
        completed = 0
        for file in model.get('files', []):
            target = destination / file['name']
            try:
                completed += min(target.stat().st_size, max(0, int(file.get('size') or 0)))
            except OSError:
                continue
        partial = 0
        cache = destination / '.cache' / 'huggingface' / 'download'
        if cache.is_dir():
            for target in cache.rglob('*.incomplete'):
                try:
                    partial += target.stat().st_size
                except OSError:
                    continue
        downloaded = min(total, completed + partial)
    percent = round(downloaded / total * 100) if total else 0
    if not complete and not model.get('downloaded'):
        percent = min(percent, 99)
    return {'downloaded_bytes': downloaded, 'total_bytes': total, 'progress_percent': percent}


def download(model, destination=None):
    if not model['compatible']:
        raise ValueError(model['compatibility_error'])
    destination = Path(destination or model.get('download_path') or ROOT / 'models' / model['id']).expanduser().resolve()
    if not destination.parent.is_dir():
        raise ValueError('The selected download destination no longer exists.')
    marker = destination / '.forge-download'
    if not marker.is_file() or marker.read_text().strip() != model['id']:
        raise ValueError('The model download destination is invalid.')
    if model['download_bytes'] + GB > shutil.disk_usage(destination.parent).free:
        raise ValueError('Not enough free disk space to download this model.')
    snapshot_download(model['repo_id'], revision=model['revision'], local_dir=destination, allow_patterns=[f['name'] for f in model['files']], max_workers=2)
    local = inspect_local(str(destination))
    if not local['compatible']:
        raise ValueError(local['compatibility_error'])
    result = {**model, 'path': str(destination), 'downloaded': True}
    result.pop('id', None)
    result.pop('download_path', None)
    marker.unlink(missing_ok=True)
    return save('model', result, model['id'])
