import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import models, operations, storage
from backend.app import SaveOutputRequest, create_app, prepare_optimization_output
from backend.datasets import parse_dataset, score


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(models, 'ROOT', tmp_path)
    with TestClient(create_app(start_worker=False)) as client:
        yield client


def examples(count=20):
    return b'\n'.join(json.dumps({'prompt': f'What is {i} + 1?', 'response': str(i+1)}).encode() for i in range(count))


def test_dataset_splits_are_reproducible_disjoint_and_deduplicated():
    data = examples() + b'\n' + examples(1)
    splits, duplicates = parse_dataset(data)
    assert duplicates == 1
    assert splits == parse_dataset(data)[0]
    prompt_sets = [{json.dumps(r['messages'][:-1]) for r in rows} for rows in splits.values()]
    assert sum(map(len, prompt_sets)) == 20
    assert not prompt_sets[0] & prompt_sets[1]
    assert not prompt_sets[0] & prompt_sets[2]
    assert not prompt_sets[1] & prompt_sets[2]


def test_dataset_uses_mlx_validation_filename(tmp_path, monkeypatch):
    from backend import datasets
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(datasets, 'ROOT', tmp_path)
    storage.initialize()
    result = datasets.import_dataset(examples(), 'examples.jsonl')
    assert (Path(result['path']) / 'valid.jsonl').is_file()
    assert not (Path(result['path']) / 'validation.jsonl').exists()
    training_row = json.loads((Path(result['path']) / 'train.jsonl').read_text().splitlines()[0])
    assert training_row == {'text': training_row['text']}
    assert 'user:' in training_row['text'] and 'assistant:' in training_row['text']
    assert json.loads((Path(result['path']) / 'evaluation.jsonl').read_text().splitlines()[0])['messages']


def test_conflicting_duplicate_and_invalid_chat_are_rejected():
    with pytest.raises(ValueError, match='Conflicting'):
        parse_dataset(examples() + b'\n' + b'{"prompt":"What is 0 + 1?","response":"wrong"}')
    with pytest.raises(ValueError, match='alternate'):
        parse_dataset(b'{"messages":[{"role":"user","content":"Hi"},{"role":"user","content":"Bye"}]}')
    with pytest.raises(ValueError, match='five unique'):
        parse_dataset(examples(4))


def test_metrics_use_all_references_without_substring_false_positives():
    assert score('Paris!', ['London', 'paris'])['exact_match'] == 1
    assert score('Not Paris', ['Paris'])['exact_match'] == 0
    assert score('red blue', ['red green'])['token_f1'] == .5


def test_repo_links_are_validated():
    assert models.repository_id('https://huggingface.co/org/model') == 'org/model'
    for value in ['https://evil.example/org/model', 'http://huggingface.co/org/model', '../secret', 'https://huggingface.co/org/model/blob/main/file', 'unsloth/model-Q4_0.gguf']:
        with pytest.raises(ValueError):
            models.repository_id(value)


def test_gguf_repository_reports_the_actual_format_problem(monkeypatch):
    monkeypatch.setattr(models, 'supported', lambda config: (True, None))
    result = models.metadata({'model_type': 'qwen'}, [{'name': 'config.json', 'size': 100}], 'test', format_names=['model.gguf'])
    assert not result['compatible']
    assert result['model_format'] == 'gguf'
    assert result['compatibility_error'] == 'GGUF optimization is not supported. Choose a supported Safetensors model.'


@pytest.mark.parametrize(('names', 'config', 'expected'), [
    (['model.safetensors'], {}, 'safetensors'),
    (['model.safetensors'], {'quantization': {'bits': 4}}, 'mlx'),
    (['model.safetensors'], {'quantization_config': {'quant_method': 'mlx'}}, 'mlx'),
    (['pytorch_model.bin'], {}, 'pytorch'),
    (['model.pt'], {}, 'pytorch'),
    (['model.pth'], {}, 'pytorch'),
    (['model.gguf'], {}, 'gguf'),
    (['model.onnx'], {}, 'onnx'),
    (['saved_model.pb'], {}, 'tensorflow'),
    (['model.engine'], {}, 'tensorrt'),
    (['model.plan'], {}, 'tensorrt'),
])
def test_model_format_detection(names, config, expected):
    assert models.detect_model_format(config, names)[0] == expected


def test_windows_cuda_detection_does_not_depend_on_gpu_name_probe(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        xpu=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setattr(
        models.importlib.util,
        'find_spec',
        lambda name: object() if name in {'torch', 'transformers'} else None,
    )
    assert models._windows_engine(None) == ('torch-cuda', 'PyTorch CUDA available')


def test_optimization_plan_checks_engine_dataset_and_method(monkeypatch, tmp_path):
    model = {'downloaded': True, 'path': str(tmp_path), 'compatible': True, 'model_format': 'safetensors', 'source_bits': 16}
    monkeypatch.setattr(operations, 'hardware', lambda: {'engine_id': 'mlx'})
    operations.validate_plan(model, {'quantize': True, 'bits': 4})
    with pytest.raises(ValueError, match='dataset'):
        operations.validate_plan(model, {'fine_tune': True})
    with pytest.raises(ValueError, match='calibration dataset'):
        operations.validate_plan(model, {'prune': True, 'prune_method': 'wanda'})
    for method in ['magnitude', 'wanda', 'sparsegpt', 'structured', 'nm']:
        operations.validate_plan(model, {'prune': True, 'prune_method': method, 'dataset_id': 'dataset'})
    with pytest.raises(ValueError, match='supported pruning'):
        operations.validate_plan(model, {'prune': True, 'prune_method': 'unknown'})


def test_fine_tuning_methods_validate_platform_and_precision(monkeypatch, tmp_path):
    model = {'downloaded': True, 'path': str(tmp_path), 'compatible': True, 'model_format': 'safetensors', 'source_bits': 16}
    monkeypatch.setattr(operations, 'hardware', lambda: {'engine_id': 'mlx'})
    for method in ['lora', 'qlora', 'full']:
        operations.validate_plan(model, {'fine_tune': True, 'fine_tune_method': method, 'dataset_id': 'dataset'})
    with pytest.raises(ValueError, match='LoRA, QLoRA'):
        operations.validate_plan(model, {'fine_tune': True, 'fine_tune_method': 'unknown', 'dataset_id': 'dataset'})
    monkeypatch.setattr(operations, 'hardware', lambda: {'engine_id': 'torch-cpu'})
    with pytest.raises(ValueError, match='CUDA or ROCm'):
        operations.validate_plan(model, {'fine_tune': True, 'fine_tune_method': 'qlora', 'dataset_id': 'dataset'})
    four_bit = {**model, 'source_bits': 4}
    with pytest.raises(ValueError, match='16-bit, 24-bit, or 32-bit'):
        operations.validate_plan(four_bit, {'fine_tune': True, 'fine_tune_method': 'full', 'dataset_id': 'dataset'})


@pytest.mark.parametrize('source_bits', [4, 8, 16, 24, 32])
@pytest.mark.parametrize('target_bits', [4, 8, 16, 24, 32])
def test_every_precision_pair_is_validated(monkeypatch, tmp_path, source_bits, target_bits):
    monkeypatch.setattr(operations, 'hardware', lambda: {'engine_id': 'mlx'})
    model = {
        'downloaded': True,
        'path': str(tmp_path),
        'compatible': True,
        'model_format': 'safetensors',
        'source_bits': source_bits,
    }
    payload = {'quantize': True, 'bits': target_bits}
    if source_bits == target_bits:
        with pytest.raises(ValueError, match='different'):
            operations.validate_plan(model, payload)
    else:
        operations.validate_plan(model, payload)


@pytest.mark.parametrize('source_bits', [4, 8, 16, 24, 32])
def test_pruning_precision_eligibility(monkeypatch, tmp_path, source_bits):
    monkeypatch.setattr(operations, 'hardware', lambda: {'engine_id': 'mlx'})
    model = {
        'downloaded': True,
        'path': str(tmp_path),
        'compatible': True,
        'model_format': 'safetensors',
        'source_bits': source_bits,
    }
    payload = {'prune': True, 'prune_method': 'magnitude'}
    if source_bits in (16, 24, 32):
        operations.validate_plan(model, payload)
    else:
        with pytest.raises(ValueError, match='16-bit, 24-bit, or 32-bit'):
            operations.validate_plan(model, payload)


def test_magnitude_pruning_writes_real_zeroed_weights(tmp_path):
    import mlx.core as mx
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    (source / 'config.json').write_text('{"model_type":"llama","torch_dtype":"float16"}')
    (source / 'tokenizer.json').write_text('{}')
    mx.save_safetensors(str(source / 'model.safetensors'), {'weight': mx.array([[1.0, 2.0], [3.0, 4.0]])})
    operations.magnitude_prune(source, output, 0.5)
    result = mx.load(str(output / 'model.safetensors'))['weight'].tolist()
    assert result == [[0.0, 0.0], [3.0, 4.0]]
    assert json.loads((output / 'config.json').read_text())['forge_pruning']['target_ratio'] == .5


@pytest.mark.parametrize('pruner', [operations.prune_model])
def test_pruning_rejects_targets_above_eighty_percent(tmp_path, pruner):
    with pytest.raises(ValueError, match='between 1 and 80 percent'):
        pruner(tmp_path / 'source', tmp_path / 'output', .81)


def test_torch_pruning_rejects_targets_above_eighty_percent(tmp_path):
    from backend import torch_backend

    with pytest.raises(ValueError, match='between 1 and 80 percent'):
        torch_backend.prune_model(
            tmp_path / 'source',
            tmp_path / 'output',
            .81,
            'magnitude',
            None,
            'torch-cpu',
        )


@pytest.mark.parametrize('method', ['structured', 'nm'])
def test_weight_only_pruning_methods_write_valid_sparse_weights(tmp_path, method):
    import mlx.core as mx
    source = tmp_path / 'source'
    output = tmp_path / method
    source.mkdir()
    (source / 'config.json').write_text('{"model_type":"llama","torch_dtype":"float16"}')
    (source / 'tokenizer.json').write_text('{}')
    mx.save_safetensors(str(source / 'model.safetensors'), {'weight': mx.arange(1, 17).reshape(4, 4).astype(mx.float32)})
    operations.prune_model(source, output, .5, method)
    result = mx.load(str(output / 'model.safetensors'))['weight']
    values = result.tolist()
    assert sum(value == 0 for row in values for value in row) == 8
    metadata = json.loads((output / 'config.json').read_text())['forge_pruning']
    assert metadata['method'] == method
    if method == 'nm':
        assert metadata['pattern'] == '2:4'


def test_wanda_uses_calibration_activations(tmp_path, monkeypatch):
    import mlx.core as mx
    source = tmp_path / 'source'
    output = tmp_path / 'wanda'
    source.mkdir()
    (source / 'config.json').write_text('{"model_type":"llama","torch_dtype":"float16"}')
    (source / 'tokenizer.json').write_text('{}')
    mx.save_safetensors(str(source / 'model.safetensors'), {'weight': mx.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])})
    monkeypatch.setattr(operations, '_calibration_inputs', lambda *args, **kwargs: {'weight': operations.np.array([[1., 10., 1., 10.]])})
    operations.prune_model(source, output, .5, 'wanda', 'dataset')
    values = mx.load(str(output / 'model.safetensors'))['weight'].tolist()
    assert sum(value == 0 for row in values for value in row) == 4
    assert values[0][0] == 0 and values[0][1] != 0


def test_sparsegpt_uses_blockwise_hessian_and_preserves_shape():
    import mlx.core as mx
    weight = mx.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    inputs = operations.np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]])
    result = operations._sparsegpt(weight, inputs, .5, block_size=4)
    assert result.shape == weight.shape
    assert sum(value == 0 for row in result.tolist() for value in row) == 4


def test_memory_is_operation_specific_and_uses_runtime_values(monkeypatch):
    monkeypatch.setattr(models, 'hardware', lambda: {'ram_available_bytes': 4_000_000_000})
    model = {'weight_bytes': 3_000_000_000, 'largest_shard_bytes': 1_000_000_000, 'parameter_count': 1_500_000_000, 'source_bits': 16}
    quant = models.estimate(model, ['quantize'], 4)
    training = models.estimate(model, ['retrain'], 4)
    qlora = models.estimate(model, ['retrain'], 4, fine_tune_method='qlora')
    full = models.estimate(model, ['retrain'], 4, fine_tune_method='full')
    assert quant['required_bytes'] < training['required_bytes']
    assert qlora['required_bytes'] < training['required_bytes'] < full['required_bytes']
    assert quant['insufficient_memory']
    assert quant['message'] == 'Available memory is not enough for optimization. Choose a smaller model or pre-quantized version.'
    assert models.estimate(model, ['quantize'], 4, available_bytes=40_000_000_000)['message'] is None
    assert models.estimate(model, [], 4)['message'] is None


def test_windows_hardware_detects_gpu_engine_and_live_memory(monkeypatch):
    monkeypatch.setattr(models.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(models.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(models.platform, 'processor', lambda: 'Intel64 Family 6')
    monkeypatch.setattr(models.psutil, 'virtual_memory', lambda: type('Memory', (), {'total': 16_000_000_000, 'available': 9_500_000_000})())
    monkeypatch.setattr(models, '_run', lambda command: '' if command[0] == 'nvidia-smi' else '{"Name":"AMD Radeon 780M","AdapterRAM":2147483648}')
    monkeypatch.setattr(models.importlib.util, 'find_spec', lambda name: object() if name == 'torch_directml' else None)
    result = models.hardware()
    assert result['os'] == 'Windows'
    assert result['gpu'] == 'AMD Radeon 780M'
    assert result['gpu_memory_bytes'] == 2147483648
    assert result['engine_id'] == 'torch-directml'
    assert result['engine'] == 'PyTorch DirectML available'
    assert result['ram_total_bytes'] == 16_000_000_000
    assert result['ram_available_bytes'] == 9_500_000_000


def test_intel_mac_uses_pytorch_backend(monkeypatch):
    monkeypatch.setattr(models.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(models.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(models.platform, 'processor', lambda: 'Intel Core i7')
    monkeypatch.setattr(models, '_linux_engine', lambda vendor: ('torch-cpu', 'PyTorch CPU available'))
    result = models.hardware()
    assert result['os'] == 'Darwin'
    assert result['engine_id'] == 'torch-cpu'
    assert result['graphics_requirement'] == 'PyTorch-compatible CPU'


def test_local_inspection_uses_original_path_and_rejects_missing_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path / 'db')
    monkeypatch.setattr(models, 'supported', lambda config: (True, None))
    storage.initialize()
    folder = tmp_path / 'original'
    folder.mkdir()
    (folder / 'config.json').write_text('{"model_type":"llama","torch_dtype":"float16"}')
    (folder / 'tokenizer.json').write_text('{}')
    import struct
    header = json.dumps({'weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    (folder / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0\0')
    result = models.inspect_local(str(folder))
    assert result['path'] == str(folder.resolve())
    assert list(folder.glob('*.safetensors')) == [folder / 'model.safetensors']
    (folder / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"missing.safetensors"}}')
    with pytest.raises(ValueError, match='missing'):
        models.inspect_local(str(folder))


def test_backend_origin_protection(client):
    assert client.get('/health').status_code == 200
    assert client.get('/health', headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/models/inspect', headers={'Origin': 'null'}, json={}).status_code == 403
    assert client.get('/health', headers={'Host': 'evil.example'}).status_code == 400


def test_system_folder_picker_returns_selection_or_cancel(client, monkeypatch, tmp_path):
    app_module = importlib.import_module('backend.app')
    monkeypatch.setattr(app_module, 'pick_folder', lambda: str(tmp_path))
    assert client.post('/folders/pick').json() == {'path': str(tmp_path)}
    monkeypatch.setattr(app_module, 'pick_folder', lambda: None)
    assert client.post('/folders/pick').json() == {'path': None}


def test_upload_and_invalid_requests(client):
    response = client.post('/datasets', files={'file': ('math.jsonl', examples(), 'application/jsonl')})
    assert response.status_code == 201
    assert response.json()['counts'] == {'test': 2, 'validation': 2, 'train': 16}
    assert client.post('/models/inspect', json={'source': 'local', 'value': '', 'bits': 18}).status_code == 422
    assert client.post('/models/not-found/load').status_code == 422
    assert client.get('/datasets').json()[0]['sha256']


def test_single_worker_queue_and_restart_recovery(client):
    job_id = storage.create_job('load', {'model_id': 'test'})
    with pytest.raises(RuntimeError, match='already running'):
        storage.create_job('download', {'model_id': 'another'})
    assert storage.next_job()['id'] == job_id
    storage.recover_jobs()
    assert storage.job(job_id)['status'] == 'interrupted'
    assert storage.create_job('load', {'model_id': 'new'}) != job_id


def test_active_job_can_be_cancelled(client):
    job_id = storage.create_job('download', {'model_id': 'test'})
    assert client.get('/jobs').json()['id'] == job_id
    storage.update_job(job_id, 'running', 'Downloading model files')
    response = client.post(f'/jobs/{job_id}/cancel')
    assert response.status_code == 200
    assert response.json()['status'] == 'cancelled'
    assert storage.active_job() is None
    assert client.get('/jobs').json() is None


def test_optimization_can_be_paused_resumed_and_cancelled(client, tmp_path):
    job_id = storage.create_job('optimize', {'model_id': 'test'})
    storage.update_job(job_id, 'running', 'Quantizing weights')
    paused = client.post(f'/jobs/{job_id}/pause')
    assert paused.status_code == 200
    assert paused.json()['status'] == 'paused'
    assert storage.active_job()['id'] == job_id
    with pytest.raises(RuntimeError, match='already running'):
        storage.create_job('optimize', {'model_id': 'another'})
    resumed = client.post(f'/jobs/{job_id}/resume')
    assert resumed.status_code == 200
    assert resumed.json()['status'] == 'running'
    assert client.post(f'/jobs/{job_id}/pause').json()['status'] == 'paused'
    assert client.post(f'/jobs/{job_id}/cancel').json()['status'] == 'cancelled'
    assert storage.active_job() is None


def test_queued_optimization_can_pause_without_starting(client):
    job_id = storage.create_job('optimize', {'model_id': 'test'})
    assert client.post(f'/jobs/{job_id}/pause').json()['status'] == 'paused'
    resumed = client.post(f'/jobs/{job_id}/resume').json()
    assert resumed['status'] == 'queued'
    assert storage.next_job()['id'] == job_id


def test_cancelling_optimization_removes_only_its_partial_output(client, tmp_path):
    job_id = storage.create_job('optimize', {'model_id': 'test'})
    partial = tmp_path / 'outputs' / job_id
    partial.mkdir(parents=True)
    (partial / 'partial.safetensors').write_bytes(b'partial')
    sibling = tmp_path / 'outputs' / 'completed'
    sibling.mkdir()
    storage.update_job(job_id, 'running', 'Pruning weights')
    assert client.post(f'/jobs/{job_id}/cancel').json()['status'] == 'cancelled'
    assert not partial.exists()
    assert sibling.exists()


def test_forced_cancel_wins_worker_shutdown_race(client):
    job_id = storage.create_job('optimize', {'model_id': 'test'})
    storage.update_job(job_id, 'failed', 'Failed', error='worker stopped')
    assert storage.cancel_job(job_id, force=True)['status'] == 'cancelled'
    storage.update_job(job_id, 'failed', 'Failed', error='late worker error')
    assert storage.job(job_id)['status'] == 'cancelled'


def test_partial_download_can_be_discarded_but_completed_model_is_protected(client, tmp_path):
    partial = tmp_path / 'models' / 'partial-model'
    partial.mkdir(parents=True)
    (partial / 'incomplete.safetensors').write_bytes(b'partial')
    storage.save('model', {'name': 'partial', 'downloaded': False}, 'partial-model')
    response = client.post('/models/partial-model/download/discard')
    assert response.status_code == 200
    assert response.json() == {'status': 'discarded'}
    assert not partial.exists()
    storage.save('model', {'name': 'complete', 'downloaded': True}, 'complete-model')
    assert client.post('/models/complete-model/download/discard').status_code == 422


def test_download_job_reports_measured_byte_progress(client, tmp_path):
    destination = tmp_path / 'download'
    destination.mkdir()
    (destination / 'config.json').write_bytes(b'x' * 100)
    cache = destination / '.cache' / 'huggingface' / 'download'
    cache.mkdir(parents=True)
    (cache / 'weights.incomplete').write_bytes(b'x' * 250)
    storage.save('model', {
        'name': 'partial', 'downloaded': False, 'download_path': str(destination),
        'download_bytes': 1000, 'files': [
            {'name': 'config.json', 'size': 100},
            {'name': 'model.safetensors', 'size': 900},
        ],
    }, 'progress-model')
    job_id = storage.create_job('download', {'model_id': 'progress-model', 'destination': str(destination)})
    storage.update_job(job_id, 'running', 'Downloading model files')
    result = client.get(f'/jobs/{job_id}').json()
    assert result['downloaded_bytes'] == 350
    assert result['total_bytes'] == 1000
    assert result['progress_percent'] == 35


def test_custom_download_destination_is_dedicated_and_safely_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'ROOT', tmp_path / 'forge')
    destination_root = tmp_path / 'downloads'
    destination_root.mkdir()
    model = {'id': 'abcdef123456', 'repo_id': 'owner/My Model'}
    destination = Path(models.prepare_download_destination(model, str(destination_root)))
    assert destination.parent == destination_root
    assert destination.name == 'My-Model-abcdef12'
    assert (destination / '.forge-download').read_text() == model['id']
    models.discard_partial_download({**model, 'download_path': str(destination)})
    assert not destination.exists()


def test_custom_download_destination_never_deletes_unmanaged_folder(tmp_path):
    destination = tmp_path / 'existing'
    destination.mkdir()
    (destination / 'user-file.txt').write_text('keep')
    models.discard_partial_download({'id': 'model-id', 'download_path': str(destination)})
    assert (destination / 'user-file.txt').read_text() == 'keep'


def test_optimization_output_uses_fixed_unnumbered_folders_and_safe_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path / 'forge')
    storage.initialize()
    parent = tmp_path / 'results'
    parent.mkdir()
    first = Path(prepare_optimization_output(str(parent), 'Optimized'))
    second = Path(prepare_optimization_output(str(parent), 'FineTuned'))
    assert first.name == 'Optimized'
    assert second.name == 'FineTuned'
    assert (first / '.forge-operation').is_file()
    with pytest.raises(ValueError, match='already exists'):
        prepare_optimization_output(str(parent), 'Optimized')
    storage.create_job('optimize', {'output_path': str(first)})
    job_id = storage.next_job()['id']
    storage.discard_job_output(job_id)
    assert not first.exists()
    assert second.exists()


def test_save_output_folder_name_validation():
    assert SaveOutputRequest(destination='/tmp', folder_name=' My Run ').folder_name == 'My Run'
    with pytest.raises(ValueError):
        SaveOutputRequest(destination='/tmp', folder_name='../My Run')


def test_optimization_output_can_be_saved_inside_the_model_folder(tmp_path):
    source = tmp_path / 'model'
    source.mkdir()
    destination = Path(
        prepare_optimization_output(str(source), 'Optimized')
    )
    assert destination.parent == source
    assert destination.name == 'Optimized'


def test_copy_model_skips_an_output_folder_nested_inside_the_source(tmp_path):
    source = tmp_path / 'model'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    run_directory = source / 'Tiny-Model-optimized'
    run_directory.mkdir()
    destination = run_directory / 'converted'
    operations._copy_model(source, destination)
    assert (destination / 'config.json').read_text() == '{}'
    assert not (destination / run_directory.name).exists()


def test_completed_output_is_saved_only_after_destination_is_selected(client, tmp_path, monkeypatch):
    app_module = importlib.import_module('backend.app')
    source_model = tmp_path / 'source-model'
    source_model.mkdir()
    storage.save('model', {'name': 'Tiny Model', 'path': str(source_model)}, 'source-model')
    job_id = storage.create_job('optimize', {'model_id': 'source-model'})
    run_root = tmp_path / 'outputs' / job_id
    output = run_root / '8-bit'
    output.mkdir(parents=True)
    (output / 'config.json').write_text('{}')
    (output / 'model.safetensors').write_bytes(b'weights')
    result = {
        'model': {'id': 'temporary'},
        'output_path': str(output),
        'adapter_path': None,
        'operations': ['quantize'],
        'saved': False,
    }
    storage.update_job(job_id, 'succeeded', 'Complete', result)
    destination_root = tmp_path / 'saved'
    destination_root.mkdir()
    monkeypatch.setattr(
        app_module,
        'inspect_local',
        lambda value: {'id': 'saved-model', 'path': value},
    )
    response = client.post(
        f'/jobs/{job_id}/save-output',
        json={'destination': str(destination_root), 'folder_name': 'My Optimized Model'},
    )
    assert response.status_code == 200
    saved = response.json()
    assert saved['saved'] is True
    assert Path(saved['output_path']).parent == destination_root
    assert Path(saved['output_path']).name == 'My Optimized Model'
    assert (Path(saved['output_path']) / 'model.safetensors').read_bytes() == b'weights'
    assert not run_root.exists()


def test_failed_cleanup_does_not_crash_job_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    storage.initialize()
    output = tmp_path / 'outputs' / 'job-id'
    output.mkdir(parents=True)
    monkeypatch.setattr(
        storage.shutil,
        'rmtree',
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError('busy')),
    )
    storage.discard_job_output('job-id')


def test_restarted_job_replaces_its_stale_internal_output(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    run_directory = tmp_path / 'outputs' / 'job-id'
    run_directory.mkdir(parents=True)
    (run_directory / 'stale').write_text('old')
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(operations, 'validate_plan', lambda *args: None)
    monkeypatch.setattr(operations, 'hardware', lambda: {'engine_id': 'mlx'})
    monkeypatch.setattr(
        operations,
        'inspect_local',
        lambda path: {'id': 'result', 'path': path},
    )
    result = operations.run_optimization(
        'job-id',
        {'path': str(source)},
        {'prune': False, 'fine_tune': False, 'quantize': False},
    )
    assert not (run_directory / 'stale').exists()
    assert result['output_path'] == str(source)


def test_optimization_endpoint_validates_then_queues_real_job(client, monkeypatch, tmp_path):
    model = storage.save('model', {
        'name': 'tiny', 'downloaded': True, 'path': str(tmp_path), 'compatible': True,
        'model_format': 'safetensors', 'source_bits': 16, 'weight_bytes': 1000,
        'largest_shard_bytes': 1000, 'parameter_count': 500,
    }, 'tiny-model')
    available = {'engine_id': 'mlx', 'ram_available_bytes': 10_000_000_000}
    monkeypatch.setattr(operations, 'hardware', lambda: available)
    monkeypatch.setattr(models, 'hardware', lambda: available)
    response = client.post('/optimizations', json={
        'model_id': model['id'], 'fine_tune': False, 'prune': False,
        'quantize': True, 'bits': 4,
    })
    assert response.status_code == 202
    assert response.json()['kind'] == 'optimize'
    assert storage.next_job()['payload']['bits'] == 4


def test_every_fine_tuning_method_reaches_the_api(client, monkeypatch, tmp_path):
    model = storage.save('model', {
        'name': 'tiny', 'downloaded': True, 'path': str(tmp_path), 'compatible': True,
        'model_format': 'safetensors', 'source_bits': 16, 'weight_bytes': 1000,
        'largest_shard_bytes': 1000, 'parameter_count': 500,
    }, 'fine-tuning-model')
    dataset = storage.save('dataset', {'name': 'training', 'path': str(tmp_path)}, 'training')
    available = {'engine_id': 'mlx', 'ram_available_bytes': 100_000_000_000}
    monkeypatch.setattr(operations, 'hardware', lambda: available)
    monkeypatch.setattr(models, 'hardware', lambda: available)
    for method in ['lora', 'qlora', 'full']:
        response = client.post('/optimizations', json={
            'model_id': model['id'], 'dataset_id': dataset['id'],
            'fine_tune': True, 'fine_tune_method': method, 'iterations': 1,
        })
        assert response.status_code == 202, (method, response.text)
        job = storage.next_job()
        assert job['payload']['fine_tune_method'] == method
        storage.cancel_job(job['id'])
    assert client.post('/optimizations', json={
        'model_id': model['id'], 'dataset_id': dataset['id'],
        'fine_tune': True, 'fine_tune_method': 'invalid',
    }).status_code == 422


def test_every_pruning_method_and_numeric_boundary_reaches_the_api(client, monkeypatch, tmp_path):
    model = storage.save('model', {
        'name': 'tiny', 'downloaded': True, 'path': str(tmp_path), 'compatible': True,
        'model_format': 'safetensors', 'source_bits': 16, 'weight_bytes': 1000,
        'largest_shard_bytes': 1000, 'parameter_count': 500,
    }, 'pruning-model')
    dataset = storage.save('dataset', {'name': 'calibration', 'path': str(tmp_path)}, 'calibration')
    available = {'engine_id': 'mlx', 'ram_available_bytes': 10_000_000_000}
    monkeypatch.setattr(operations, 'hardware', lambda: available)
    monkeypatch.setattr(models, 'hardware', lambda: available)
    for method in ['magnitude', 'wanda', 'sparsegpt', 'structured', 'nm']:
        response = client.post('/optimizations', json={
            'model_id': model['id'], 'dataset_id': dataset['id'], 'prune': True,
            'prune_method': method, 'prune_ratio': .1,
        })
        assert response.status_code == 202, (method, response.text)
        storage.cancel_job(response.json()['id'])
    base = {'model_id': model['id'], 'prune': True, 'prune_method': 'magnitude'}
    for field, value in [('prune_ratio', 0), ('prune_ratio', .81), ('iterations', 0), ('iterations', 5001)]:
        assert client.post('/optimizations', json={**base, field: value}).status_code == 422
    assert client.post('/optimizations', json={**base, 'prune_method': 'unknown'}).status_code == 422
