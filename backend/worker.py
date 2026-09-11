"""One persistent, separate process owns model memory and all heavy jobs."""
import gc
import json
import os
import time
from pathlib import Path

from . import storage
from .datasets import score
from .models import download, estimate, hardware, inspect_local
from .operations import run_optimization


def run():
    storage.initialize()
    pid_file = storage.ROOT / 'worker.pid'
    pid_file.write_text(str(os.getpid()))
    model = tokenizer = device = None
    loaded_id = None
    try:
        while True:
            job = storage.next_job()
            if not job:
                time.sleep(.25)
                continue
            job_id, payload = job['id'], job['payload']
            try:
                storage.update_job(job_id, 'running', 'Starting')
                info = storage.get(payload['model_id'], 'model')
                if job['kind'] == 'download':
                    storage.update_job(job_id, 'running', 'Downloading model files')
                    result = download(info, payload.get('destination'))
                elif job['kind'] == 'optimize':
                    result = run_optimization(job_id, info, payload)
                else:
                    if not info.get('path'):
                        raise ValueError('Download the model before loading or evaluating it.')
                    current = inspect_local(info['path'])
                    if not current['compatible']:
                        raise ValueError(current['compatibility_error'])
                    if loaded_id != info['id']:
                        model = tokenizer = device = None
                        loaded_id = None
                        gc.collect()
                        engine_id = hardware()['engine_id']
                        if engine_id == 'mlx':
                            import mlx.core as mx
                            mx.clear_cache()
                        else:
                            from .torch_backend import clear_cache
                            clear_cache(engine_id)
                        budget = estimate(current, ['evaluate' if job['kind'] == 'baseline' else 'load'])
                        if budget['insufficient_memory']:
                            raise ValueError(f"Loading this model requires an estimated {budget['required_gb']:g} GB of available memory.")
                        storage.update_job(job_id, 'running', 'Loading model into memory')
                        if engine_id == 'mlx':
                            from mlx_lm import load
                            model, tokenizer = load(info['path'], tokenizer_config={'trust_remote_code': False})
                        else:
                            from .torch_backend import load_for_inference
                            model, tokenizer, device = load_for_inference(info['path'], engine_id)
                        loaded_id = info['id']
                    if job['kind'] == 'load':
                        result = {'model_id': loaded_id, 'loaded': True, 'engine': engine_id, 'path': info['path']}
                    elif job['kind'] == 'baseline':
                        result = evaluate(job_id, model, tokenizer, info, payload, engine_id, device)
                    else:
                        raise ValueError('Unsupported job type.')
                storage.update_job(job_id, 'succeeded', 'Complete', result)
            # Model libraries and filesystem calls can raise different exceptions.
            # Convert every ordinary job failure into a terminal result.
            except Exception as error:  # noqa: BLE001
                model = tokenizer = device = None
                loaded_id = None
                gc.collect()
                if storage.job(job_id)['status'] == 'cancelled':
                    continue
                if job['kind'] == 'optimize':
                    storage.discard_job_output(job_id)
                storage.update_job(job_id, 'failed', 'Failed', error=str(error)[:1500])
    finally:
        try:
            if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass


def evaluate(job_id, model, tokenizer, info, payload, engine_id='mlx', device=None):
    dataset = storage.get(payload['dataset_id'], 'dataset')
    records = [json.loads(line) for line in (Path(dataset['path']) / 'evaluation.jsonl').read_text().splitlines()][:payload.get('max_samples', 20)]
    results = []
    started = time.monotonic()
    for i, record in enumerate(records):
        messages = record['messages'][:-1]
        if getattr(tokenizer, 'chat_template', None):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = '\n'.join(m['content'] for m in messages)
        context_length = len(tokenizer.encode(prompt))
        if context_length > 2048:
            raise ValueError('Baseline prompts must be at most 2,048 tokens; shorten this dataset example.')
        before = time.monotonic()
        if engine_id == 'mlx':
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            response = generate(model, tokenizer, prompt=prompt, max_tokens=payload.get('max_tokens', 128), sampler=make_sampler(temp=0), verbose=False)
        else:
            from .torch_backend import generate
            response = generate(model, tokenizer, device, prompt, payload.get('max_tokens', 128))
        elapsed = time.monotonic() - before
        results.append({'prompt_messages': messages, 'reference_answers': record['answers'], 'prediction': response, 'category': record['category'], 'seconds': elapsed, 'generated_tokens': len(tokenizer.encode(response)), **score(response, record['answers'])})
        storage.update_job(job_id, 'running', f'Evaluated {i + 1}/{len(records)}')
    def average(rows, metric):
        return sum(r[metric] for r in rows) / len(rows)
    report = {'model_id': info['id'], 'model_revision': info.get('revision'), 'dataset_id': dataset['id'], 'dataset_sha256': dataset['sha256'], 'split': 'test', 'samples': len(results), 'max_tokens': payload.get('max_tokens', 128), 'temperature': 0, 'seconds': time.monotonic() - started, 'metrics': {m: average(results, m) for m in ['exact_match', 'token_f1']}, 'by_category': {c: {m: average([r for r in results if r['category'] == c], m) for m in ['exact_match', 'token_f1']} for c in sorted({r['category'] for r in results})}, 'limitations': 'Reference exact match and token overlap only. These do not establish factual accuracy, hallucination, sycophancy or abstention quality; inspect predictions and use an appropriate labeled benchmark.', 'predictions': results}
    directory = storage.ROOT / 'evaluations'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{job_id}.json').write_text(json.dumps(report, indent=2))
    return storage.save('evaluation', report, job_id)


if __name__ == '__main__':
    run()
