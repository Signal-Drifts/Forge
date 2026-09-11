import os
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)
from pydantic import BaseModel, Field, field_validator

from . import storage
from .datasets import import_dataset
from .models import (
    discard_partial_download,
    download_progress,
    estimate,
    hardware,
    inspect_hub,
    inspect_local,
    pick_folder,
    prepare_download_destination,
)

ALLOWED_ORIGINS = {'http://localhost:3000', 'http://127.0.0.1:3000'}


class ModelRequest(BaseModel):
    source: Literal['local', 'huggingface']
    value: str = Field(min_length=1, max_length=2000)
    operations: list[Literal['quantize', 'prune', 'retrain']] = Field(default=['quantize'], max_length=3)
    bits: Literal[4, 8, 16, 24, 32] = 4
    fine_tune_method: Literal['lora', 'qlora', 'full'] = 'lora'


class CheckRequest(BaseModel):
    operations: list[Literal['quantize', 'prune', 'retrain']] = Field(default=['quantize'], max_length=3)
    bits: Literal[4, 8, 16, 24, 32] = 4
    fine_tune_method: Literal['lora', 'qlora', 'full'] = 'lora'


class DownloadRequest(BaseModel):
    destination: str | None = Field(default=None, max_length=2000)


class BaselineRequest(BaseModel):
    model_id: str
    dataset_id: str
    max_samples: int = Field(default=20, ge=1, le=100)
    max_tokens: int = Field(default=128, ge=1, le=512)


class OptimizeRequest(BaseModel):
    model_id: str
    dataset_id: str | None = None
    fine_tune: bool = False
    fine_tune_method: Literal['lora', 'qlora', 'full'] = 'lora'
    prune: bool = False
    prune_method: Literal['magnitude', 'wanda', 'sparsegpt', 'structured', 'nm'] = 'magnitude'
    prune_ratio: float = Field(default=0.1, gt=0, le=0.8)
    quantize: bool = False
    bits: Literal[4, 8, 16, 24, 32] = 4
    iterations: int = Field(default=100, ge=1, le=5000)


class SaveOutputRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=2000)
    folder_name: str = Field(min_length=1, max_length=120)

    @field_validator('folder_name')
    @classmethod
    def validate_folder_name(cls, value: str):
        name = value.strip()
        if (
            name in {'.', '..'}
            or any(character in name for character in '/\\<>:"|?*\0')
            or name.endswith(('.', ' '))
        ):
            raise ValueError('Enter a valid folder name.')
        return name


def prepare_optimization_output(parent_value: str, folder_name: str) -> str:
    parent = Path(parent_value).expanduser().resolve(strict=True)
    if not parent.is_dir():
        raise ValueError('Select an output folder.')
    if not os.access(parent, os.W_OK):
        raise PermissionError('The selected output folder is not writable.')
    destination = parent / folder_name
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise ValueError(
            f'A folder named {folder_name} already exists in this location. Choose another location.'
        ) from error
    (destination / '.forge-operation').write_text('in-progress\n')
    return str(destination)


def create_app(start_worker=True):
    worker_lock = threading.Lock()

    def start_model_worker(app):
        pid_file = storage.ROOT / 'worker.pid'
        if pid_file.exists():
            try:
                old = psutil.Process(int(pid_file.read_text().strip()))
                if 'backend.worker' in ' '.join(old.cmdline()):
                    old.terminate()
                    old.wait(timeout=5)
            except (ValueError, OSError, psutil.Error, psutil.TimeoutExpired):
                pass
            pid_file.unlink(missing_ok=True)
        log = (storage.ROOT / 'worker.log').open('a')
        environment = {**os.environ, 'FORGE_DATA_DIR': str(storage.ROOT)}
        process = subprocess.Popen(
            [sys.executable, '-m', 'backend.worker'],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log,
            stderr=log,
            start_new_session=os.name != 'nt',
        )
        app.state.worker = process
        app.state.worker_log = log

    def stop_model_worker(app):
        process = getattr(app.state, 'worker', None)
        log = getattr(app.state, 'worker_log', None)
        if process and process.poll() is None:
            try:
                worker = psutil.Process(process.pid)
                worker.resume()
                for child in worker.children(recursive=True):
                    child.resume()
            except psutil.Error:
                pass
            if os.name != 'nt':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        if log and not log.closed:
            log.close()
        app.state.worker = None
        app.state.worker_log = None

    def set_model_worker_paused(app, paused):
        process = getattr(app.state, 'worker', None)
        if not process or process.poll() is not None:
            raise ValueError('The model worker is unavailable.')
        try:
            worker = psutil.Process(process.pid)
            children = worker.children(recursive=True)
            if paused:
                for child in reversed(children):
                    child.suspend()
                worker.suspend()
            else:
                worker.resume()
                for child in children:
                    child.resume()
        except psutil.Error as error:
            raise ValueError('The operation could not be paused or resumed.') from error

    def ensure_model_worker(app):
        if not start_worker:
            return False
        with worker_lock:
            process = getattr(app.state, 'worker', None)
            if process and process.poll() is None:
                return True
            stop_model_worker(app)
            storage.recover_jobs()
            start_model_worker(app)
            return True

    @asynccontextmanager
    async def lifespan(app):
        storage.initialize()
        app.state.worker = None
        app.state.worker_log = None
        if start_worker:
            storage.recover_jobs()
            start_model_worker(app)
        try:
            yield
        finally:
            if start_worker:
                with worker_lock:
                    stop_model_worker(app)
                storage.recover_jobs()

    app = FastAPI(title='Forge Local Backend', version='0.1.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=sorted(ALLOWED_ORIGINS), allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])

    @app.middleware('http')
    async def local_access(request, call_next):
        origin = request.headers.get('origin')
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse(status_code=403, content={'detail': 'This backend only accepts the local Forge interface.'})
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_input(request, error):
        return JSONResponse(status_code=422, content={'detail': str(error)})

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, error):
        return JSONResponse(status_code=404, content={'detail': 'The selected folder or model files no longer exist.'})

    @app.exception_handler(PermissionError)
    async def permission_error(request, error):
        return JSONResponse(status_code=403, content={'detail': 'The local backend cannot read this folder.'})

    def queue(kind, payload):
        ensure_model_worker(app)
        try:
            return storage.job(storage.create_job(kind, payload))
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    def job_with_progress(current):
        if not current or current['kind'] != 'download':
            return current
        try:
            model = storage.get(current['payload']['model_id'], 'model')
            progress = download_progress(model, current['status'] == 'succeeded')
        except (KeyError, ValueError):
            progress = {'downloaded_bytes': 0, 'total_bytes': 0, 'progress_percent': 0}
        return {**current, **progress}

    @app.get('/health')
    def health():
        ensure_model_worker(app)
        return {'status': 'ok', 'worker_ready': bool(app.state.worker and app.state.worker.poll() is None), 'implemented': ['hardware', 'model_inspection', 'download', 'load', 'dataset_validation', 'baseline_evaluation', 'mlx_lora', 'mlx_qlora', 'mlx_full_fine_tuning', 'mlx_quantization', 'pytorch_lora', 'pytorch_qlora', 'pytorch_full_fine_tuning', 'pytorch_quantization', 'windows_model_execution', 'linux_model_execution', 'magnitude_pruning', 'wanda_pruning', 'sparsegpt_pruning', 'structured_pruning', 'nm_pruning'], 'not_implemented': ['direct_training_of_inference_formats']}

    @app.get('/hardware')
    def read_hardware():
        return hardware()

    @app.get('/folders')
    def folders(path: str | None = None):
        home = Path.home().resolve()
        directory = Path(path).expanduser().resolve(strict=True) if path else home / 'Downloads'
        if not directory.is_relative_to(home):
            raise ValueError('Browse folders within your home directory, or enter a full local model path.')
        if not directory.is_dir():
            raise ValueError('Select a folder.')
        children = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name.casefold()):
            try:
                if child.is_dir() and not child.name.startswith('.') and child.resolve().is_relative_to(home):
                    children.append({'name': child.name, 'path': str(child), 'model': (child / 'config.json').is_file()})
            except OSError:
                continue
        return {'path': str(directory), 'parent': str(directory.parent) if directory != home else None, 'folders': children[:500]}

    @app.post('/folders/pick')
    def choose_folder():
        return {'path': pick_folder()}

    @app.post('/folders/pick-download')
    def choose_download_folder():
        return {'path': pick_folder('Choose Download Destination')}

    @app.post('/folders/pick-output')
    def choose_output_folder():
        return {'path': pick_folder('Choose Output Folder')}

    @app.get('/models')
    def models():
        return storage.list_records('model')

    @app.post('/models/inspect')
    def inspect_model(request: ModelRequest):
        try:
            model = inspect_local(request.value) if request.source == 'local' else inspect_hub(request.value)
        except GatedRepoError as error:
            raise HTTPException(403, 'Access to this model is restricted. Accept its access terms on Hugging Face and sign in locally.') from error
        except RepositoryNotFoundError as error:
            raise HTTPException(404, 'Model not found or not accessible. Copy the exact owner/model-name from Hugging Face. Private models require sign-in.') from error
        except EntryNotFoundError as error:
            raise HTTPException(422, 'This repository is missing the model configuration required by this tool.') from error
        except (ValueError, FileNotFoundError, PermissionError):
            raise
        except Exception as error:
            raise HTTPException(502, 'Could not read the Hugging Face model. Check the model ID, network connection, and access permissions.') from error
        return {'model': model, 'memory': estimate(model, request.operations, request.bits, fine_tune_method=request.fine_tune_method), 'hardware': hardware()}

    @app.post('/models/{model_id}/check')
    def check(model_id: str, request: CheckRequest):
        model = storage.get(model_id, 'model')
        return estimate(model, request.operations, request.bits, fine_tune_method=request.fine_tune_method)

    @app.post('/models/{model_id}/download', status_code=202)
    def download_model(model_id: str, request: DownloadRequest | None = None):
        model = storage.get(model_id, 'model')
        if not model['compatible']:
            raise ValueError(model['compatibility_error'])
        if model.get('downloaded'):
            return {'status': 'succeeded', 'result': model}
        destination = prepare_download_destination(model, request.destination if request else None)
        saved = {key: value for key, value in model.items() if key != 'id'}
        saved['download_path'] = destination
        storage.save('model', saved, model_id)
        return queue('download', {'model_id': model_id, 'destination': destination})

    @app.post('/models/{model_id}/download/discard')
    def discard_download(model_id: str):
        model = storage.get(model_id, 'model')
        if model.get('downloaded'):
            raise ValueError('A completed model cannot be discarded as a partial download.')
        discard_partial_download(model)
        saved = {key: value for key, value in model.items() if key not in {'id', 'download_path'}}
        storage.save('model', saved, model_id)
        return {'status': 'discarded'}

    @app.post('/models/{model_id}/load', status_code=202)
    def load_model(model_id: str):
        model = storage.get(model_id, 'model')
        if not model['compatible'] or not model.get('path'):
            raise ValueError('Select a compatible downloaded model first.')
        return queue('load', {'model_id': model_id})

    @app.post('/optimizations', status_code=202)
    def optimize(request: OptimizeRequest):
        from .operations import validate_plan
        model = storage.get(request.model_id, 'model')
        payload = request.model_dump()
        if request.dataset_id:
            storage.get(request.dataset_id, 'dataset')
        validate_plan(model, payload)
        operations = [name for name, enabled in [('prune', request.prune), ('retrain', request.fine_tune), ('quantize', request.quantize)] if enabled]
        memory = estimate(model, operations, request.bits, fine_tune_method=request.fine_tune_method)
        if memory['insufficient_memory']:
            raise ValueError(memory['message'])
        return queue('optimize', payload)

    @app.post('/jobs/{job_id}/save-output')
    def save_output(job_id: str, request: SaveOutputRequest):
        current = storage.job(job_id)
        if current['kind'] != 'optimize' or current['status'] != 'succeeded' or not current.get('result'):
            raise ValueError('Only a completed optimization can be saved.')
        result = current['result']
        source = Path(result['output_path']).resolve(strict=True)
        run_root = (storage.ROOT / 'outputs' / job_id).resolve(strict=True)
        if not source.is_relative_to(run_root):
            raise ValueError('This output has already been saved.')
        destination = Path(
            prepare_optimization_output(request.destination, request.folder_name)
        )
        marker = destination / '.forge-operation'
        try:
            for item in source.iterdir():
                target = destination / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
            adapter_path = result.get('adapter_path')
            saved_adapter = None
            if adapter_path:
                adapter_source = Path(adapter_path).resolve(strict=True)
                if adapter_source != source and adapter_source.is_relative_to(run_root):
                    saved_adapter = destination / 'adapters'
                    if not saved_adapter.exists():
                        shutil.copytree(adapter_source, saved_adapter)
            saved_model = inspect_local(str(destination))
            saved_result = {
                **result,
                'model': saved_model,
                'output_path': str(destination),
                'adapter_path': str(saved_adapter) if saved_adapter else None,
                'saved': True,
            }
            marker.unlink(missing_ok=True)
            storage.update_job(job_id, 'succeeded', 'Output saved', saved_result)
            shutil.rmtree(run_root, ignore_errors=True)
            return saved_result
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    @app.post('/jobs/{job_id}/discard-output')
    def discard_output(job_id: str):
        current = storage.job(job_id)
        result = current.get('result') or {}
        if current['kind'] != 'optimize' or current['status'] != 'succeeded':
            raise ValueError('Only a completed optimization can be discarded.')
        if result.get('saved'):
            raise ValueError('A saved output cannot be discarded from this screen.')
        storage.discard_job_output(job_id)
        storage.update_job(
            job_id,
            'cancelled',
            'Output discarded',
            error='Discarded by user',
        )
        return storage.job(job_id)

    @app.get('/jobs/{job_id}')
    def get_job(job_id: str):
        ensure_model_worker(app)
        return job_with_progress(storage.job(job_id))

    @app.get('/jobs')
    def active_job():
        return job_with_progress(storage.active_job())

    @app.post('/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        current = storage.job(job_id)
        if current['status'] not in ('queued', 'running', 'paused'):
            return current
        running_pause = current['status'] == 'paused' and current.get('progress') != 'Queued operation paused'
        if (current['status'] == 'running' or running_pause) and start_worker:
            with worker_lock:
                storage.update_job(job_id, 'cancelled', 'Operation stopped', error='Stopped by user')
                stop_model_worker(app)
                if current['kind'] == 'optimize':
                    storage.discard_job_output(job_id)
                start_model_worker(app)
                return storage.job(job_id)
        return storage.cancel_job(job_id)

    @app.post('/jobs/{job_id}/pause')
    def pause_job(job_id: str):
        current = storage.job(job_id)
        if current['kind'] != 'optimize':
            raise ValueError('Only an optimization can be paused.')
        if start_worker and current['status'] == 'running':
            with worker_lock:
                set_model_worker_paused(app, True)
                try:
                    return storage.pause_job(job_id)
                except Exception:
                    set_model_worker_paused(app, False)
                    raise
        return storage.pause_job(job_id)

    @app.post('/jobs/{job_id}/resume')
    def resume_job(job_id: str):
        current = storage.job(job_id)
        if current['kind'] != 'optimize':
            raise ValueError('Only an optimization can be resumed.')
        if start_worker:
            with worker_lock:
                resumed = storage.resume_job(job_id)
                try:
                    if resumed['status'] == 'running':
                        set_model_worker_paused(app, False)
                except Exception:
                    storage.update_job(job_id, 'paused', 'Operation paused')
                    raise
                return resumed
        return storage.resume_job(job_id)

    @app.post('/datasets', status_code=201)
    async def upload_dataset(file: Annotated[UploadFile, File()]):
        content = await file.read(20 * 1024 * 1024 + 1)
        await file.close()
        return import_dataset(content, Path(file.filename or 'dataset.jsonl').name)

    @app.get('/datasets')
    def datasets():
        return storage.list_records('dataset')

    @app.post('/evaluations', status_code=202)
    def baseline(request: BaselineRequest):
        model = storage.get(request.model_id, 'model')
        storage.get(request.dataset_id, 'dataset')
        if not model['compatible'] or not model.get('path'):
            raise ValueError('Select a compatible downloaded model first.')
        return queue('baseline', request.model_dump())

    @app.get('/evaluations/{report_id}')
    def evaluation(report_id: str):
        return storage.get(report_id, 'evaluation')

    return app


app = create_app()
