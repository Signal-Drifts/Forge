# Forge local backend

Implemented: hardware detection, model-format detection, compatible Safetensors inspection, revision-pinned Hugging Face downloads, model loading, JSONL validation, baseline evaluation, LoRA, QLoRA, Full Fine-Tuning, Magnitude, Wanda, SparseGPT, Structured and 2:4 N:M pruning, and 4, 8, 16, 24, or 32-bit output conversion. Apple silicon uses MLX; Windows, Linux, and Intel Mac use PyTorch.

## Run

From the project directory, with Python 3.12 and uv installed:

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.in
backend/run.sh
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.in
.\backend\run.ps1
```

Linux or Intel Mac:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.in
backend/run.sh
```

Apple silicon uses MLX. Windows, Linux, and Intel Mac use PyTorch. QLoRA on
Windows or Linux requires a CUDA or ROCm GPU. Full Fine-Tuning requires a
16-bit or higher model and enough memory for model weights, gradients, and
optimizer state.

Run `npm run dev` in another terminal and open http://localhost:3000. The API binds only to 127.0.0.1:8000. Interactive API documentation is at http://127.0.0.1:8000/docs. Weight-changing and evaluation operations use MLX on Apple silicon and PyTorch on Windows, Linux, and Intel Mac.

Local folders are read in place, not uploaded or duplicated. The folder browser starts in Downloads; a full model path can also be entered. Hugging Face requires a model ID or its model-page HTTPS URL. Private/gated models use the machine's existing Hugging Face authentication, and license approval must already be completed on Hugging Face. Remote Python code and pickle weights are not loaded.

Downloaded models, SQLite metadata, dataset splits, worker logs and evaluation reports are stored in `.forge/`, which is excluded from Git. Downloads are pinned to the repository commit inspected before the download. Only one heavy job runs at a time in a separate process. The loaded model stays in that process until another model is loaded or the backend stops. An interrupted job is marked interrupted after restart; a new download attempt can reuse completed files.

## API workflow

1. `GET /hardware` reports processor, GPU, GPU memory when available, total and available RAM, free disk and engine; macOS uses system information and Windows uses NVIDIA SMI or Windows CIM with CUDA, DirectML and CPU capability checks.
2. `POST /models/inspect` accepts `{ "source": "local", "value": "/absolute/model/folder", "operations": ["quantize"], "bits": 4 }`, or `source: "huggingface"` with a model ID.
3. `POST /models/{id}/download` queues a download; `GET /jobs` restores the active job; `POST /jobs/{job_id}/cancel` stops it and preserves resumable partial files; `POST /models/{id}/download/discard` deletes an abandoned partial download.
4. `POST /models/{id}/check` accepts operations and bits and recalculates memory estimates from current available RAM.
5. `POST /models/{id}/load` loads the downloaded/local model in the MLX worker without modifying its weights.
6. `POST /datasets` accepts multipart `file` containing UTF-8 JSONL; `GET /datasets` lists registered splits.
7. `POST /evaluations` accepts `{ "model_id": "...", "dataset_id": "...", "max_samples": 20, "max_tokens": 128 }`; poll the returned job, then read `GET /evaluations/{report_id}`.
8. `POST /optimizations` accepts pruning, quantization, and `fine_tune_method` values `lora`, `qlora`, or `full`; poll `GET /jobs/{id}` or stop the job through `POST /jobs/{id}/cancel`.

## Dataset format

Each JSONL line is one of:

```json
{"prompt":"What is 2 + 2?","response":"4","category":"accuracy"}
{"messages":[{"role":"user","content":"Capital of France?"},{"role":"assistant","content":"Paris"}],"answers":["Paris","Paris, France"],"category":"accuracy"}
```

At least five unique prompts are required. Optional categories: general, accuracy, hallucinations, sycophancy, abstention. Optional `answers` provides alternate acceptable reference answers. Train/validation/test splits use a fixed seed and approximately 80/10/10 proportions with nonempty validation and test splits. Identical normalized prompts are deduplicated before splitting; conflicting answers are rejected. This prevents exact duplicate prompt leakage, not semantic leakage between paraphrases. Dataset uploads are capped at 20 MB and each example at 16,000 characters. Baselines reject prompts over 2,048 tokens rather than silently truncating them.

## Memory estimates and limitations

The approved memory message substitutes the rounded estimated peak in GB at runtime. It is a planning warning, not a measured minimum or an assertion that all operations have identical requirements. API responses include `estimate_only`, `by_operation`, `basis`, available bytes and required bytes. The peak of selected operations is used, not their sum; they run sequentially. Quantization estimates account for source weights and conversion copies, so a model is not rejected merely because its unquantized inference footprint is large. A memory warning does not block downloading; insufficient disk space or an unsupported format does. Actual loading has its own runtime memory check.

Current formulas are conservative heuristics based on weights, largest shard, output-copy size and fixed workspace allowances. Actual requirements vary with architecture, batch size, sequence length, method, optimizer and engine. Full Fine-Tuning includes a larger planning allowance for gradients and optimizer state. Estimates remain planning values and should be refined from observed peaks for each supported architecture.

The API supervises the model worker. A stopped worker is recovered and restarted automatically before the next health check, job request, or job poll.

Wanda and SparseGPT use captured Linear-layer inputs from representative training examples. SparseGPT applies blockwise calibration Hessians to bound memory use. Structured pruning zeros complete output rows while preserving tensor shapes. N:M uses a fixed 2:4 pattern; runtime speedups still depend on hardware and kernel support.

Baseline reports contain normalized exact match, token F1, category breakdowns, actual predictions, timing and model/dataset identifiers. These metrics are suitable for reference-answer tasks and do not certify factual truth, hallucination rates or sycophancy quality. Generated outputs need task-appropriate benchmarks or human review. No fabricated safety/accuracy scores are shown.

## Validation

```sh
.venv/bin/python -m pytest backend/tests -q
npx tsc --noEmit
npm run build
```
