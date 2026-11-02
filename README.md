# Forge

Local model in, tuned model out. Your model files stay on your machine.

Website: [https://signal-drifts.github.io/YoFile/](https://signal-drifts.github.io/YoFile/)

Current status: **development build**

## Why this exists

Fine-tuning, pruning, and quantizing a local language model usually means assembling separate command-line tools, translating model formats, estimating memory by hand, and tracking output folders across several runs.

The underlying libraries already do the hard model work. What is missing is one clear local interface that checks the machine, validates the model and dataset, runs the selected operation, shows progress, and lets the user choose where the finished model is saved.

Forge is that interface. It keeps model files on the computer and runs supported operations through MLX on Apple silicon or PyTorch on Windows, Linux, and Intel Mac.

## What it does

- Opens a local model folder or inspects a Hugging Face model ID or model-page URL
- Checks model format, architecture, runtime, available memory, and disk space before work starts
- Downloads Hugging Face models at the exact inspected repository revision
- Fine-tunes with LoRA, QLoRA, or Full Fine-Tuning
- Prunes with Magnitude, Wanda, SparseGPT, Structured, or fixed 2:4 N:M pruning
- Converts output precision to 4, 8, 16, 24, or 32 bits when the selected backend supports it
- Validates UTF-8 JSONL datasets, removes exact duplicate prompts, and creates deterministic train, validation, and test splits
- Runs heavy model work in one supervised worker process with progress, pause, resume, stop, and recovery states
- Keeps temporary outputs private until the operation finishes and the user chooses a save folder
- Produces baseline exact-match and token-F1 reports through the local API

## What it avoids

| Manual model tooling | Forge |
|---|---|
| Separate commands for inspection, training, pruning, and conversion | One local workflow |
| Guess whether the model and machine are compatible | Compatibility and memory checks before execution |
| Download whichever repository state is current later | Download the exact inspected revision |
| Lose track of partial jobs after a worker restart | Recover job state and resumable downloads |
| Save automatically to an opaque generated path | Choose and name the output folder after completion |
| Send local model paths or datasets to a hosted workflow | Run the backend on `127.0.0.1` |

## How to use

Install Node.js 22 or newer, Python 3.12, and the platform dependencies listed in [`backend/requirements.in`](backend/requirements.in).

Apple silicon, Linux, or Intel Mac:

```sh
npm ci
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.in
backend/run.sh
```

Windows PowerShell:

```powershell
npm ci
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.in
.\backend\run.ps1
```

In a second terminal:

```sh
npm run dev
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000), select a model, configure an operation, and press **Run**. Forge asks for the final output location after processing completes.

## Dataset format

Each line must be one JSON object in prompt-response or chat-message form:

```json
{"prompt":"What is 2 + 2?","response":"4","category":"accuracy"}
{"messages":[{"role":"user","content":"Capital of France?"},{"role":"assistant","content":"Paris"}],"answers":["Paris","Paris, France"],"category":"accuracy"}
```

At least five unique prompts are required. Optional categories are `general`, `accuracy`, `hallucinations`, `sycophancy`, and `abstention`. Uploads are limited to 20 MB, and each example is limited to 16,000 characters. A working example is included at [`samples/gemma-3-270m-finetune-sample.jsonl`](samples/gemma-3-270m-finetune-sample.jsonl).

## Tech stack

**Interface**

- [React](https://react.dev) and TypeScript provide the four-step local workflow.
- [Vinext](https://github.com/cloudflare/vinext) and Vite run the frontend.
- Base UI components preserve keyboard operation, focus handling, and accessible dialogs.

**Backend**

- [FastAPI](https://fastapi.tiangolo.com) exposes a loopback-only API at `127.0.0.1:8000`.
- [MLX LM](https://github.com/ml-explore/mlx-lm) runs model operations on Apple silicon.
- [PyTorch](https://pytorch.org), Transformers, PEFT, Quanto, and bitsandbytes provide Windows, Linux, and Intel Mac execution paths.
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) inspects and downloads revision-pinned repositories.
- SQLite stores model, dataset, job, and evaluation metadata in the ignored local `.forge/` directory.

**How an operation actually runs**

The interface sends model and operation settings to the local API. The API validates the plan and memory estimate, then queues one worker job. When several operations are selected, the worker runs them in this order: prune, fine-tune, then quantize. Intermediate files remain under `.forge/outputs/`; after completion, Forge copies the result only to the folder the user selects.

The worker owns model memory and processes one heavy job at a time. A stopped download keeps valid partial files for resume. A cancelled optimization removes its temporary output. Restart recovery marks interrupted work clearly instead of presenting a partial model as complete.

**Distribution**

- Forge currently runs from source as a local web interface and Python backend.
- A packaged desktop installer is not included in this repository yet.
- Model files, datasets, outputs, worker logs, and local database records are excluded from Git.

## Limits

Forge changes supported Safetensors models and supported PyTorch checkpoints. GGUF, ONNX, TensorFlow SavedModel, and TensorRT are detected but are not trained directly.

QLoRA through the PyTorch backend requires a CUDA or ROCm GPU. Full Fine-Tuning requires a 16-bit, 24-bit, or 32-bit source model and substantially more memory than adapter training. A 24-bit output uses float32 storage with 24-bit precision metadata.

Memory checks are conservative planning estimates based on source weights, the largest shard, output copies, and workspace allowances. Actual peak memory depends on model architecture, sequence length, optimizer, runtime, and hardware.

The macOS workflow has been exercised locally. Windows and Linux branches have automated coverage, but complete model runs still need verification on each target hardware and driver combination.

## Validation

```sh
uvx ruff check backend
.venv/bin/python -m pytest -q backend/tests
node --test --experimental-strip-types tests/*.test.ts
npx tsc --noEmit --noUnusedLocals --noUnusedParameters
npm run lint
npm run build
npm audit --audit-level=moderate
uv pip check --python .venv/bin/python
```

Current validation: 84 backend tests, 58 frontend tests, production build, dependency compatibility, and zero known npm vulnerabilities.

## License

MIT.
