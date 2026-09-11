import assert from 'node:assert/strict';
import test from 'node:test';

import { modelRequirementState } from '../lib/model-requirements.ts';

const review = (os: string, engine_id: string | null, engine: string) => ({
  hardware: {
    chip: os === 'Darwin' ? 'Apple M1' : 'x86_64',
    architecture: os === 'Darwin' ? 'arm64' : 'x86_64',
    os,
    gpu: os === 'Windows' ? 'NVIDIA GPU' : 'Not detected',
    gpu_memory_bytes: null,
    graphics_requirement:
      os === 'Darwin'
        ? 'Apple silicon GPU, M1 or newer'
        : 'PyTorch-compatible GPU or CPU',
    engine_id,
    engine,
    ram_total_bytes: 16_000_000_000,
    ram_available_bytes: 8_000_000_000,
    disk_free_bytes: 50_000_000_000,
  },
  memory: {
    message: null,
    insufficient_memory: false,
    required_gb: 2,
    available_bytes: 8_000_000_000,
    output_weight_bytes_estimate: 1_000_000_000,
    estimate_only: true,
  },
  model: {
    id: 'model',
    source: 'local' as const,
    name: 'model',
    path: '/model',
    downloaded: true,
    compatible: true,
    compatibility_error: null,
    model_format: 'safetensors',
    model_format_name: 'Safetensors',
    parameter_count: 1,
    source_bits: 16,
    weight_bytes: 1,
    download_bytes: 1,
  },
});

void test('supported MLX, Windows PyTorch, and Linux PyTorch systems pass', () => {
  for (const system of [
    review('Darwin', 'mlx', 'MLX available'),
    review('Windows', 'torch-cuda', 'PyTorch CUDA available'),
    review('Linux', 'torch-cpu', 'PyTorch CPU available'),
  ]) {
    assert.equal(modelRequirementState(system).requirementsMet, true);
  }
});

void test('a missing execution runtime fails the compatibility check', () => {
  const state = modelRequirementState(review('Linux', null, 'Unavailable'));
  assert.equal(state.requirementsMet, false);
  assert.equal(state.checks[1], false);
});
