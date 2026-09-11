import type { HardwareInfo, MemoryCheck, ModelInfo } from './backend';

type CompatibilityReview = {
  hardware: HardwareInfo;
  memory: MemoryCheck;
  model: ModelInfo;
};

export function modelRequirementState(review: CompatibilityReview | null) {
  if (!review) {
    return {
      checks: [] as boolean[],
      compute: '',
      runtime: '',
      requirementsMet: false,
    };
  }

  const runtimeName =
    review.hardware.os === 'Darwin' && review.hardware.architecture === 'arm64'
      ? 'MLX'
      : 'PyTorch';
  const detectedCompute =
    review.hardware.gpu !== 'Not detected'
      ? review.hardware.gpu
      : review.hardware.chip;
  const checks = [
    !review.memory.insufficient_memory,
    Boolean(review.hardware.engine_id),
    review.model.compatible,
  ];

  return {
    checks,
    compute: `${review.hardware.graphics_requirement} / ${detectedCompute} detected`,
    runtime: `${runtimeName} required / ${
      review.model.compatible
        ? review.hardware.engine
        : review.model.compatibility_error || 'Model unsupported'
    }`,
    requirementsMet: checks.every(Boolean),
  };
}
