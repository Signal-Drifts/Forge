export const precisionOptions = [4, 8, 16, 24, 32] as const;

export type OptimizationPromptItem = {
  title: 'Optimization' | 'Model' | 'Prune' | 'Quantize' | 'Fine-Tuning';
  message: string;
};

export type OptimizationPrompt = {
  title: 'Complete Optimization Settings' | 'Confirm Optimization';
  canRun: boolean;
  items: OptimizationPromptItem[];
};

type OptimizationSelection = {
  sourceBits: number | null | undefined;
  prune: boolean;
  method: string;
  ratio: string;
  hasDataset: boolean;
  quantize: boolean;
  targetBits: string;
};

const pruningBits = new Set([16, 24, 32]);
const calibrationMethods = new Set(['wanda', 'sparsegpt']);

export function getOptimizationPrompt({
  sourceBits,
  prune,
  method,
  ratio,
  hasDataset,
  quantize,
  targetBits,
}: OptimizationSelection): OptimizationPrompt {
  const errors: OptimizationPromptItem[] = [];

  if (!prune && !quantize) {
    errors.push({
      title: 'Optimization',
      message: 'Select Prune or Quantize.',
    });
  }

  if (
    !sourceBits ||
    !(precisionOptions as readonly number[]).includes(sourceBits)
  ) {
    errors.push({
      title: 'Model',
      message: 'Select a model with a recognized precision.',
    });
  }

  if (prune && sourceBits && !pruningBits.has(sourceBits)) {
    errors.push({
      title: 'Prune',
      message: 'Pruning requires a 16-bit, 24-bit, or 32-bit model.',
    });
  } else if (prune && !method) {
    errors.push({
      title: 'Prune',
      message: 'Select a pruning method.',
    });
  } else if (
    prune &&
    method !== 'nm' &&
    !(Number(ratio) >= 1 && Number(ratio) <= 80)
  ) {
    errors.push({
      title: 'Prune',
      message: 'Enter a pruning percentage from 1% to 80%.',
    });
  } else if (prune && calibrationMethods.has(method) && !hasDataset) {
    errors.push({
      title: 'Prune',
      message: `Choose a calibration dataset for ${method === 'wanda' ? 'Wanda' : 'SparseGPT'}.`,
    });
  }

  const target = Number(targetBits);
  if (
    quantize &&
    (!targetBits || !(precisionOptions as readonly number[]).includes(target))
  ) {
    errors.push({
      title: 'Quantize',
      message: 'Select a target precision.',
    });
  } else if (quantize && target === sourceBits) {
    errors.push({
      title: 'Quantize',
      message: `Select a precision different from the model's ${sourceBits}-bit precision.`,
    });
  }

  if (errors.length) {
    return {
      title: 'Complete Optimization Settings',
      canRun: false,
      items: errors,
    };
  }

  const warnings: OptimizationPromptItem[] = [];
  if (prune) {
    warnings.push({
      title: 'Prune',
      message:
        "Pruning changes the model's weights and may affect model quality.",
    });
  }
  if (quantize && sourceBits) {
    if (target < sourceBits) {
      warnings.push({
        title: 'Quantize',
        message: `Changing from ${sourceBits}-bit to ${target}-bit may reduce precision. Continue?`,
      });
    }
  }

  return {
    title: 'Confirm Optimization',
    canRun: true,
    items: warnings,
  };
}
