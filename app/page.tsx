'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Check, FolderOpen, Square } from 'lucide-react';
import Link from 'next/link';
import { WorkflowNote } from '@/components/workflow-note';
import { DisclosureGuide } from '@/components/disclosure-guide';
import { ModelSource } from '@/components/model-source';
import {
  backend,
  DatasetInfo,
  Job,
  MemoryCheck,
  ModelInfo,
  OptimizationResult,
} from '@/lib/backend';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  getOptimizationPrompt,
  OptimizationPromptItem,
  precisionOptions,
} from '@/lib/optimization-rules';
import { getRunSignature, RunScope } from '@/lib/optimization-run';

const terminal = new Set(['succeeded', 'failed', 'interrupted', 'cancelled']);
const pause = () => new Promise((resolve) => setTimeout(resolve, 800));
const pruningMethods = [
  ['magnitude', 'Magnitude'],
  ['wanda', 'Wanda'],
  ['sparsegpt', 'SparseGPT'],
  ['structured', 'Structured'],
  ['nm', 'N:M (2:4)'],
] as const;
const emptySelectValue = '__select__';
const modelOutputPath = (model: ModelInfo) => model.path || model.name;
const outputDestination = (value: string) => {
  const path = value.trim().replace(/[\\/]+$/, '');
  const separatorIndex = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  if (separatorIndex < 1) return null;
  const folderName = path.slice(separatorIndex + 1);
  const destination =
    separatorIndex === 2 && /^[A-Za-z]:/.test(path)
      ? path.slice(0, separatorIndex + 1)
      : path.slice(0, separatorIndex) || '/';
  return { destination, folderName };
};
const evaluationChecks = [
  {
    title: 'Accuracy',
    description: 'Measures how often predictions match the expected answers.',
    href: 'https://huggingface.co/docs/evaluate/types_of_evaluations',
  },
  {
    title: 'Hallucinations',
    description:
      'Checks for claims that are unsupported or factually inconsistent.',
    href: 'https://arxiv.org/abs/2202.03629',
  },
  {
    title: 'Sycophancy',
    description:
      'Checks whether the model agrees with users instead of the evidence.',
    href: 'https://arxiv.org/abs/2502.08177',
  },
  {
    title: 'Abstention',
    description:
      'Checks whether the model withholds answers when information is insufficient.',
    href: 'https://arxiv.org/abs/2506.09038',
  },
] as const;

function InfoBar({
  index,
  title,
  children,
  href,
}: {
  index: number;
  title: string;
  children: React.ReactNode;
  href: string;
}) {
  return (
    <div className="reference-bar">
      <span className="workflow-index">{String(index).padStart(2, '0')}</span>
      <div className="workflow-copy">
        <strong>{title}</strong>
        <p>{children}</p>
      </div>
      <a href={href} target="_blank" rel="noopener noreferrer">
        Learn More
      </a>
    </div>
  );
}

function OptimizationGuide() {
  return (
    <DisclosureGuide title="Optimization Guide" countLabel="3 items">
      <div className="reference-bars">
        <InfoBar
          index={1}
          title="Prune"
          href="https://pytorch.org/tutorials/intermediate/pruning_tutorial.html"
        >
          Removes low-value weights before fine-tuning.
        </InfoBar>
        <InfoBar
          index={2}
          title="Quantize"
          href="https://huggingface.co/docs/transformers/quantization/overview"
        >
          Reduces weight precision after fine-tuning.
        </InfoBar>
        <InfoBar
          index={3}
          title="Weights"
          href="https://developers.google.com/machine-learning/crash-course/neural-networks/parameters"
        >
          Learned numerical values that control the model&apos;s behavior.
        </InfoBar>
      </div>
    </DisclosureGuide>
  );
}

function FineTuneGuide() {
  return (
    <DisclosureGuide title="Fine-Tuning Guide" countLabel="2 items">
      <div className="reference-bars">
        <InfoBar
          index={1}
          title="Adjust Weights"
          href="https://huggingface.co/docs/peft/main/en/conceptual_guides/lora"
        >
          Trains LoRA adapters from the selected dataset.
        </InfoBar>
        <InfoBar
          index={2}
          title="Training Steps"
          href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments"
        >
          Number of batches used to update the adapters.
        </InfoBar>
      </div>
    </DisclosureGuide>
  );
}

function EvaluationGuide() {
  return (
    <DisclosureGuide title="Evaluation Guide" countLabel="4 items">
      <div className="reference-bars">
        {evaluationChecks.map(({ title, description, href }, index) => (
          <InfoBar key={title} index={index + 1} title={title} href={href}>
            {description}
          </InfoBar>
        ))}
      </div>
    </DisclosureGuide>
  );
}

export default function Home() {
  const [bits, setBits] = useState('');
  const [quantize, setQuantize] = useState(false);
  const tune = true;
  const [fineTuneMethod, setFineTuneMethod] = useState('lora');
  const [prune, setPrune] = useState(false);
  const [ratio, setRatio] = useState('');
  const [method, setMethod] = useState('');
  const [iterations, setIterations] = useState('');
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState<ModelInfo | null>(null);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [fineTuneModelBusy, setFineTuneModelBusy] = useState(false);
  const [fineTuneModelError, setFineTuneModelError] = useState('');
  const [memory, setMemory] = useState<MemoryCheck | null>(null);
  const [memoryError, setMemoryError] = useState('');
  const [connected, setConnected] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [completedJobId, setCompletedJobId] = useState<string | null>(null);
  const [jobPaused, setJobPaused] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [runStatus, setRunStatus] = useState('');
  const [runError, setRunError] = useState('');
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [saveBusy, setSaveBusy] = useState(false);
  const [outputPath, setOutputPath] = useState('');
  const [activeRunScope, setActiveRunScope] = useState<RunScope>('optimize');
  const [pruneTooltipOpen, setPruneTooltipOpen] = useState(false);
  const [confirmation, setConfirmation] = useState<{
    scope: RunScope;
    title: string;
    canRun: boolean;
    items: OptimizationPromptItem[];
    kind?: 'validation' | 'warning' | 'repeat';
    signature?: string;
  } | null>(null);
  const polling = useRef(true);
  const completedRunSignatures = useRef(new Set<string>());
  const pendingRunSignature = useRef('');
  const calibrationInput = useRef<HTMLInputElement>(null);
  const trainingInput = useRef<HTMLInputElement>(null);

  const operations = useMemo(
    () => [
      ...(prune ? ['prune'] : []),
      ...(quantize ? ['quantize'] : []),
      ...(tune ? ['retrain'] : []),
    ],
    [quantize, prune, tune],
  );

  const selectModel = useCallback((value: ModelInfo | null) => {
    setSelected(value);
    setPrune(false);
    setMethod('');
    setRatio('');
    setQuantize(false);
    setBits('');
    setFineTuneMethod('lora');
    setIterations('');
    setDataset(null);
    setDatasetError('');
    setFineTuneModelError('');
    setMemory(null);
    setMemoryError('');
    setResult(null);
    setCompletedJobId(null);
    setJobPaused(false);
    setRunStatus('');
    setRunError('');
    setOutputPath('');
    setActiveRunScope('optimize');
    setConfirmation(null);
    setPruneTooltipOpen(false);
  }, []);

  useEffect(() => {
    const previous = history.scrollRestoration;
    history.scrollRestoration = 'manual';
    window.scrollTo(0, 0);
    return () => {
      history.scrollRestoration = previous;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const checkHealth = () => {
      void backend<{ status: string }>('/health', {
        signal: controller.signal,
      })
        .then(() => setConnected(true))
        .catch((error) => {
          if (error.name !== 'AbortError') setConnected(false);
        });
    };
    checkHealth();
    const timer = window.setInterval(checkHealth, 3000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    backend<MemoryCheck>(`/models/${selected.id}/check`, {
      method: 'POST',
      body: JSON.stringify({
        operations,
        bits: Number(bits || selected.source_bits || 16),
        fine_tune_method: fineTuneMethod,
      }),
      signal: controller.signal,
    })
      .then((value) => {
        setMemory(value);
        setMemoryError('');
        setConnected(true);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setMemory(null);
          setMemoryError(error.message);
          setConnected(false);
        }
      });
    return () => controller.abort();
  }, [selected, operations, bits, fineTuneMethod]);

  const finishJob = useCallback(async (id: string) => {
    polling.current = true;
    let job = await backend<Job<OptimizationResult>>(`/jobs/${id}`);
    while (polling.current && !terminal.has(job.status)) {
      setRunStatus(job.progress || 'Running operation');
      setJobPaused(job.status === 'paused');
      await pause();
      job = await backend<Job<OptimizationResult>>(`/jobs/${id}`);
    }
    if (!polling.current) return;
    setJobId(null);
    setJobPaused(false);
    if (job.status === 'succeeded' && job.result) {
      setResult(job.result);
      setCompletedJobId(id);
      setRunStatus('Operation complete');
      if (pendingRunSignature.current) {
        completedRunSignatures.current.add(pendingRunSignature.current);
        pendingRunSignature.current = '';
      }
      return;
    }
    if (job.status === 'cancelled') {
      setRunStatus('Operation stopped');
      return;
    }
    throw new Error(job.error || 'Optimization did not complete.');
  }, []);

  useEffect(() => {
    backend<Job<OptimizationResult> | null>('/jobs')
      .then((job) => {
        if (!job || job.kind !== 'optimize' || terminal.has(job.status)) return;
        setJobId(job.id);
        setJobPaused(job.status === 'paused');
        setRunOpen(true);
        setRunStatus(job.progress || 'Running operation');
        void finishJob(job.id).catch((error) => setRunError(error.message));
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setConnected(false);
      });
    return () => {
      polling.current = false;
    };
  }, [finishJob]);

  const uploadDataset = async (file?: File) => {
    if (!file) return;
    setDatasetBusy(true);
    setDatasetError('');
    setDataset(null);
    try {
      const body = new FormData();
      body.append('file', file);
      setDataset(
        await backend<DatasetInfo>('/datasets', { method: 'POST', body }),
      );
    } catch (error) {
      setDatasetError((error as Error).message);
    } finally {
      setDatasetBusy(false);
    }
  };

  const chooseFineTuneModel = async () => {
    setFineTuneModelBusy(true);
    setFineTuneModelError('');
    try {
      const folder = await backend<{ path: string | null }>('/folders/pick', {
        method: 'POST',
      });
      if (!folder.path) return;
      const checked = await backend<{
        model: ModelInfo;
        memory: MemoryCheck;
      }>('/models/inspect', {
        method: 'POST',
        body: JSON.stringify({
          source: 'local',
          value: folder.path,
          operations: ['retrain'],
          bits: selected?.source_bits || 16,
        }),
      });
      if (!checked.model.compatible) {
        throw new Error(
          checked.model.compatibility_error ||
            'This model is not compatible with Fine-Tuning.',
        );
      }
      selectModel(checked.model);
      setMemory(checked.memory);
    } catch (error) {
      setFineTuneModelError((error as Error).message);
    } finally {
      setFineTuneModelBusy(false);
    }
  };

  const startOptimization = async (scope: RunScope, signature: string) => {
    if (!selected) return;
    setActiveRunScope(scope);
    setOutputPath(modelOutputPath(selected));
    const runFineTune = scope !== 'optimize' && tune;
    const runPrune = scope !== 'fine-tune' && prune;
    const runQuantize = scope !== 'fine-tune' && quantize;
    setRunError('');
    pendingRunSignature.current = signature;
    setRunStatus(
      scope === 'fine-tune'
        ? 'Starting fine-tuning'
        : scope === 'optimize'
          ? 'Starting optimization'
          : 'Starting workflow',
    );
    setResult(null);
    setCompletedJobId(null);
    setJobPaused(false);
    setRunOpen(true);
    try {
      const job = await backend<Job<OptimizationResult>>('/optimizations', {
        method: 'POST',
        body: JSON.stringify({
          model_id: selected.id,
          dataset_id:
            runFineTune || (runPrune && ['wanda', 'sparsegpt'].includes(method))
              ? dataset?.id
              : null,
          fine_tune: runFineTune,
          fine_tune_method: runFineTune ? fineTuneMethod : 'lora',
          prune: runPrune,
          prune_method: runPrune ? method : 'magnitude',
          prune_ratio: runPrune && method !== 'nm' ? Number(ratio) / 100 : 0.1,
          quantize: runQuantize,
          bits: Number(bits || selected.source_bits || 16),
          iterations: runFineTune ? Number(iterations) : 100,
        }),
      });
      setJobId(job.id);
      await finishJob(job.id);
    } catch (error) {
      setJobId(null);
      pendingRunSignature.current = '';
      setRunError((error as Error).message);
    }
  };

  const pickOutputFolder = async () => {
    const folder = await backend<{ path: string | null }>(
      '/folders/pick-output',
      { method: 'POST' },
    );
    if (folder.path) setOutputPath(folder.path);
    return folder.path;
  };

  const persistOutput = async (path: string) => {
    if (!completedJobId) return;
    const output = outputDestination(path);
    if (!output) {
      setRunError('Enter a complete output folder path.');
      return;
    }
    setSaveBusy(true);
    setRunError('');
    try {
      const saved = await backend<OptimizationResult>(
        `/jobs/${completedJobId}/save-output`,
        {
          method: 'POST',
          body: JSON.stringify({
            destination: output.destination,
            folder_name: output.folderName,
          }),
        },
      );
      setResult(saved);
      setRunStatus('Output saved');
    } catch (error) {
      setRunError((error as Error).message);
    } finally {
      setSaveBusy(false);
    }
  };

  const browseOutputFolder = async () => {
    setSaveBusy(true);
    setRunError('');
    try {
      await pickOutputFolder();
    } catch (error) {
      setRunError((error as Error).message);
    } finally {
      setSaveBusy(false);
    }
  };

  const saveOutput = async () => {
    if (!outputPath.trim()) {
      setRunError('Enter an output folder path.');
      return;
    }
    await persistOutput(outputPath);
  };

  const discardCompletedOutput = async () => {
    if (!completedJobId) return;
    setSaveBusy(true);
    setRunError('');
    try {
      await backend(`/jobs/${completedJobId}/discard-output`, {
        method: 'POST',
      });
      setResult(null);
      setCompletedJobId(null);
      setRunStatus('');
      setRunOpen(false);
      setStep(1);
      window.scrollTo(0, 0);
    } catch (error) {
      setRunError((error as Error).message);
    } finally {
      setSaveBusy(false);
    }
  };

  const requestOptimization = (scope: RunScope) => {
    if (!selected) {
      const fineTuningOnly = scope === 'fine-tune';
      setConfirmation({
        scope,
        title: fineTuningOnly
          ? 'Complete Fine-Tuning Settings'
          : 'Complete Optimization Settings',
        canRun: false,
        items: [
          {
            title: fineTuningOnly ? 'Fine-Tuning' : 'Model',
            message: fineTuningOnly
              ? 'Select a model, method, training dataset, and steps before running.'
              : 'Select a model, prune or quantize values before running.',
          },
        ],
      });
      return;
    }
    const runPrune = scope !== 'fine-tune' && prune;
    const runQuantize = scope !== 'fine-tune' && quantize;
    const runFineTune = scope !== 'optimize' && tune;
    const signature = getRunSignature({
      modelId: selected.id,
      modelPath: selected.path,
      prune: runPrune,
      method: runPrune ? method : '',
      ratio: runPrune && method !== 'nm' ? ratio : '',
      quantize: runQuantize,
      bits: runQuantize ? bits : '',
      fineTune: runFineTune,
      fineTuneMethod: runFineTune ? fineTuneMethod : '',
      iterations: runFineTune ? iterations : '',
      datasetId:
        runFineTune || (runPrune && ['wanda', 'sparsegpt'].includes(method))
          ? dataset?.id || ''
          : '',
    });
    const fineTuneErrors: OptimizationPromptItem[] = [];
    if (runFineTune && !dataset) {
      fineTuneErrors.push({
        title: 'Fine-Tuning',
        message: 'Choose a training dataset.',
      });
    } else if (runFineTune && !validIterations) {
      fineTuneErrors.push({
        title: 'Fine-Tuning',
        message: 'Enter Training Steps from 1 to 5000.',
      });
    }
    if (fineTuneErrors.length) {
      setConfirmation({
        scope,
        title: 'Complete Optimization Settings',
        canRun: false,
        items: fineTuneErrors,
      });
      return;
    }
    if (scope === 'workflow' && !runPrune && !runQuantize && !runFineTune) {
      setConfirmation({
        scope,
        title: 'Complete Optimization Settings',
        canRun: false,
        items: [
          {
            title: 'Optimization',
            message: 'Select at least one operation.',
          },
        ],
      });
      return;
    }
    if (scope === 'optimize' || runPrune || runQuantize) {
      const prompt = getOptimizationPrompt({
        sourceBits: selected.source_bits,
        prune: runPrune,
        method,
        ratio,
        hasDataset: !!dataset,
        quantize: runQuantize,
        targetBits: bits,
      });
      if (prompt.canRun && completedRunSignatures.current.has(signature)) {
        setConfirmation({
          scope,
          title: 'Run Optimization Again?',
          canRun: true,
          kind: 'repeat',
          signature,
          items: [
            {
              title: 'Optimization',
              message:
                'This model was already optimized with these settings. Run it again?',
            },
          ],
        });
        return;
      }
      if (prompt.canRun && !prompt.items.length) {
        void startOptimization(scope, signature);
        return;
      }
      setConfirmation({
        scope,
        ...prompt,
        kind: prompt.canRun ? 'warning' : 'validation',
        signature,
      });
      return;
    }
    if (completedRunSignatures.current.has(signature)) {
      setConfirmation({
        scope,
        title: 'Run Fine-Tuning Again?',
        canRun: true,
        kind: 'repeat',
        signature,
        items: [
          {
            title: 'Fine-Tuning',
            message:
              'This model was already fine-tuned with these settings. Run it again?',
          },
        ],
      });
      return;
    }
    void startOptimization(scope, signature);
  };

  const pauseOptimization = async () => {
    if (!jobId) return;
    setRunError('');
    try {
      const job = await backend<Job>(`/jobs/${jobId}/pause`, {
        method: 'POST',
      });
      setJobPaused(true);
      setRunStatus(job.progress || 'Operation paused');
    } catch (error) {
      setRunError((error as Error).message);
    }
  };

  const resumeOptimization = async () => {
    if (!jobId) return;
    setRunError('');
    try {
      const job = await backend<Job>(`/jobs/${jobId}/resume`, {
        method: 'POST',
      });
      setJobPaused(false);
      setRunStatus(job.progress || 'Resuming operation');
    } catch (error) {
      setRunError((error as Error).message);
    }
  };

  const cancelOptimization = async () => {
    if (!jobId) return;
    setRunError('');
    try {
      await backend(`/jobs/${jobId}/cancel`, { method: 'POST' });
      polling.current = false;
      setJobId(null);
      setJobPaused(false);
      pendingRunSignature.current = '';
      setRunStatus('');
      setRunOpen(false);
    } catch (error) {
      setRunError((error as Error).message);
    }
  };

  const outputBits =
    quantize && bits ? Number(bits) : selected?.source_bits || 16;
  const size = memory
    ? (memory.output_weight_bytes_estimate / 1e9).toFixed(1)
    : selected
      ? (
          ((selected.weight_bytes / 1e9) * outputBits) /
          (selected.source_bits || 16)
        ).toFixed(1)
      : '0.0';
  const methodLabel =
    pruningMethods.find(([value]) => value === method)?.[1] || '';
  const needsCalibration = prune && ['wanda', 'sparsegpt'].includes(method);
  const fineTuneMethodLabel =
    fineTuneMethod === 'qlora'
      ? 'QLoRA'
      : fineTuneMethod === 'full'
        ? 'Full Fine-Tuning'
        : 'LoRA';
  const reviewPipelineLabel = [
    ...(prune ? ['Prune'] : []),
    ...(quantize ? ['Quantize'] : []),
    'Fine-Tuning',
  ].join('/');
  const reviewPipelineValue = [
    ...(prune
      ? [
          `${methodLabel || 'Select'} · ${method === 'nm' ? '2:4' : `${ratio || 'Select'}%`}`,
        ]
      : []),
    ...(quantize ? [`${bits || 'Select'}-bit`] : []),
    `${fineTuneMethodLabel}/${iterations || 'Select'} Steps`,
  ].join('/');
  const iterationCount = Number(iterations);
  const validIterations =
    Number.isInteger(iterationCount) &&
    iterationCount >= 1 &&
    iterationCount <= 5000;
  const validFineTune = !!dataset && validIterations;
  const steps = ['Model', 'Optimize', 'Fine-Tuning', 'Review'];

  const playTabClick = useCallback(() => {
    const AudioContextClass =
      window.AudioContext ||
      (window as typeof window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!AudioContextClass) return;
    const context = new AudioContextClass();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = 520;
    gain.gain.setValueAtTime(0.022, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.035);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.035);
    oscillator.addEventListener('ended', () => void context.close());
  }, []);

  const optimizePanel = (
    <>
      <div className="optimization-list">
        <div className="optimization-row prune-row">
          <Checkbox
            id="prune"
            className="operation-checkbox"
            checked={prune}
            onCheckedChange={(checked) => {
              const enabled = checked === true;
              setPrune(enabled);
              if (!enabled) {
                setMethod('');
                setRatio('');
              }
            }}
          />
          <div className="operation-name">
            <label htmlFor="prune">Prune</label>
            <Tooltip
              open={pruneTooltipOpen}
              onOpenChange={(open) => setPruneTooltipOpen(open)}
            >
              <TooltipTrigger
                closeOnClick={false}
                render={
                  <button
                    type="button"
                    className="prune-info-trigger"
                    aria-label="About pruning availability"
                    onClick={() => setPruneTooltipOpen((open) => !open)}
                  >
                    ?
                  </button>
                }
              />
              <TooltipContent side="top">
                Pruning is available only for 16-bit, 24-bit, or 32-bit models.
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="prune-controls">
            <Select
              value={method || emptySelectValue}
              onValueChange={(value) => {
                setMethod(value === emptySelectValue ? '' : value || '');
                if (value === emptySelectValue) setRatio('');
              }}
              disabled={!prune}
            >
              <SelectTrigger
                aria-label="Pruning method"
                className="method-control"
              >
                <SelectValue>{methodLabel || 'Select'}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={emptySelectValue}>Select</SelectItem>
                {pruningMethods.map(([value, label]) => (
                  <SelectItem key={value} value={value}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {method === 'nm' ? (
              <output className="pattern-control" aria-label="Sparsity pattern">
                2:4
              </output>
            ) : (
              <label className="percentage-control" aria-disabled={!prune}>
                <input
                  aria-label="Pruning target percentage"
                  type="number"
                  min="1"
                  max="80"
                  value={ratio}
                  disabled={!prune}
                  onChange={(event) => setRatio(event.target.value)}
                />
                <span>%</span>
              </label>
            )}
          </div>
        </div>
        <div className="optimization-row">
          <Checkbox
            id="quantize"
            className="operation-checkbox"
            checked={quantize}
            onCheckedChange={(checked) => {
              const enabled = checked === true;
              setQuantize(enabled);
              if (!enabled) setBits('');
            }}
          />
          <label htmlFor="quantize">Quantize</label>
          <div className="quantize-controls" aria-disabled={!quantize}>
            <div className="optimization-control">
              <Select
                value={bits || emptySelectValue}
                onValueChange={(value) =>
                  setBits(value === emptySelectValue ? '' : value || '')
                }
                disabled={!quantize}
              >
                <SelectTrigger
                  aria-label="Output precision"
                  className="precision-control"
                >
                  <SelectValue>{bits ? `${bits}-bit` : 'Select'}</SelectValue>
                </SelectTrigger>
                <SelectContent
                  alignItemWithTrigger={false}
                  side="bottom"
                  collisionAvoidance={{
                    side: 'none',
                    align: 'shift',
                    fallbackAxisSide: 'none',
                  }}
                >
                  <SelectItem value={emptySelectValue}>Select</SelectItem>
                  {precisionOptions.map((value) => (
                    <SelectItem
                      key={value}
                      value={String(value)}
                      disabled={value === selected?.source_bits}
                    >
                      {value}-bit
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <output className="optimization-size">
              {quantize ? `~ ${size} GB` : 'Off'}
            </output>
          </div>
        </div>
      </div>
      <div className="panel-run-action">
        <button
          type="button"
          className="primary"
          disabled={!!jobId}
          onClick={() => requestOptimization('optimize')}
        >
          Run
        </button>
      </div>
      {needsCalibration && (
        <div className="calibration-field">
          <span>Calibration dataset</span>
          <div className="calibration-file-control">
            <button
              type="button"
              className="secondary choose-file-button"
              disabled={datasetBusy}
              onClick={() => calibrationInput.current?.click()}
            >
              Choose File
            </button>
            <input
              ref={calibrationInput}
              hidden
              type="file"
              accept=".jsonl,application/json"
              disabled={datasetBusy}
              onChange={(event) => void uploadDataset(event.target.files?.[0])}
            />
            {dataset && (
              <span className="chosen-file-name">{dataset.name}</span>
            )}
          </div>
          <small>{methodLabel} requires representative JSONL examples.</small>
        </div>
      )}
      {datasetError && (
        <p className="model-error" role="alert">
          {datasetError}
        </p>
      )}
    </>
  );

  const fineTunePanel = (
    <div className="fine-tune-panel">
      <div className="fine-tune-model-field">
        <span>Model</span>
        <button
          type="button"
          className="fine-tune-model-control"
          disabled={fineTuneModelBusy || !!jobId}
          onClick={() => void chooseFineTuneModel()}
        >
          <span>
            {selected?.path || 'Select original/pruned/quantized model'}
          </span>
          <FolderOpen size={18} aria-hidden="true" />
        </button>
      </div>
      {fineTuneModelError && (
        <p className="model-error" role="alert">
          {fineTuneModelError}
        </p>
      )}
      <div className="optimization-row fine-tune-method-row">
        <span className="operation-name">Method</span>
        <div className="fine-tune-method-control">
          <Select value={fineTuneMethod} onValueChange={(value) => setFineTuneMethod(value || 'lora')}>
            <SelectTrigger aria-label="Fine-Tuning method" className="method-control">
              <SelectValue>{fineTuneMethodLabel}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="lora">LoRA</SelectItem>
              <SelectItem value="qlora">QLoRA</SelectItem>
              <SelectItem value="full">Full Fine-Tuning</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="training-fields">
          <div className="training-row">
            <span>Training Dataset</span>
            <div className="training-file-control">
              <button
                type="button"
                className="secondary choose-file-button"
                disabled={datasetBusy}
                onClick={() => trainingInput.current?.click()}
              >
                Choose File
              </button>
              <input
                ref={trainingInput}
                hidden
                type="file"
                accept=".jsonl,application/json"
                disabled={datasetBusy}
                onChange={(event) =>
                  void uploadDataset(event.target.files?.[0])
                }
              />
              {dataset && (
                <span className="chosen-file-name">{dataset.name}</span>
              )}
            </div>
          </div>
          <label className="training-row compact-number-field">
            <span>Training Steps</span>
            <input
              type="number"
              min="1"
              max="5000"
              step="1"
              placeholder="Enter a value between 1 and 5000"
              value={iterations}
              onChange={(event) => setIterations(event.target.value)}
              onKeyDown={(event) => {
                if (
                  event.key === 'Enter' &&
                  selected &&
                  tune &&
                  !jobId &&
                  validFineTune
                ) {
                  event.preventDefault();
                  requestOptimization('fine-tune');
                }
              }}
            />
          </label>
          {datasetBusy && <p className="model-status">Validating dataset</p>}
          {dataset && (
            <div className="dataset-ready">
              <Check size={16} />
              <strong>{dataset.name}</strong>
              <span>{dataset.counts.train} training examples</span>
            </div>
          )}
          {datasetError && (
            <p className="model-error" role="alert">
              {datasetError}
            </p>
          )}
      </div>
      <div className="panel-run-action">
        <button
          type="button"
          className="primary"
          disabled={!!jobId}
          onClick={() => requestOptimization('fine-tune')}
        >
          Run
        </button>
      </div>
    </div>
  );

  const reviewPanel = (
    <div className="review-capsules">
      <div className="review-row">
        <span>Model/Size</span>
        <b>
          {selected?.name || 'Not selected'}/~{size} GB
        </b>
      </div>
      <div className="review-row">
        <span>{reviewPipelineLabel}</span>
        <b>{reviewPipelineValue || 'No operation selected'}</b>
      </div>
      {tune && (
        <div className="review-row">
          <span>Dataset</span>
          <b>{dataset?.name}</b>
        </div>
      )}
      {result && (
        <div className="review-row result-path">
          <span>Output</span>
          <b>{result.output_path}</b>
        </div>
      )}
    </div>
  );

  return (
    <main className="app design-2">
      <header className="app-header">
        <Link className="brand" href="/">
          <Box />
          <span className="brand-copy">
            <strong>FORGE</strong>
            <small>LLM Fine-Tuning Automated</small>
          </span>
        </Link>
        <span className="prototype">
          <i />
          {connected ? 'Local backend' : 'Backend offline'}
        </span>
      </header>
      <div className="workspace">
        <div className="workspace-heading">
          <h1>New Experiment</h1>
        </div>
        <div className="wizard">
          <div className="stepper">
            {steps.map((name, index) => (
              <button
                className={step === index ? 'active' : ''}
                aria-current={step === index ? 'step' : undefined}
                onClick={() => {
                  playTabClick();
                  setStep(index);
                  window.scrollTo(0, 0);
                }}
                key={name}
              >
                <b>{index + 1}</b>
                {name}
              </button>
            ))}
          </div>
          <div className="wizard-body compact-body">
            <div className="wizard-form">
              <h2>{steps[step]}</h2>
              {step === 0 ? (
                <ModelSource
                  selected={selected}
                  onSelected={selectModel}
                  operations={operations}
                  bits={Number(bits || selected?.source_bits || 16)}
                  fineTuneMethod={fineTuneMethod}
                />
              ) : step === 1 ? (
                optimizePanel
              ) : step === 2 ? (
                fineTunePanel
              ) : (
                reviewPanel
              )}
              {memory?.message && (
                <output className="memory-warning">
                  <p>{memory.message}</p>
                  <small>
                    Runtime estimate · available{' '}
                    {(memory.available_bytes / 1e9).toFixed(1)} GB
                  </small>
                </output>
              )}
              {memoryError && (
                <p role="alert" className="model-error">
                  {memoryError}
                </p>
              )}
            </div>
          </div>
        </div>
        <div className="secondary-guides">
          {step === 0 && <WorkflowNote />}
          {step === 1 && <OptimizationGuide />}
          {step === 2 && <FineTuneGuide />}
          {step === 3 && <EvaluationGuide />}
        </div>
      </div>
      <Dialog
        open={runOpen}
        onOpenChange={(open) => {
          if (!jobId && (!result || result.saved)) setRunOpen(open);
        }}
      >
        <DialogContent
          className="model-dialog progress-dialog"
          showCloseButton={!jobId && (!result || !!result.saved)}
          onKeyDown={(event) => {
            if (event.key !== 'Enter') return;
            if (jobPaused && jobId) {
              event.preventDefault();
              void resumeOptimization();
            } else if (result?.saved) {
              event.preventDefault();
              setRunOpen(false);
            }
          }}
        >
          <DialogTitle>
            {result
              ? result.saved
                ? 'Output Saved'
                : result.operations.includes('fine_tune')
                  ? 'Fine-tuning completed successfully.'
                  : 'Optimization completed successfully.'
              : jobPaused
                ? activeRunScope === 'fine-tune'
                  ? 'Fine-Tuning Paused'
                  : 'Optimization Paused'
                : activeRunScope === 'fine-tune'
                  ? 'Fine-Tuning'
                  : 'Running Optimization'}
          </DialogTitle>
          {result ? (
            result.saved || runError ? (
              <DialogDescription>
                {runError ||
                  (result.operations.includes('fine_tune')
                    ? 'The fine-tuned model was saved successfully.'
                    : 'The optimized model was saved successfully.')}
              </DialogDescription>
            ) : null
          ) : jobPaused ? (
            <DialogDescription>The operation is paused.</DialogDescription>
          ) : (
            <div className="operation-progress-row">
              <DialogDescription>{runError || runStatus}</DialogDescription>
              {jobId && !jobPaused && (
                <div className="indeterminate-progress" aria-hidden="true">
                  <i />
                </div>
              )}
            </div>
          )}
          {result?.saved && (
            <div className="output-location">
              <span>Output folder</span>
              <strong>{result.output_path}</strong>
            </div>
          )}
          {result && !result.saved && (
            <div className="output-name-group">
              <div className="output-name-control">
                <input
                  value={outputPath}
                  maxLength={2000}
                  placeholder="Enter output folder path"
                  aria-label="Output folder path"
                  onChange={(event) => setOutputPath(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !saveBusy) {
                      event.preventDefault();
                      void saveOutput();
                    }
                  }}
                />
                <button
                  type="button"
                  className="output-browse-button"
                  aria-label="Choose save location"
                  title="Choose save location"
                  disabled={saveBusy}
                  onClick={() => void browseOutputFolder()}
                >
                  <FolderOpen size={18} />
                </button>
              </div>
            </div>
          )}
          <div className="dialog-actions">
            {jobId && !jobPaused && (
              <button
                className="stop-button"
                onClick={() => void pauseOptimization()}
              >
                <Square size={14} />
                Stop Operation
              </button>
            )}
            {jobId && jobPaused && (
              <>
                <button
                  className="secondary"
                  onClick={() => void cancelOptimization()}
                >
                  Cancel
                </button>
                <button
                  className="primary"
                  onClick={() => void resumeOptimization()}
                >
                  Resume
                </button>
              </>
            )}
            {result && !result.saved && (
              <>
                <button
                  className="secondary"
                  disabled={saveBusy}
                  onClick={() => void discardCompletedOutput()}
                >
                  Cancel
                </button>
                <button
                  className="primary"
                  disabled={saveBusy}
                  onClick={() => void saveOutput()}
                >
                  Save Output
                </button>
              </>
            )}
            {!jobId && (!result || result.saved) && (
              <button className="primary" onClick={() => setRunOpen(false)}>
                Close
              </button>
            )}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!confirmation}
        onOpenChange={(open) => {
          if (!open) setConfirmation(null);
        }}
      >
        <DialogContent
          className="model-dialog confirmation-dialog"
          showCloseButton={false}
          onKeyDown={(event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            if (confirmation?.canRun) {
              const scope = confirmation.scope;
              const signature = confirmation.signature;
              setConfirmation(null);
              if (scope && signature) void startOptimization(scope, signature);
            } else {
              setConfirmation(null);
            }
          }}
        >
          <DialogTitle className="confirmation-title">
            {confirmation?.title}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Review the selected operations before continuing.
          </DialogDescription>
          <div className="confirmation-items">
            {confirmation?.items.map((item, index) => (
              <div
                className={`confirmation-item${confirmation.items.length === 1 ? ' is-single' : ''}`}
                key={`${item.title}-${index}`}
              >
                {confirmation.items.length > 1 && (
                  <span>{String(index + 1).padStart(2, '0')}</span>
                )}
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.message}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="dialog-actions confirmation-actions">
            <button
              type="button"
              className={confirmation?.canRun ? 'secondary' : 'primary'}
              onClick={() => setConfirmation(null)}
            >
              {confirmation?.canRun ? 'No' : 'OK'}
            </button>
            {confirmation?.canRun && (
              <button
                type="button"
                className="primary"
                onClick={() => {
                  const scope = confirmation?.scope;
                  const signature = confirmation?.signature;
                  setConfirmation(null);
                  if (scope && signature)
                    void startOptimization(scope, signature);
                }}
              >
                {confirmation?.kind === 'repeat' ? 'Yes' : 'Yes, Run'}
              </button>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </main>
  );
}
