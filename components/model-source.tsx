'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Folder,
  FolderOpen,
  Download,
  Square,
  Check,
  X,
  RotateCw,
} from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { backend, HardwareInfo, ModelInfo, MemoryCheck } from '@/lib/backend';
import { modelRequirementState } from '@/lib/model-requirements';

type Job = {
  id: string;
  kind?: string;
  payload?: { model_id?: string };
  status: string;
  progress?: string;
  error?: string;
  result?: ModelInfo;
  downloaded_bytes?: number;
  total_bytes?: number;
  progress_percent?: number;
};
type Review = { model: ModelInfo; memory: MemoryCheck; hardware: HardwareInfo };
type Notice = { title: string; message: string; retry?: boolean };
const terminal = new Set(['succeeded', 'failed', 'interrupted', 'cancelled']);
const wait = (signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(done, 800);
    function done() {
      signal.removeEventListener('abort', abort);
      resolve();
    }
    function abort() {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    }
    signal.addEventListener('abort', abort, { once: true });
  });

const requirementMessage = (checks: boolean[]) => {
  const failed = ['memory', 'compute', 'runtime'].filter(
    (_, index) => !checks[index],
  );
  if (!failed.length)
    return 'The memory, compute, and runtime requirements are met.';
  if (failed.length === 1) return `The ${failed[0]} requirement is not met.`;
  if (failed.length === 2)
    return `The ${failed[0]} and ${failed[1]} requirements are not met.`;
  return 'The memory, compute, and runtime requirements are not met.';
};

export function ModelSource({
  selected,
  onSelected,
  operations,
  bits,
  fineTuneMethod,
}: {
  selected: ModelInfo | null;
  onSelected: (model: ModelInfo | null) => void;
  operations: string[];
  bits: number;
  fineTuneMethod: string;
}) {
  const [source, setSource] = useState(
    selected?.source === 'huggingface' ? 'huggingface' : 'local',
  );
  const [localPath, setLocalPath] = useState(selected?.path || '');
  const [repo, setRepo] = useState(selected?.repo_id || '');
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<'idle' | 'checking' | 'downloading'>(
    'idle',
  );
  const [status, setStatus] = useState('');
  const [downloadProgress, setDownloadProgress] = useState({
    downloaded: 0,
    total: 0,
    percent: 0,
  });
  const [error, setError] = useState('');
  const [review, setReview] = useState<Review | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [downloadModel, setDownloadModel] = useState<ModelInfo | null>(null);
  const [downloadPaused, setDownloadPaused] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [folderBusy, setFolderBusy] = useState(false);
  const active = useRef<AbortController | null>(null);
  const reviewOk = useRef<HTMLButtonElement | null>(null);
  const noticeOk = useRef<HTMLButtonElement | null>(null);
  const downloadCommand = useRef(false);
  const interactionScroll = useRef(0);
  const restoreInteractionPosition = () => {
    const top = interactionScroll.current;
    requestAnimationFrame(() =>
      requestAnimationFrame(() => window.scrollTo(0, top)),
    );
  };

  /* oxlint-disable react/react-compiler -- synchronize a selection made in another tab */
  useEffect(() => {
    if (!selected?.path) return;
    // A model chosen from Fine-Tuning must also appear in the Model tab.
    setSource('local');
    setLocalPath(selected.path);
  }, [selected?.path]);
  /* oxlint-enable react/react-compiler */

  const updateDownloadProgress = useCallback((job: Job) => {
    if (
      typeof job.downloaded_bytes !== 'number' ||
      typeof job.total_bytes !== 'number' ||
      typeof job.progress_percent !== 'number'
    )
      return;
    setDownloadProgress({
      downloaded: job.downloaded_bytes,
      total: job.total_bytes,
      percent: job.progress_percent,
    });
  }, []);

  const finishJob = useCallback(
    async (id: string, controller: AbortController) => {
      let job = await backend<Job>(`/jobs/${id}`, {
        signal: controller.signal,
      });
      updateDownloadProgress(job);
      while (!terminal.has(job.status)) {
        setStatus(job.progress || 'Downloading model files');
        await wait(controller.signal);
        job = await backend<Job>(`/jobs/${id}`, { signal: controller.signal });
        updateDownloadProgress(job);
      }
      setJobId(null);
      setBusy(false);
      setPhase('idle');
      if (job.status === 'succeeded' && job.result) {
        setDownloadModel(null);
        setDownloadPaused(false);
        onSelected(job.result);
        setStatus('Model ready');
        return;
      }
      if (job.status === 'cancelled') {
        setStatus('Download stopped');
        setDownloadPaused(true);
        return;
      }
      if (job.status === 'interrupted') {
        setStatus('');
        setNotice({
          title: 'Worker restarted',
          message:
            'The model worker restarted automatically. Retry the download.',
          retry: true,
        });
        return;
      }
      throw new Error(job.error || 'Download did not complete.');
    },
    [onSelected, updateDownloadProgress],
  );

  useEffect(() => {
    const controller = new AbortController();
    active.current = controller;
    backend<Job | null>('/jobs', { signal: controller.signal })
      .then(async (job) => {
        if (!job || job.kind !== 'download' || terminal.has(job.status)) return;
        if (job.payload?.model_id) {
          const models = await backend<ModelInfo[]>('/models', {
            signal: controller.signal,
          });
          setDownloadModel(
            models.find((model) => model.id === job.payload?.model_id) || null,
          );
        }
        setJobId(job.id);
        setBusy(true);
        setPhase('downloading');
        setStatus(job.progress || 'Downloading model files');
        updateDownloadProgress(job);
        void finishJob(job.id, controller).catch((e) => {
          if ((e as Error).name !== 'AbortError') {
            setBusy(false);
            setPhase('idle');
            setError((e as Error).message);
          }
        });
      })
      // Active-job recovery is optional. The page health check reports outages.
      .catch(() => {});
    return () => controller.abort();
  }, [finishJob, updateDownloadProgress]);

  const choose = async () => {
    setFolderBusy(true);
    setError('');
    try {
      const result = await backend<{ path: string | null }>('/folders/pick', {
        method: 'POST',
      });
      if (result.path) {
        setLocalPath(result.path);
        resetResult();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setFolderBusy(false);
    }
  };
  const resetResult = () => {
    setReview(null);
    setNotice(null);
    setError('');
    setStatus('');
    onSelected(null);
  };

  const inspect = async (localPathOverride?: string) => {
    interactionScroll.current = window.scrollY;
    if (source === 'huggingface') {
      const value = repo.trim();
      if (value.toLowerCase().endsWith('.gguf')) {
        setError(
          'Enter the repository ID, not a GGUF filename. Example: unsloth/Qwen3.8-27B-GGUF.',
        );
        return;
      }
      if (
        !value.startsWith('https://huggingface.co/') &&
        !/^[^/\s]+\/[^/\s]+$/.test(value)
      ) {
        setError(
          'Enter owner/model-name or paste the Hugging Face model link.',
        );
        return;
      }
    }
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setPhase('checking');
    setError('');
    setStatus('Checking model');
    setReview(null);
    onSelected(null);
    try {
      const result = await backend<Review>('/models/inspect', {
        method: 'POST',
        body: JSON.stringify({
          source,
          value: source === 'local' ? localPathOverride || localPath : repo,
          operations,
          bits,
          fine_tune_method: fineTuneMethod,
        }),
        signal: controller.signal,
      });
      setReview(result);
      setStatus('');
      restoreInteractionPosition();
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setError((e as Error).message);
        setStatus('');
        onSelected(null);
      }
    } finally {
      if (!controller.signal.aborted) {
        setBusy(false);
        setPhase('idle');
      }
    }
  };

  const startDownload = async (
    modelOverride?: ModelInfo,
    destination?: string,
  ) => {
    const model = modelOverride || review?.model || downloadModel;
    if (!model || downloadCommand.current) return;
    downloadCommand.current = true;
    setReview(null);
    setNotice(null);
    setError('');
    setDownloadModel(model);
    setDownloadPaused(false);
    setBusy(true);
    setPhase('downloading');
    setStatus('Starting download');
    setDownloadProgress({
      downloaded: 0,
      total: model.download_bytes,
      percent: 0,
    });
    const controller = new AbortController();
    active.current = controller;
    try {
      const job = await backend<Job>(`/models/${model.id}/download`, {
        method: 'POST',
        body: destination ? JSON.stringify({ destination }) : undefined,
        signal: controller.signal,
      });
      downloadCommand.current = false;
      if (job.status === 'succeeded' && job.result) {
        onSelected(job.result);
        setStatus('Model ready');
        setDownloadModel(null);
        setBusy(false);
        setPhase('idle');
        return;
      }
      setJobId(job.id);
      await finishJob(job.id, controller);
    } catch (e) {
      downloadCommand.current = false;
      if ((e as Error).name !== 'AbortError') {
        setBusy(false);
        setPhase('idle');
        setError((e as Error).message);
        if ((e as Error).message.includes('already running')) {
          const running = await backend<Job | null>('/jobs');
          if (running?.kind === 'download') {
            setJobId(running.id);
            setBusy(true);
            setPhase('downloading');
            setStatus(running.progress || 'Model operation running');
            updateDownloadProgress(running);
            void finishJob(running.id, controller).catch((jobError) => {
              if ((jobError as Error).name !== 'AbortError') {
                setBusy(false);
                setPhase('idle');
                setError((jobError as Error).message);
              }
            });
          }
          setNotice({
            title: 'Another operation is running',
            message:
              'Open the running operation or stop it before starting this download.',
          });
        }
      }
    }
  };

  const chooseDownloadDestination = async (model: ModelInfo) => {
    setFolderBusy(true);
    setError('');
    try {
      const result = await backend<{ path: string | null }>(
        '/folders/pick-download',
        { method: 'POST' },
      );
      if (result.path) await startDownload(model, result.path);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setFolderBusy(false);
    }
  };

  const requirementState = modelRequirementState(review);
  const requirementChecks = requirementState.checks;
  const requirementsMet = requirementState.requirementsMet;

  const stopDownload = async () => {
    if (!jobId || downloadCommand.current) return;
    downloadCommand.current = true;
    try {
      await backend<Job>(`/jobs/${jobId}/cancel`, { method: 'POST' });
      active.current?.abort();
      setJobId(null);
      setBusy(false);
      setPhase('idle');
      setStatus('Download stopped');
      setDownloadPaused(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      downloadCommand.current = false;
    }
  };

  const cancelDownload = async () => {
    if (downloadCommand.current) return;
    downloadCommand.current = true;
    active.current?.abort();
    try {
      if (jobId) {
        await backend<Job>(`/jobs/${jobId}/cancel`, { method: 'POST' });
      }
      if (downloadModel && !downloadModel.downloaded) {
        await backend(`/models/${downloadModel.id}/download/discard`, {
          method: 'POST',
        });
      }
    } catch (e) {
      setError((e as Error).message);
      downloadCommand.current = false;
      return;
    }
    setJobId(null);
    setBusy(false);
    setPhase('idle');
    setStatus('');
    setDownloadModel(null);
    setDownloadPaused(false);
    setDownloadProgress({ downloaded: 0, total: 0, percent: 0 });
    downloadCommand.current = false;
  };

  return (
    <div className="model-source-shell">
      <Tabs
        value={source}
        onValueChange={(value) => {
          if (!busy) {
            setSource(String(value));
            resetResult();
          }
        }}
      >
        <TabsList className="source-tabs">
          <TabsTrigger
            value="local"
            disabled={busy || folderBusy}
            onClick={() => void choose()}
          >
            <Folder />
            Local folder
          </TabsTrigger>
          <TabsTrigger value="huggingface" disabled={busy}>
            <Download />
            Hugging Face
          </TabsTrigger>
        </TabsList>
        <TabsContent value="local">
          <div className="local-folder-field">
            <div className="path-input-control">
              <input
                aria-label="Model folder path"
                placeholder="Enter Model Folder Path"
                value={localPath}
                disabled={busy || folderBusy}
                onChange={(e) => {
                  setLocalPath(e.target.value);
                  resetResult();
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && localPath.trim()) {
                    event.preventDefault();
                    void inspect();
                  }
                }}
              />
              <button
                className="path-browse-button"
                type="button"
                aria-label="Browse model folder"
                disabled={busy || folderBusy}
                onClick={() => void choose()}
              >
                <FolderOpen size={16} aria-hidden="true" />
              </button>
            </div>
            <button
              className="primary folder-picker-button model-source-action-button"
              type="button"
              disabled={busy || folderBusy || !localPath.trim()}
              onClick={() => void inspect()}
            >
              {phase === 'checking' && (
                <RotateCw
                  className="compatibility-spinner"
                  size={15}
                  aria-hidden="true"
                />
              )}
              <span>Check Compatibility</span>
            </button>
          </div>
        </TabsContent>
        <TabsContent value="huggingface">
          <div className="local-folder-field hugging-face-field">
            <input
              aria-label="Hugging Face model link or ID"
              aria-invalid={!!error}
              aria-describedby={error ? 'model-source-error' : undefined}
              placeholder="Owner/Model Name or URL"
              value={repo}
              disabled={busy}
              onChange={(e) => {
                setRepo(e.target.value);
                resetResult();
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && repo.trim()) {
                  event.preventDefault();
                  void inspect();
                }
              }}
            />
            <button
              className="primary folder-picker-button model-source-action-button"
              type="button"
              disabled={busy || !repo.trim()}
              onClick={() => void inspect()}
            >
              {phase === 'checking' && (
                <RotateCw
                  className="compatibility-spinner"
                  size={15}
                  aria-hidden="true"
                />
              )}
              <span>Check Compatibility</span>
            </button>
          </div>
        </TabsContent>
      </Tabs>
      {(phase === 'downloading' || downloadPaused) && (
        <div className="model-action">
          {phase === 'downloading' ? (
            <div className="download-actions">
              <button
                className="stop-button"
                disabled={!jobId}
                onClick={stopDownload}
              >
                <Square size={14} />
                Stop download
              </button>
              <button
                className="secondary"
                disabled={!jobId}
                onClick={cancelDownload}
              >
                Cancel
              </button>
            </div>
          ) : downloadPaused && downloadModel ? (
            <div className="download-actions">
              <button
                className="primary"
                onClick={() => void startDownload(downloadModel)}
              >
                <Download size={16} />
                Resume download
              </button>
              <button className="secondary" onClick={cancelDownload}>
                Cancel
              </button>
            </div>
          ) : null}
        </div>
      )}
      <div className="model-feedback" aria-live="polite">
        {(phase === 'downloading' || downloadPaused) && (
          <output className="inline-download">
            <span>{status || 'Downloading model files'}</span>
            <div className="inline-download-meter" aria-live="off">
              <progress
                className={`download-progress-track${downloadPaused ? ' is-paused' : ''}`}
                aria-label={
                  downloadPaused ? 'Download stopped' : 'Download in progress'
                }
                aria-valuetext={`${downloadProgress.percent}%`}
                max={100}
                value={downloadProgress.percent}
              />
              <span className="download-progress-percent">
                {downloadProgress.percent}%
              </span>
            </div>
          </output>
        )}
        {status && phase !== 'downloading' && !downloadPaused && (
          <output className="model-status">{status}</output>
        )}
        {error && (
          <p id="model-source-error" role="alert" className="model-error">
            {error}
          </p>
        )}
        {selected?.downloaded && (
          <div className="selected-model">
            <strong>{selected.name}</strong>
            <span>
              {(selected.weight_bytes / 1e9).toFixed(2)} GB
              {selected.source_bits ? ` · ${selected.source_bits}-bit` : ''}
            </span>
          </div>
        )}
      </div>
      <Dialog
        open={!!review}
        onOpenChange={(open) => {
          if (!open) setReview(null);
        }}
      >
        <DialogContent
          className="model-dialog model-check-dialog"
          initialFocus={reviewOk}
          showCloseButton={false}
        >
          <DialogTitle className="model-check-message">
            {requirementMessage(requirementChecks)}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Model memory, compute, and runtime requirement results.
          </DialogDescription>
          {review && (
            <div className="model-check-bars">
              <div className="model-check-bar">
                <span>Memory Requirement</span>
                <strong>
                  {review.memory.required_gb.toFixed(1)} GB required /{' '}
                  {(review.memory.available_bytes / 1e9).toFixed(1)} GB
                  available
                </strong>
                <i
                  className={requirementChecks[0] ? 'is-passed' : 'is-failed'}
                  aria-label={requirementChecks[0] ? 'Passed' : 'Failed'}
                >
                  {requirementChecks[0] ? <Check /> : <X />}
                </i>
              </div>
              <div className="model-check-bar">
                <span>Compute Requirement</span>
                <strong>{requirementState.compute}</strong>
                <i
                  className={requirementChecks[1] ? 'is-passed' : 'is-failed'}
                  aria-label={requirementChecks[1] ? 'Passed' : 'Failed'}
                >
                  {requirementChecks[1] ? <Check /> : <X />}
                </i>
              </div>
              <div className="model-check-bar">
                <span>Runtime Requirement</span>
                <strong>{requirementState.runtime}</strong>
                <i
                  className={requirementChecks[2] ? 'is-passed' : 'is-failed'}
                  aria-label={requirementChecks[2] ? 'Passed' : 'Failed'}
                >
                  {requirementChecks[2] ? <Check /> : <X />}
                </i>
              </div>
            </div>
          )}
          <div className="model-check-actions">
            <button
              ref={reviewOk}
              className="primary model-check-ok"
              onClick={() => {
                if (
                  requirementsMet &&
                  review &&
                  (source === 'local' || review.model.downloaded)
                ) {
                  onSelected(review.model);
                  setStatus('Model ready');
                }
                setReview(null);
              }}
            >
              OK
            </button>
            {source === 'huggingface' &&
              requirementsMet &&
              review &&
              !review.model.downloaded && (
                <button
                  className="secondary model-check-download"
                  disabled={folderBusy}
                  onClick={() => void chooseDownloadDestination(review.model)}
                >
                  <Download size={16} />
                  Download
                </button>
              )}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!notice}
        onOpenChange={(open) => {
          if (!open) setNotice(null);
        }}
      >
        <DialogContent
          className="model-dialog model-check-dialog model-error-dialog"
          initialFocus={noticeOk}
          showCloseButton={false}
        >
          <DialogTitle className="model-check-message">
            {notice?.title}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Model operation error details.
          </DialogDescription>
          <div className="model-check-bar model-error-detail">
            <span>Reason</span>
            <strong>{notice?.message}</strong>
            <i className="is-failed" aria-label="Error">
              <X />
            </i>
          </div>
          <div className="model-check-actions">
            <button
              ref={noticeOk}
              className="primary model-check-ok"
              onClick={() => setNotice(null)}
            >
              OK
            </button>
            {notice?.retry && (
              <button
                className="secondary model-check-download"
                onClick={() => {
                  setNotice(null);
                  void inspect();
                }}
              >
                <RotateCw size={16} />
                Retry
              </button>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
