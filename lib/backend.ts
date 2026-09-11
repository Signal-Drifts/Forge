export const BACKEND = 'http://127.0.0.1:8000';
export type ModelInfo = {
  id: string;
  source: 'local' | 'huggingface';
  name: string;
  path: string | null;
  repo_id?: string;
  downloaded: boolean;
  compatible: boolean;
  compatibility_error: string | null;
  model_format: string;
  model_format_name: string;
  parameter_count: number | null;
  source_bits: number | null;
  weight_bytes: number;
  download_bytes: number;
};
export type MemoryCheck = {
  message: string | null;
  insufficient_memory: boolean;
  required_gb: number;
  available_bytes: number;
  output_weight_bytes_estimate: number;
  estimate_only: boolean;
};
export type HardwareInfo = {
  chip: string;
  architecture: string;
  os: string;
  gpu: string;
  gpu_memory_bytes: number | null;
  graphics_requirement: string;
  engine_id: string | null;
  engine: string;
  ram_total_bytes: number;
  ram_available_bytes: number;
  disk_free_bytes: number;
};
export type DatasetInfo = {
  id: string;
  name: string;
  counts: { train: number; validation: number; test: number };
  duplicates_removed: number;
};
export type OptimizationResult = {
  model: ModelInfo;
  output_path: string;
  adapter_path: string | null;
  operations: string[];
  saved?: boolean;
};
export type Job<T = unknown> = {
  id: string;
  kind: string;
  status: string;
  progress?: string;
  error?: string;
  result?: T;
};
export async function backend<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    const headers = new Headers(options.headers);
    if (!(options.body instanceof FormData) && !headers.has('Content-Type'))
      headers.set('Content-Type', 'application/json');
    response = await fetch(BACKEND + path, {
      ...options,
      headers,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError')
      throw error;
    throw new Error(
      'Local backend is unavailable. Start backend/run.sh and retry.',
    );
  }
  let body: { detail?: unknown };
  try {
    body = (await response.json()) as { detail?: unknown };
  } catch {
    throw new Error(
      response.ok
        ? 'The backend returned an invalid response.'
        : 'The request could not be completed. Please try later.',
    );
  }
  if (!response.ok)
    throw new Error(
      typeof body.detail === 'string'
        ? body.detail
        : 'The request could not be completed. Please try later.',
    );
  return body as T;
}
