export type RunScope = 'optimize' | 'fine-tune' | 'workflow';

type RunSignatureInput = {
  modelId: string;
  modelPath: string | null;
  prune: boolean;
  method: string;
  ratio: string;
  quantize: boolean;
  bits: string;
  fineTune: boolean;
  fineTuneMethod: string;
  iterations: string;
  datasetId: string;
};

export function getRunSignature(input: RunSignatureInput): string {
  return JSON.stringify(input);
}
