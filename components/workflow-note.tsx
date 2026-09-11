import { DisclosureGuide } from '@/components/disclosure-guide';

const steps = [
  {
    title: 'Choose a Base Model',
    description:
      'Select an open-source model family such as Llama, Qwen, or Mistral.',
    url: 'https://huggingface.co/docs/transformers/main/en/models',
  },
  {
    title: 'Apply PEFT or LoRA',
    description:
      'Freeze the base model and train a small set of adapter parameters to reduce memory use.',
    url: 'https://huggingface.co/docs/peft/main/en/conceptual_guides/lora',
  },
  {
    title: 'Prepare the Dataset',
    description:
      'Gather instruction-response pairs, conversations, or classification labels in JSONL files.',
    url: 'https://huggingface.co/docs/datasets/loading',
  },
  {
    title: 'Train',
    description:
      'Run Transformers and PEFT, or use an optimized framework for smaller hardware.',
    url: 'https://huggingface.co/docs/transformers/training',
  },
  {
    title: 'Export',
    description:
      'Load the adapter directly or merge it into the model for supported inference engines.',
    url: 'https://huggingface.co/docs/peft/main/en/developer_guides/model_merging',
  },
] as const;

const formats = [
  [
    'Safetensors · .safetensors',
    'A secure and fast tensor serialization format for model weights.',
    'https://huggingface.co/docs/safetensors/index',
  ],
  [
    'PyTorch · .bin, .pt, .pth',
    'Native PyTorch checkpoint files containing model weights or training state.',
    'https://pytorch.org/tutorials/beginner/saving_loading_models.html',
  ],
  [
    'GGUF · GPT-Generated Unified Format · .gguf',
    'A portable single-file model format used by llama.cpp and Ollama.',
    'https://github.com/ggml-org/ggml/blob/master/docs/gguf.md',
  ],
  [
    'ONNX · Open Neural Network Exchange · .onnx',
    'A portable graph format for moving models between supported runtimes.',
    'https://onnx.ai/onnx/intro/',
  ],
  [
    'MLX · .safetensors + config.json',
    'Apple’s machine-learning framework uses Safetensors weights and configuration files.',
    'https://ml-explore.github.io/mlx/build/html/index.html',
  ],
  [
    'TensorFlow SavedModel · saved_model.pb',
    'A model directory containing a TensorFlow program, variables, and signatures.',
    'https://www.tensorflow.org/guide/saved_model',
  ],
  [
    'NVIDIA TensorRT · .engine, .plan',
    'A compiled inference engine optimized for a specific NVIDIA GPU environment.',
    'https://docs.nvidia.com/deeplearning/tensorrt/latest/',
  ],
] as const;

function LearnMore({ url, label }: { url: string; label: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`Learn more about ${label}`}
    >
      Learn More
    </a>
  );
}

export function WorkflowNote() {
  return (
    <>
      <DisclosureGuide
        title="Typical Fine-Tuning Workflow"
        countLabel="5 items"
      >
        <ol className="reference-bars">
          {steps.map((step, index) => (
            <li className="reference-bar" key={step.title}>
              <span className="workflow-index">
                {String(index + 1).padStart(2, '0')}
              </span>
              <div className="workflow-copy">
                <strong>{step.title}</strong>
                <p>{step.description}</p>
              </div>
              <LearnMore url={step.url} label={step.title} />
            </li>
          ))}
        </ol>
      </DisclosureGuide>
      <DisclosureGuide title="Model Formats" countLabel="7 items">
        <div className="reference-bars format-reference">
          {formats.map(([name, description, url], index) => (
            <div className="reference-bar" key={name}>
              <span className="workflow-index">
                {String(index + 1).padStart(2, '0')}
              </span>
              <div className="workflow-copy">
                <strong>{name}</strong>
                <p>{description}</p>
              </div>
              <LearnMore url={url} label={name} />
            </div>
          ))}
        </div>
      </DisclosureGuide>
    </>
  );
}
