"use client";


const COMMON_MODELS = [
  "openai/gpt-5-mini",
  "openai/gpt-4o-mini",
  "openai/gpt-5",
  "anthropic/claude-sonnet-4",
  "anthropic/claude-3.5-sonnet",
  "google/gemini-2.0-flash",
  "meta-llama/llama-3.1-8b-instruct",
  "gpt-4o-mini",
  "gpt-4o",
];

const PROVIDERS = [
  { value: "openrouter", label: "OpenRouter" },
  { value: "openai", label: "OpenAI" },
  { value: "vllm", label: "vLLM" },
];

interface ModelSelectorProps {
  provider: string;
  model: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (model: string) => void;
}

export function ModelSelector({
  provider,
  model,
  onProviderChange,
  onModelChange,
}: ModelSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <select
        value={provider}
        onChange={(e) => onProviderChange(e.target.value)}
        className="px-2 py-1.5 text-xs rounded-md border bg-white"
        style={{
          borderColor: "var(--border-strong)",
          color: "var(--text-secondary)",
          fontFamily: "var(--font-sans)",
        }}
      >
        {PROVIDERS.map((p) => (
          <option key={p.value} value={p.value}>
            {p.label}
          </option>
        ))}
      </select>
      <div className="relative">
        <input
          type="text"
          list="model-suggestions"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
          placeholder="Model name..."
          className="px-2 py-1.5 text-xs rounded-md border bg-white w-48"
          style={{
            borderColor: "var(--border-strong)",
            color: "var(--text-primary)",
            fontFamily: "var(--font-mono)",
          }}
        />
        <datalist id="model-suggestions">
          {COMMON_MODELS.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </div>
    </div>
  );
}
