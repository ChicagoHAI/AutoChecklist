"use client";

import type { ScorerConfig } from "@/lib/types";

interface ScorerConfigPanelProps {
  value: ScorerConfig;
  onChange: (config: ScorerConfig) => void;
  disabled?: boolean;
}

function Toggle({
  options,
  value,
  onChange,
  disabled,
}: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <div
      className="inline-flex rounded-md p-0.5"
      style={{ backgroundColor: "var(--surface-elevated)" }}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(opt.value)}
          className="px-2.5 py-1 text-xs font-medium rounded transition-all disabled:opacity-50"
          style={{
            backgroundColor: value === opt.value ? "white" : "transparent",
            color:
              value === opt.value
                ? "var(--accent-primary)"
                : "var(--text-secondary)",
            boxShadow:
              value === opt.value ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function ScorerConfigPanel({
  value,
  onChange,
  disabled,
}: ScorerConfigPanelProps) {
  const metricOptions =
    value.mode === "batch"
      ? [
          { value: "pass", label: "Pass Rate" },
          { value: "weighted", label: "Weighted" },
        ]
      : [
          { value: "pass", label: "Pass Rate" },
          { value: "weighted", label: "Weighted" },
          { value: "normalized", label: "Normalized" },
        ];

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-1.5">
        <span
          className="text-xs font-medium"
          style={{ color: "var(--text-tertiary)" }}
        >
          Mode
        </span>
        <Toggle
          options={[
            { value: "batch", label: "Batch" },
            { value: "item", label: "Item" },
          ]}
          value={value.mode}
          disabled={disabled}
          onChange={(mode) => {
            const m = mode as "batch" | "item";
            // Auto-switch normalized → pass when switching to batch
            const metric =
              m === "batch" && value.primary_metric === "normalized"
                ? "pass"
                : value.primary_metric;
            onChange({ ...value, mode: m, primary_metric: metric });
          }}
        />
      </div>

      <div className="flex items-center gap-1.5">
        <span
          className="text-xs font-medium"
          style={{ color: "var(--text-tertiary)" }}
        >
          Metric
        </span>
        <Toggle
          options={metricOptions}
          value={value.primary_metric}
          disabled={disabled}
          onChange={(metric) =>
            onChange({
              ...value,
              primary_metric: metric as "pass" | "weighted" | "normalized",
            })
          }
        />
      </div>

      <label
        className="flex items-center gap-1.5 text-xs cursor-pointer"
        style={{ color: "var(--text-tertiary)" }}
      >
        <input
          type="checkbox"
          checked={value.capture_reasoning}
          disabled={disabled}
          onChange={(e) =>
            onChange({ ...value, capture_reasoning: e.target.checked })
          }
          className="rounded"
          style={{ accentColor: "var(--accent-primary)" }}
        />
        Reasoning
      </label>
    </div>
  );
}
