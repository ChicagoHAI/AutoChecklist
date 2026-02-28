"use client";

import { useGenerators, useRegistry } from "@/lib/hooks";
import { GENERATOR_CLASSES, ScorerConfig } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";

const thClass =
  "text-left px-3 py-2 text-xs font-semibold uppercase tracking-wider";
const thStyle = { color: "var(--text-secondary)", fontFamily: "var(--font-serif)" };

function TableWrapper({
  label,
  children,
  className = "mt-10",
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="mb-3">
        <h2
          className="text-sm font-semibold"
          style={{ color: "var(--text-secondary)" }}
        >
          {label}
        </h2>
        <div
          className="h-0.5 w-10 mt-1 rounded-md"
          style={{ backgroundColor: "var(--accent-primary)" }}
        />
      </div>
      <div
        className="border rounded-md overflow-hidden"
        style={{ borderColor: "var(--border)" }}
      >
        {children}
      </div>
    </div>
  );
}

export function GeneratorClassTable() {
  const classes = Object.entries(GENERATOR_CLASSES);

  return (
    <TableWrapper label="Generator Reference" className="">
      <table className="w-full text-sm">
        <thead>
          <tr
            style={{
              backgroundColor: "var(--surface-manila)",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <th className={thClass} style={thStyle}>Class</th>
            <th className={thClass} style={thStyle}>Level</th>
            <th className={thClass} style={thStyle}>Description</th>
            <th className={thClass} style={thStyle}>Pipelines</th>
          </tr>
        </thead>
        <tbody>
          {classes.map(([cls, info], i) => (
            <tr
              key={cls}
              style={{
                borderBottom:
                  i < classes.length - 1
                    ? "1px solid var(--border)"
                    : undefined,
                backgroundColor:
                  i % 2 === 0 ? "var(--surface)" : "var(--surface-elevated)",
              }}
            >
              <td
                className="px-3 py-2 font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                {info.label}
              </td>
              <td className="px-3 py-2">
                <Badge variant="info">
                  {info.level}
                </Badge>
              </td>
              <td
                className="px-3 py-2 text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                {info.description}
              </td>
              <td className="px-3 py-2">
                <div className="flex flex-wrap gap-1">
                  {info.methods.map((m) => (
                    <Badge key={m} variant="info">
                      {m}
                    </Badge>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableWrapper>
  );
}

/** Format a scorer config dict as a short summary string */
function formatScorerConfig(config: ScorerConfig | string | undefined): string {
  if (!config) return "—";
  if (typeof config === "string") return config;
  const parts: string[] = [config.mode, config.primary_metric];
  if (config.capture_reasoning) parts.push("reasoning");
  return parts.join(" / ");
}

export function PipelineReferenceTable() {
  const { generators, isLoading } = useGenerators();
  const { data: registry } = useRegistry();
  const defaultScorers = registry?.default_scorers ?? {};

  if (isLoading || generators.length === 0) return null;

  return (
    <TableWrapper label="Pipelines">
      <table className="w-full text-sm">
        <thead>
          <tr
            style={{
              backgroundColor: "var(--surface-manila)",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <th className={thClass} style={thStyle}>Pipeline</th>
            <th className={thClass} style={thStyle}>Class</th>
            <th className={thClass} style={thStyle}>Description</th>
            <th className={thClass} style={thStyle}>Default Scorer</th>
          </tr>
        </thead>
        <tbody>
          {generators.map((gen, i) => (
            <tr
              key={gen.name}
              style={{
                borderBottom:
                  i < generators.length - 1
                    ? "1px solid var(--border)"
                    : undefined,
                backgroundColor:
                  i % 2 === 0 ? "var(--surface)" : "var(--surface-elevated)",
              }}
            >
              <td
                className="px-3 py-2 font-medium"
                style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
              >
                {gen.name}
              </td>
              <td className="px-3 py-2">
                {gen.generator_class ? (
                  <Badge variant="info">
                    {GENERATOR_CLASSES[gen.generator_class]?.label ?? gen.generator_class}
                  </Badge>
                ) : (
                  <span style={{ color: "var(--text-tertiary)" }}>—</span>
                )}
              </td>
              <td
                className="px-3 py-2 text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                {gen.description}
                {gen.detail && (
                  <span style={{ color: "var(--text-tertiary)" }}>
                    {" "}{gen.detail}
                  </span>
                )}
              </td>
              <td
                className="px-3 py-2 text-xs"
                style={{
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {formatScorerConfig(defaultScorers[gen.name] as ScorerConfig | string | undefined)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableWrapper>
  );
}

/** Static scorer config reference explaining each setting */
export function ScorerReferenceTable() {
  const configRows = [
    {
      setting: "Mode",
      options: "batch, item",
      description:
        "Batch: scores all checklist items in 1 LLM call (fast, cheap). Item: scores 1 item per LLM call (slower, supports logprobs and per-item weights).",
    },
    {
      setting: "Primary Metric",
      options: "pass, weighted, normalized",
      description:
        "Controls which metric Score.primary_score returns. All 3 are always computed. Normalized is item-mode only and automatically uses logprobs.",
    },
    {
      setting: "Capture Reasoning",
      options: "on / off",
      description:
        "Includes per-item reasoning text in scorer output. Works in both batch and item mode.",
    },
    {
      setting: "Scorer Prompt",
      options: "(default), rlcf, rocketeval",
      description:
        "Prompt template for scoring. Default depends on mode. rlcf and rocketeval are paper-specific prompts used by their respective presets.",
    },
  ];

  const metricRows = [
    {
      metric: "pass_rate",
      computed: "yes_count / applicable_count",
      usage: "General-purpose. Simple pass/fail ratio.",
    },
    {
      metric: "weighted_score",
      computed: "Σ(weight × score) / Σ(weight)",
      usage: "When items have different importance weights.",
    },
    {
      metric: "normalized_score",
      computed: "Average logprob confidence",
      usage: "Item mode only. Confidence-based probability scores instead of binary yes/no.",
    },
  ];

  return (
    <>
      <TableWrapper label="Scorer Config Reference">
        <table className="w-full text-sm">
          <thead>
            <tr
              style={{
                backgroundColor: "var(--surface-manila)",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <th className={thClass} style={thStyle}>Setting</th>
              <th className={thClass} style={thStyle}>Options</th>
              <th className={thClass} style={thStyle}>Description</th>
            </tr>
          </thead>
          <tbody>
            {configRows.map((row, i) => (
              <tr
                key={row.setting}
                style={{
                  borderBottom:
                    i < configRows.length - 1
                      ? "1px solid var(--border)"
                      : undefined,
                  backgroundColor:
                    i % 2 === 0 ? "var(--surface)" : "var(--surface-elevated)",
                }}
              >
                <td
                  className="px-3 py-2 font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  {row.setting}
                </td>
                <td
                  className="px-3 py-2 text-xs"
                  style={{
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {row.options}
                </td>
                <td
                  className="px-3 py-2 text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {row.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableWrapper>

      <TableWrapper label="Scoring Metrics">
        <table className="w-full text-sm">
          <thead>
            <tr
              style={{
                backgroundColor: "var(--surface-manila)",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <th className={thClass} style={thStyle}>Metric</th>
              <th className={thClass} style={thStyle}>Computed As</th>
              <th className={thClass} style={thStyle}>When to Use</th>
            </tr>
          </thead>
          <tbody>
            {metricRows.map((row, i) => (
              <tr
                key={row.metric}
                style={{
                  borderBottom:
                    i < metricRows.length - 1
                      ? "1px solid var(--border)"
                      : undefined,
                  backgroundColor:
                    i % 2 === 0 ? "var(--surface)" : "var(--surface-elevated)",
                }}
              >
                <td
                  className="px-3 py-2 font-medium"
                  style={{
                    color: "var(--text-primary)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {row.metric}
                </td>
                <td
                  className="px-3 py-2 text-xs"
                  style={{
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {row.computed}
                </td>
                <td
                  className="px-3 py-2 text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {row.usage}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableWrapper>
    </>
  );
}

/** Combined reference tables for the home page Reference tab */
export function MethodReferenceTable() {
  return (
    <>
      <GeneratorClassTable />
      <PipelineReferenceTable />
      <ScorerReferenceTable />
    </>
  );
}
