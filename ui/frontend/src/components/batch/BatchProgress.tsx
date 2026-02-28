"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import type { BatchSummary } from "@/lib/types";

interface BatchProgressProps {
  batch: BatchSummary;
  onCancel: () => void;
  onViewResults: () => void;
  cancelling?: boolean;
}

const STATUS_BADGE: Record<
  string,
  { variant: "default" | "success" | "error" | "warning" | "info"; label: string }
> = {
  pending: { variant: "default", label: "Pending" },
  running: { variant: "info", label: "Running" },
  completed: { variant: "success", label: "Completed" },
  failed: { variant: "error", label: "Failed" },
  cancelled: { variant: "warning", label: "Cancelled" },
};

function formatElapsed(startedAt: string | null): string {
  if (!startedAt) return "--";
  const start = new Date(startedAt).getTime();
  const now = Date.now();
  const seconds = Math.floor((now - start) / 1000);

  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

export function BatchProgress({
  batch,
  onCancel,
  onViewResults,
  cancelling,
}: BatchProgressProps) {
  const [elapsed, setElapsed] = useState(() => formatElapsed(batch.started_at));
  const isActive = batch.status === "running" || batch.status === "pending";
  const isDone = batch.status === "completed";
  const percentage =
    batch.total > 0 ? Math.round((batch.completed / batch.total) * 100) : 0;
  const badgeInfo = STATUS_BADGE[batch.status] ?? STATUS_BADGE.pending;

  useEffect(() => {
    if (!isActive || !batch.started_at) return;

    const interval = setInterval(() => {
      setElapsed(formatElapsed(batch.started_at));
    }, 1000);

    return () => clearInterval(interval);
  }, [isActive, batch.started_at]);

  // Compute final elapsed when batch finishes (derived state, no effect needed)
  const finalElapsed = useMemo(() => {
    if (!isActive && batch.started_at && batch.completed_at) {
      const start = new Date(batch.started_at).getTime();
      const end = new Date(batch.completed_at).getTime();
      const seconds = Math.floor((end - start) / 1000);
      if (seconds < 60) return `${seconds}s`;
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s`;
    }
    return null;
  }, [isActive, batch.started_at, batch.completed_at]);

  const displayElapsed = finalElapsed ?? elapsed;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2
              className="text-base font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Batch Evaluation
            </h2>
            <Badge variant={badgeInfo.variant}>{badgeInfo.label}</Badge>
          </div>
          {batch.config?.filename && (
            <span
              className="text-xs"
              style={{
                color: "var(--text-tertiary)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {batch.config.filename}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Progress bar */}
        <div>
          <div className="flex justify-between mb-2">
            <span
              className="text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              {batch.completed} / {batch.total} items
            </span>
            <span
              className="text-sm font-medium"
              style={{
                color: "var(--text-primary)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {percentage}%
            </span>
          </div>
          <Progress value={batch.completed} max={batch.total} size="md" />
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-4 gap-4">
          <StatBlock label="Completed" value={String(batch.completed)} />
          <StatBlock
            label="Failed"
            value={String(batch.failed)}
            color={batch.failed > 0 ? "var(--error)" : undefined}
          />
          <StatBlock label="Elapsed" value={displayElapsed} />
          <StatBlock
            label="Mean Score"
            value={
              batch.mean_score != null
                ? `${(batch.mean_score * 100).toFixed(1)}%`
                : batch.macro_pass_rate != null
                  ? `${(batch.macro_pass_rate * 100).toFixed(1)}%`
                  : "--"
            }
          />
        </div>

        {/* Error message */}
        {batch.error && (
          <div
            className="text-sm rounded-md px-4 py-3"
            style={{
              backgroundColor: "var(--error-light)",
              color: "var(--error)",
            }}
          >
            {batch.error}
          </div>
        )}

        {/* Config info */}
        {batch.config && (
          <div
            className="text-xs flex items-center gap-4 flex-wrap"
            style={{ color: "var(--text-tertiary)" }}
          >
            <span>
              {batch.config.pipeline_name
                ? "Pipeline"
                : batch.config.pipeline_mode === "score_only" && batch.config.checklist_name
                ? "Checklist"
                : "Method"}
              :{" "}
              <strong style={{ color: "var(--text-secondary)" }}>
                {batch.config.pipeline_name
                  || batch.config.checklist_name
                  || batch.config.method?.toUpperCase()
                  || "--"}
              </strong>
            </span>
            {batch.config.pipeline_mode !== "score_only" && (
              <span>
                Generator:{" "}
                <strong style={{ color: "var(--text-secondary)" }}>
                  {batch.config.generator_model || batch.config.model || "--"}
                </strong>
              </span>
            )}
            {batch.config.pipeline_mode !== "generate_only" && (
              <span>
                Scorer:{" "}
                <strong style={{ color: "var(--text-secondary)" }}>
                  {batch.config.scorer_model || batch.config.model || "--"}
                </strong>
              </span>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          {isActive && (
            <Button
              variant="outline"
              onClick={onCancel}
              loading={cancelling}
              disabled={cancelling}
            >
              Cancel
            </Button>
          )}
          {isDone && (
            <Button onClick={onViewResults}>View Results</Button>
          )}
          {(batch.status === "failed" || batch.status === "cancelled") && (
            <Button variant="outline" onClick={onViewResults}>
              View Partial Results
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function StatBlock({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div>
      <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
        {label}
      </p>
      <p
        className="text-lg font-semibold mt-0.5"
        style={{
          color: color || "var(--text-primary)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {value}
      </p>
    </div>
  );
}
