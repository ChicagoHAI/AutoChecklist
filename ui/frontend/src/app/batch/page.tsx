"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { PageHeader } from "@/components/layout/PageHeader";
import { BatchUpload } from "@/components/batch/BatchUpload";
import { useBatches } from "@/lib/hooks";
import { uploadBatch, uploadBatchFromPath, startBatch, deleteBatch } from "@/lib/api";
import type { BatchStartParams } from "@/components/batch/BatchUpload";
import type { BatchSummary } from "@/lib/types";

const STATUS_BADGE_VARIANT: Record<
  string,
  "default" | "success" | "error" | "warning" | "info"
> = {
  pending: "default",
  running: "info",
  completed: "success",
  failed: "error",
  cancelled: "warning",
};

export default function BatchPage() {
  const router = useRouter();
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const queryClient = useQueryClient();
  const { data: batches, isLoading: batchesLoading } = useBatches();

  const handleUploadAndStart = useCallback(
    async (params: BatchStartParams) => {
      if (!params.file) return;
      setUploading(true);
      setUploadError(null);

      try {
        const uploadResult = await uploadBatch(
          params.file, params.method, params.scorer, params.pipelineMode,
          params.checklistId, params.customPrompt, params.pipelineId,
          params.scorerConfig
        );
        await startBatch(
          uploadResult.batch_id, params.provider, undefined,
          params.generatorModel, params.scorerModel, params.candidateModel
        );

        queryClient.invalidateQueries({ queryKey: ["batches"] });
        router.push(`/batch/${uploadResult.batch_id}`);
      } catch (err) {
        setUploadError(
          err instanceof Error ? err.message : "Upload failed. Please try again."
        );
      } finally {
        setUploading(false);
      }
    },
    [queryClient, router]
  );

  const handlePathAndStart = useCallback(
    async (params: BatchStartParams) => {
      if (!params.filePath) return;
      setUploading(true);
      setUploadError(null);

      try {
        const uploadResult = await uploadBatchFromPath(
          params.filePath, params.method, params.scorer, params.pipelineMode,
          params.checklistId, params.customPrompt, params.pipelineId,
          params.scorerConfig
        );
        await startBatch(
          uploadResult.batch_id, params.provider, undefined,
          params.generatorModel, params.scorerModel, params.candidateModel
        );

        queryClient.invalidateQueries({ queryKey: ["batches"] });
        router.push(`/batch/${uploadResult.batch_id}`);
      } catch (err) {
        setUploadError(
          err instanceof Error ? err.message : "Failed to read file. Check the path and try again."
        );
      } finally {
        setUploading(false);
      }
    },
    [queryClient, router]
  );

  const handleSelectBatch = useCallback(
    (batch: BatchSummary) => {
      router.push(`/batch/${batch.batch_id}`);
    },
    [router]
  );

  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const handleDeleteBatch = useCallback(
    async (batchId: string) => {
      try {
        await deleteBatch(batchId);
        queryClient.invalidateQueries({ queryKey: ["batches"] });
      } catch {
        // Silently ignore — batch may already be deleted
      } finally {
        setDeleteTarget(null);
      }
    },
    [queryClient]
  );

  return (
    <div>
      <PageHeader description="Run evaluations on datasets with built-in or custom pipelines and view results." />
      <div className="space-y-6">
        <BatchUpload
          onUploadAndStart={handleUploadAndStart}
          onPathAndStart={handlePathAndStart}
          loading={uploading}
        />

        {uploadError && (
          <div
            className="text-sm rounded-md px-4 py-3"
            style={{
              backgroundColor: "var(--error-light)",
              color: "var(--error)",
            }}
          >
            {uploadError}
          </div>
        )}

        {/* Past batches list */}
        <BatchList
          batches={batches ?? []}
          loading={batchesLoading}
          onSelect={handleSelectBatch}
          onDelete={setDeleteTarget}
        />
      </div>

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <DialogContent title="Delete Batch" description="This action cannot be undone.">
          <p
            className="text-sm mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Are you sure you want to permanently delete this batch and its results?
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button
              className="bg-[var(--error)] hover:bg-[var(--error)] text-white"
              onClick={() => {
                if (deleteTarget) handleDeleteBatch(deleteTarget);
              }}
            >
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function BatchList({
  batches,
  loading,
  onSelect,
  onDelete,
}: {
  batches: BatchSummary[];
  loading: boolean;
  onSelect: (batch: BatchSummary) => void;
  onDelete: (batchId: string) => void;
}) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <h2
            className="text-base font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            Recent Batches
          </h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-12 rounded-md animate-pulse"
                style={{ backgroundColor: "var(--surface-elevated)" }}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (batches.length === 0) {
    return null;
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "--";
    const d = new Date(dateStr);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <Card>
      <CardHeader>
        <h2
          className="text-base font-semibold"
          style={{ color: "var(--text-primary)" }}
        >
          Recent Batches
        </h2>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr
                style={{
                  borderBottom: "1px solid var(--border)",
                  backgroundColor: "var(--surface-elevated)",
                }}
              >
                <th className="text-left px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  File
                </th>
                <th className="text-left px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Pipeline / Checklist
                </th>
                <th className="text-left px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Generator Model
                </th>
                <th className="text-left px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Scorer Model
                </th>
                <th className="text-left px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Status
                </th>
                <th className="text-right px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Progress
                </th>
                <th className="text-right px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Mean Score
                </th>
                <th className="text-right px-5 py-3 text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
                  Started
                </th>
                <th className="w-10 px-2 py-3" />
              </tr>
            </thead>
            <tbody>
              {batches.map((batch) => (
                <tr
                  key={batch.batch_id}
                  onClick={() => onSelect(batch)}
                  className="cursor-pointer hover:bg-[var(--surface-elevated)] transition-colors"
                  style={{ borderBottom: "1px solid var(--border)" }}
                >
                  <td
                    className="px-5 py-3"
                    style={{
                      color: "var(--text-primary)",
                      fontFamily: "var(--font-mono)",
                      fontSize: "12px",
                    }}
                  >
                    {batch.config?.filename ?? batch.batch_id.slice(0, 8)}
                  </td>
                  <td
                    className="px-5 py-3"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {batch.config?.pipeline_name
                      ? batch.config.pipeline_name
                      : batch.config?.pipeline_mode === "score_only" && batch.config?.checklist_name
                        ? batch.config.checklist_name
                        : batch.config?.method?.toUpperCase() ?? "--"}
                  </td>
                  <td
                    className="px-5 py-3"
                    style={{ color: "var(--text-secondary)", fontSize: "12px" }}
                  >
                    {batch.config?.pipeline_mode === "score_only"
                      ? "———"
                      : batch.config?.generator_model || batch.config?.model || "--"}
                  </td>
                  <td
                    className="px-5 py-3"
                    style={{ color: "var(--text-secondary)", fontSize: "12px" }}
                  >
                    {batch.config?.pipeline_mode === "generate_only"
                      ? "———"
                      : batch.config?.scorer_model || batch.config?.model || "--"}
                  </td>
                  <td className="px-5 py-3">
                    <Badge
                      variant={
                        STATUS_BADGE_VARIANT[batch.status] ?? "default"
                      }
                    >
                      {batch.status}
                    </Badge>
                  </td>
                  <td
                    className="px-5 py-3 text-right"
                    style={{
                      color: "var(--text-secondary)",
                      fontFamily: "var(--font-mono)",
                      fontSize: "12px",
                    }}
                  >
                    {batch.completed}/{batch.total}
                  </td>
                  <td
                    className="px-5 py-3 text-right font-medium"
                    style={{
                      color:
                        (batch.mean_score ?? batch.macro_pass_rate) != null
                          ? "var(--text-primary)"
                          : "var(--text-tertiary)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {(batch.mean_score ?? batch.macro_pass_rate) != null
                      ? `${((batch.mean_score ?? batch.macro_pass_rate)! * 100).toFixed(1)}%`
                      : "--"}
                  </td>
                  <td
                    className="px-5 py-3 text-right"
                    style={{
                      color: "var(--text-tertiary)",
                      fontSize: "12px",
                    }}
                  >
                    {formatDate(batch.started_at)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(batch.batch_id);
                      }}
                      className="p-1 rounded hover:bg-[var(--error-light)] transition-colors"
                      style={{ color: "var(--text-tertiary)" }}
                      title="Delete batch"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
