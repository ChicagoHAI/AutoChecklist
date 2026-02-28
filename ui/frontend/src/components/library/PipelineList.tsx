"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { usePipelines } from "@/lib/hooks";
import { deletePipeline } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import type { PipelineSummary } from "@/lib/types";

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function PipelineList() {
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const { data: pipelines, isLoading, error } = usePipelines();
  const queryClient = useQueryClient();
  const router = useRouter();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deletePipeline(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      setDeleteTarget(null);
    },
  });

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <p style={{ color: "var(--text-secondary)" }}>
            Loading pipelines...
          </p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <p style={{ color: "var(--error)" }}>
            Failed to load pipelines: {(error as Error).message}
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {pipelines && pipelines.length === 0 && (
        <Card>
          <CardContent className="py-16 text-center">
            <div
              className="w-14 h-14 rounded-md mx-auto mb-4 flex items-center justify-center"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            >
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ color: "var(--text-tertiary)" }}
              >
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            </div>
            <h3
              className="text-lg font-semibold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              No pipelines yet
            </h3>
            <p
              className="text-sm max-w-md mx-auto mb-6"
              style={{ color: "var(--text-secondary)" }}
            >
              Save a pipeline from the Evaluate tab to create reusable
              evaluation configurations with custom prompts.
            </p>
          </CardContent>
        </Card>
      )}

      {pipelines && pipelines.length > 0 && (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr
                  className="border-b"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--text-secondary)",
                  }}
                >
                  <th className="text-left px-5 py-3 font-medium">Name</th>
                  <th className="text-left px-5 py-3 font-medium">Generator</th>
                  <th className="text-left px-5 py-3 font-medium">Scorer</th>
                  <th className="text-left px-5 py-3 font-medium">Created</th>
                  <th className="text-right px-5 py-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {pipelines.map((p: PipelineSummary) => (
                  <tr
                    key={p.id}
                    className="border-b last:border-b-0 cursor-pointer transition-colors"
                    style={{
                      borderColor: "var(--border)",
                      color: "var(--text-primary)",
                    }}
                    onClick={() => router.push(`/library/pipelines/${p.id}`)}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.backgroundColor =
                        "var(--surface-elevated)")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.backgroundColor = "transparent")
                    }
                  >
                    <td className="px-5 py-3 font-medium">{p.name}</td>
                    <td className="px-5 py-3">
                      <Badge variant="info">
                        {p.generator_class === "direct" ? "Direct" : "Contrastive"}
                      </Badge>
                    </td>
                    <td className="px-5 py-3">
                      <Badge variant="default">{p.scorer_class}</Badge>
                    </td>
                    <td
                      className="px-5 py-3"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {formatDate(p.created_at)}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button
                        className="p-1.5 rounded-md transition-colors hover:bg-[var(--error-light)]"
                        style={{ color: "var(--text-tertiary)" }}
                        title="Delete pipeline"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteTarget(p.id);
                        }}
                        onMouseEnter={(e) =>
                          (e.currentTarget.style.color = "var(--error)")
                        }
                        onMouseLeave={(e) =>
                          (e.currentTarget.style.color = "var(--text-tertiary)")
                        }
                      >
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                        </svg>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Delete confirmation */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <DialogContent
          title="Delete Pipeline"
          description="This action cannot be undone."
        >
          <p
            className="text-sm mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Are you sure you want to permanently delete this pipeline?
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button
              className="bg-[var(--error)] hover:bg-[var(--error)] text-white"
              loading={deleteMutation.isPending}
              onClick={() => {
                if (deleteTarget) deleteMutation.mutate(deleteTarget);
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
