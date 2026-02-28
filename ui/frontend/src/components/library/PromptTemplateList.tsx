"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { usePromptTemplates } from "@/lib/hooks";
import {
  createPromptTemplate,
  deletePromptTemplate,
  seedDefaultPrompts,
} from "@/lib/api";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import type { PromptTemplateSummary } from "@/lib/types";

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function CreatePromptDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const queryClient = useQueryClient();
  const router = useRouter();

  const mutation = useMutation({
    mutationFn: () =>
      createPromptTemplate({
        name,
        prompt_text: "Write your evaluation prompt here.\n\nUse {input} for the instruction and {target} for the response to evaluate.",
        description,
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
      onOpenChange(false);
      setName("");
      setDescription("");
      router.push(`/library/prompts/${data.id}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Create Prompt Template"
        description="Give your prompt template a name. You can edit the prompt text after creation."
      >
        <div className="space-y-4">
          <Input
            label="Name"
            required
            placeholder="e.g., Code Quality Evaluation"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Input
            label="Description (optional)"
            placeholder="Brief description of what this prompt evaluates"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => mutation.mutate()}
              disabled={!name.trim()}
              loading={mutation.isPending}
            >
              Create
            </Button>
          </div>
          {mutation.isError && (
            <p className="text-xs" style={{ color: "var(--error)" }}>
              {(mutation.error as Error).message || "Failed to create template"}
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function PromptTemplateList() {
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const { data: templates, isLoading, error } = usePromptTemplates();
  const queryClient = useQueryClient();
  const router = useRouter();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deletePromptTemplate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
      setDeleteTarget(null);
    },
  });

  const seedMutation = useMutation({
    mutationFn: () => seedDefaultPrompts(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
    },
  });

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <p style={{ color: "var(--text-secondary)" }}>
            Loading prompt templates...
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
            Failed to load templates: {(error as Error).message}
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end gap-2">
        <Button
          variant="outline"
          onClick={() => seedMutation.mutate()}
          loading={seedMutation.isPending}
          disabled={seedMutation.isPending}
        >
          Load Built-in Prompts
        </Button>
        <Button onClick={() => setCreateOpen(true)}>Create New</Button>
      </div>

      {templates && templates.length === 0 && (
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
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
            </div>
            <h3
              className="text-lg font-semibold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              No prompt templates yet
            </h3>
            <p
              className="text-sm max-w-md mx-auto mb-6"
              style={{ color: "var(--text-secondary)" }}
            >
              Create reusable prompt templates for evaluation. You can also save
              prompts from the Playground page.
            </p>
            <Button onClick={() => setCreateOpen(true)}>
              Create First Template
            </Button>
          </CardContent>
        </Card>
      )}

      {templates && templates.length > 0 && (
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
                  <th className="text-left px-5 py-3 font-medium">Type</th>
                  <th className="text-left px-5 py-3 font-medium">
                    Description
                  </th>
                  <th className="text-left px-5 py-3 font-medium">
                    Placeholders
                  </th>
                  <th className="text-left px-5 py-3 font-medium">Created</th>
                  <th className="text-right px-5 py-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {[...templates].sort((a, b) => {
                  const typeA = a.metadata?.type || "";
                  const typeB = b.metadata?.type || "";
                  if (typeA === typeB) return 0;
                  if (typeA === "generator") return -1;
                  if (typeB === "generator") return 1;
                  return 0;
                }).map((t: PromptTemplateSummary) => (
                  <tr
                    key={t.id}
                    className="border-b last:border-b-0 cursor-pointer transition-colors"
                    style={{
                      borderColor: "var(--border)",
                      color: "var(--text-primary)",
                    }}
                    onClick={() => router.push(`/library/prompts/${t.id}`)}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.backgroundColor =
                        "var(--surface-elevated)")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.backgroundColor = "transparent")
                    }
                  >
                    <td className="px-5 py-3 font-medium">{t.name}</td>
                    <td className="px-5 py-3">
                      {t.metadata?.type === "generator" ? (
                        <Badge variant="info">
                          {t.metadata?.generator_class === "contrastive"
                            ? "ContrastiveGenerator"
                            : "DirectGenerator"}
                        </Badge>
                      ) : t.metadata?.type === "scorer" ? (
                        <Badge variant="info">Scorer</Badge>
                      ) : (
                        <span style={{ color: "var(--text-tertiary)" }}>—</span>
                      )}
                    </td>
                    <td
                      className="px-5 py-3 max-w-[200px] truncate"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {t.description || "—"}
                    </td>
                    <td className="px-5 py-3">
                      <div className="flex gap-1">
                        {t.placeholders.map((p) => (
                          <Badge key={p} variant="default">
                            {`{${p}}`}
                          </Badge>
                        ))}
                        {t.placeholders.length === 0 && (
                          <span style={{ color: "var(--text-tertiary)" }}>
                            None
                          </span>
                        )}
                      </div>
                    </td>
                    <td
                      className="px-5 py-3"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {formatDate(t.created_at)}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button
                        className="p-1.5 rounded-md transition-colors hover:bg-[var(--error-light)]"
                        style={{ color: "var(--text-tertiary)" }}
                        title="Delete template"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteTarget(t.id);
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

      <CreatePromptDialog open={createOpen} onOpenChange={setCreateOpen} />

      {/* Delete confirmation */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <DialogContent
          title="Delete Prompt Template"
          description="This action cannot be undone."
        >
          <p
            className="text-sm mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Are you sure you want to permanently delete this prompt template?
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
