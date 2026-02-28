"use client";

import { useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card, CardHeader, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { useChecklist } from "@/lib/hooks";
import { updateChecklist, deleteChecklist } from "@/lib/api";
import { METHOD_TO_CLASS_LABEL } from "@/lib/types";
import type { SavedChecklist } from "@/lib/types";

interface EditableItem {
  question: string;
  weight: number;
}

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ChecklistDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const { data: checklist, isLoading, error } = useChecklist(id);

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Loading..." />
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--text-secondary)" }}>
              Loading checklist details...
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !checklist) {
    return <ChecklistNotFound error={error} />;
  }

  // Key on checklist.id so React re-mounts the form when navigating between checklists
  return <ChecklistForm key={checklist.id} checklist={checklist} />;
}

function ChecklistNotFound({ error }: { error: Error | null }) {
  const router = useRouter();
  return (
    <div>
      <PageHeader title="Checklist Not Found" />
      <Card>
        <CardContent className="py-12 text-center">
          <p style={{ color: "var(--error)" }} className="mb-4">
            {error ? error.message : "This checklist could not be found."}
          </p>
          <Button variant="outline" onClick={() => router.push("/library")}>
            Back to Library
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function ChecklistForm({ checklist }: { checklist: SavedChecklist }) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const id = checklist.id;

  // Initialize local state from the checklist prop (runs once per mount due to key={checklist.id})
  const [name, setName] = useState(checklist.name);
  const [items, setItems] = useState<EditableItem[]>(
    checklist.items.map((i) => ({ question: i.question, weight: i.weight }))
  );
  const [hasChanges, setHasChanges] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const markChanged = useCallback(() => setHasChanges(true), []);

  const handleNameChange = (val: string) => {
    setName(val);
    markChanged();
  };

  const handleItemChange = (index: number, field: "question" | "weight", value: string | number) => {
    setItems((prev) => {
      const updated = [...prev];
      if (field === "question") {
        updated[index] = { ...updated[index], question: value as string };
      } else {
        updated[index] = { ...updated[index], weight: Number(value) || 0 };
      }
      return updated;
    });
    markChanged();
  };

  const addItem = () => {
    setItems((prev) => [...prev, { question: "", weight: 1.0 }]);
    markChanged();
  };

  const removeItem = (index: number) => {
    setItems((prev) => prev.filter((_, i) => i !== index));
    markChanged();
  };

  const saveMutation = useMutation({
    mutationFn: () =>
      updateChecklist(id, {
        name,
        items: items.filter((i) => i.question.trim() !== ""),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData(["checklist", id], data);
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
      setHasChanges(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteChecklist(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
      router.push("/library");
    },
  });

  return (
    <div>
      <PageHeader title={checklist.name}>
        <Button
          variant="outline"
          size="sm"
          onClick={() => router.push("/library")}
        >
          Back to Library
        </Button>
      </PageHeader>

      {/* Metadata section */}
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <h2
              className="text-base font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Details
            </h2>
            <div className="flex items-center gap-2">
              <Badge variant="default">{METHOD_TO_CLASS_LABEL[checklist.method] || checklist.method}</Badge>
              <Badge variant={checklist.level === "corpus" ? "info2" : "info"}>{checklist.level}</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Name"
              value={name}
              onChange={(e) => handleNameChange(e.target.value)}
            />
            <div>
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                Model
              </label>
              <p
                className="px-3 py-2 text-sm rounded-md"
                style={{
                  backgroundColor: "var(--surface-elevated)",
                  color: "var(--text-primary)",
                }}
              >
                {checklist.model || "Not specified"}
              </p>
            </div>
            <div>
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                Created
              </label>
              <p
                className="text-sm"
                style={{ color: "var(--text-primary)" }}
              >
                {formatDate(checklist.created_at)}
              </p>
            </div>
            <div>
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                Last Updated
              </label>
              <p
                className="text-sm"
                style={{ color: "var(--text-primary)" }}
              >
                {formatDate(checklist.updated_at)}
              </p>
            </div>
          </div>
          {checklist.metadata && Object.keys(checklist.metadata).length > 0 && (
            <div className="mt-4">
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                Metadata
              </label>
              <pre
                className="text-xs p-3 rounded-md overflow-x-auto"
                style={{
                  backgroundColor: "var(--surface-elevated)",
                  color: "var(--text-secondary)",
                }}
              >
                {JSON.stringify(checklist.metadata, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Items section */}
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <h2
              className="text-base font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Checklist Items ({items.length})
            </h2>
            <Button variant="outline" size="sm" onClick={addItem}>
              + Add Item
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {items.length === 0 ? (
            <p
              className="text-sm text-center py-8"
              style={{ color: "var(--text-tertiary)" }}
            >
              No items yet. Click &quot;+ Add Item&quot; to add your first
              checklist question.
            </p>
          ) : (
            <div className="space-y-3">
              {items.map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-md border"
                  style={{ borderColor: "var(--border)" }}
                >
                  <span
                    className="text-xs font-mono mt-2.5 min-w-[24px] text-center"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {idx + 1}
                  </span>
                  <div className="flex-1">
                    <input
                      type="text"
                      className="w-full px-3 py-2 rounded-md border text-sm border-[var(--border)] bg-white text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)]"
                      placeholder="Enter yes/no question..."
                      value={item.question}
                      onChange={(e) =>
                        handleItemChange(idx, "question", e.target.value)
                      }
                    />
                  </div>
                  <div className="w-20">
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="10"
                      className="w-full px-2 py-2 rounded-md border text-sm text-center border-[var(--border)] bg-white text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)]"
                      title="Weight"
                      value={item.weight}
                      onChange={(e) =>
                        handleItemChange(idx, "weight", e.target.value)
                      }
                    />
                  </div>
                  <button
                    className="p-2 rounded-md transition-colors hover:bg-[var(--error-light)] mt-0.5"
                    style={{ color: "var(--text-tertiary)" }}
                    title="Remove item"
                    onClick={() => removeItem(idx)}
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
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Action bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            onClick={() => saveMutation.mutate()}
            disabled={!hasChanges}
            loading={saveMutation.isPending}
          >
            Save Changes
          </Button>
          {saveMutation.isSuccess && !hasChanges && (
            <span
              className="text-sm"
              style={{ color: "var(--success)" }}
            >
              Saved
            </span>
          )}
          {saveMutation.isError && (
            <span className="text-sm" style={{ color: "var(--error)" }}>
              {(saveMutation.error as Error).message || "Save failed"}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            className="text-[var(--error)] hover:bg-[var(--error-light)]"
            onClick={() => setDeleteOpen(true)}
          >
            Delete
          </Button>
        </div>
      </div>

      {/* Delete confirmation dialog */}
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent
          title="Delete Checklist"
          description="This action cannot be undone."
        >
          <p
            className="text-sm mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Are you sure you want to permanently delete &quot;{checklist.name}
            &quot;?
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              Cancel
            </Button>
            <Button
              className="bg-[var(--error)] hover:bg-[var(--error)] text-white"
              loading={deleteMutation.isPending}
              onClick={() => deleteMutation.mutate()}
            >
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
