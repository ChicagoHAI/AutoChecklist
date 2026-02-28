"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { useChecklists } from "@/lib/hooks";
import { createChecklist, deleteChecklist, importChecklist } from "@/lib/api";
import { METHOD_TO_CLASS_LABEL } from "@/lib/types";
import type { ChecklistSummary } from "@/lib/types";

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function EmptyState({ onCreateClick }: { onCreateClick: () => void }) {
  return (
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
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2" />
            <rect x="9" y="3" width="6" height="4" rx="1" />
            <path d="M9 14l2 2 4-4" />
          </svg>
        </div>
        <h3
          className="text-lg font-semibold mb-2"
          style={{ color: "var(--text-primary)" }}
        >
          No checklists yet
        </h3>
        <p
          className="text-sm max-w-md mx-auto mb-6"
          style={{ color: "var(--text-secondary)" }}
        >
          Create your first checklist to start building a library of reusable
          evaluation criteria. You can also save checklists from the Evaluate
          page.
        </p>
        <Button onClick={onCreateClick}>Create New Checklist</Button>
      </CardContent>
    </Card>
  );
}

function CreateDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [method, setMethod] = useState("custom");
  const [customMethod, setCustomMethod] = useState("");
  const [level, setLevel] = useState("instance");
  const queryClient = useQueryClient();
  const router = useRouter();

  const mutation = useMutation({
    mutationFn: () =>
      createChecklist({
        name,
        method: method === "custom" ? (customMethod.trim() || "custom") : method,
        level,
        items: [],
        metadata: {},
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
      onOpenChange(false);
      setName("");
      setMethod("custom");
      setCustomMethod("");
      setLevel("instance");
      router.push(`/checklists/${data.id}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Create New Checklist" description="Give your checklist a name and configure its settings.">
        <div className="space-y-4">
          <Input
            label="Name"
            required
            placeholder="e.g., Code Quality Checklist"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                Method
              </label>
              <select
                className="w-full px-3 py-2 rounded-md border text-sm border-[var(--border)] bg-white text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)]"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
              >
                <option value="custom">Custom</option>
                <option value="tick">TICK</option>
                <option value="rlcf">RLCF</option>
                <option value="rocketeval">RocketEval</option>
              </select>
              {method === "custom" && (
                <input
                  className="w-full mt-2 px-3 py-2 rounded-md border text-sm border-[var(--border)] bg-white text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)]"
                  placeholder="Method name (optional)"
                  value={customMethod}
                  onChange={(e) => setCustomMethod(e.target.value)}
                />
              )}
            </div>
            <div>
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                Level
              </label>
              <select
                className="w-full px-3 py-2 rounded-md border text-sm border-[var(--border)] bg-white text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)]"
                value={level}
                onChange={(e) => setLevel(e.target.value)}
              >
                <option value="instance">Instance</option>
                <option value="corpus">Corpus</option>
              </select>
            </div>
          </div>
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
              {(mutation.error as Error).message || "Failed to create checklist"}
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ImportDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const queryClient = useQueryClient();
  const router = useRouter();

  const mutation = useMutation({
    mutationFn: () => importChecklist(file!, name),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
      onOpenChange(false);
      setName("");
      setFile(null);
      router.push(`/checklists/${data.id}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Import Checklist" description="Upload a JSON or CSV file with checklist items.">
        <div className="space-y-4">
          <Input
            label="Name"
            required
            placeholder="e.g., Imported Code Quality Checklist"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <div>
            <label
              className="block text-sm font-medium mb-1.5"
              style={{ color: "var(--text-secondary)" }}
            >
              File
            </label>
            <input
              type="file"
              accept=".json,.csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-sm border rounded-md px-3 py-2"
              style={{
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
            <p
              className="mt-1.5 text-xs"
              style={{ color: "var(--text-tertiary)" }}
            >
              JSON: array of {"{question, weight}"} or object with &quot;items&quot; key. CSV: question column, optional weight column.
            </p>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => mutation.mutate()}
              disabled={!name.trim() || !file}
              loading={mutation.isPending}
            >
              Import
            </Button>
          </div>
          {mutation.isError && (
            <p className="text-xs" style={{ color: "var(--error)" }}>
              {(mutation.error as Error).message || "Failed to import checklist"}
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ChecklistTable({
  checklists,
  onDelete,
}: {
  checklists: ChecklistSummary[];
  onDelete: (id: string) => void;
}) {
  const router = useRouter();

  return (
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
              <th className="text-left px-5 py-3 font-medium">Method</th>
              <th className="text-left px-5 py-3 font-medium">Level</th>
              <th className="text-center px-5 py-3 font-medium">Items</th>
              <th className="text-left px-5 py-3 font-medium">Created</th>
              <th className="text-right px-5 py-3 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {checklists.map((cl) => (
              <tr
                key={cl.id}
                className="border-b last:border-b-0 cursor-pointer transition-colors"
                style={{
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
                onClick={() => router.push(`/checklists/${cl.id}`)}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.backgroundColor =
                    "var(--surface-elevated)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.backgroundColor = "transparent")
                }
              >
                <td className="px-5 py-3 font-medium">{cl.name}</td>
                <td className="px-5 py-3">
                  <Badge variant="default">{METHOD_TO_CLASS_LABEL[cl.method] || cl.method}</Badge>
                </td>
                <td className="px-5 py-3">
                  <Badge variant={cl.level === "corpus" ? "info2" : "info"}>{cl.level}</Badge>
                </td>
                <td className="px-5 py-3 text-center">{cl.item_count}</td>
                <td
                  className="px-5 py-3"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {formatDate(cl.created_at)}
                </td>
                <td className="px-5 py-3 text-right">
                  <button
                    className="p-1.5 rounded-md transition-colors hover:bg-[var(--error-light)]"
                    style={{ color: "var(--text-tertiary)" }}
                    title="Delete checklist"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(cl.id);
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
  );
}

export default function ChecklistsPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const { data: checklists, isLoading, error } = useChecklists();
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteChecklist(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
      setDeleteTarget(null);
    },
  });

  const handleDeleteClick = (id: string) => {
    setDeleteTarget(id);
  };

  return (
    <div>
      <PageHeader
        title="Checklist Library"
        description="Browse, manage, and reuse your saved evaluation checklists."
      >
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setImportOpen(true)}>Import</Button>
          <Button onClick={() => setCreateOpen(true)}>Create New</Button>
        </div>
      </PageHeader>

      {isLoading && (
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--text-secondary)" }}>
              Loading checklists...
            </p>
          </CardContent>
        </Card>
      )}

      {error && (
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--error)" }}>
              Failed to load checklists: {(error as Error).message}
            </p>
          </CardContent>
        </Card>
      )}

      {!isLoading && !error && checklists && checklists.length === 0 && (
        <EmptyState onCreateClick={() => setCreateOpen(true)} />
      )}

      {!isLoading && !error && checklists && checklists.length > 0 && (
        <ChecklistTable
          checklists={checklists}
          onDelete={handleDeleteClick}
        />
      )}

      <CreateDialog open={createOpen} onOpenChange={setCreateOpen} />
      <ImportDialog open={importOpen} onOpenChange={setImportOpen} />

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <DialogContent title="Delete Checklist" description="This action cannot be undone.">
          <p
            className="text-sm mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Are you sure you want to permanently delete this checklist?
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
