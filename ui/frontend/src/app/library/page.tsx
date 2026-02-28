"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/layout/PageHeader";
import { Tabs, TabContent } from "@/components/ui/Tabs";
import { PromptTemplateList } from "@/components/library/PromptTemplateList";
import { PipelineList } from "@/components/library/PipelineList";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { useChecklists } from "@/lib/hooks";
import { deleteChecklist } from "@/lib/api";
import { GENERATOR_CLASSES } from "@/lib/types";
import type { ChecklistSummary } from "@/lib/types";

const TAB_ITEMS = [
  { value: "checklists", label: "Checklists" },
  { value: "prompts", label: "Prompt Templates" },
  { value: "pipelines", label: "Pipelines" },
];

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function ChecklistsTab() {
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const { data: checklists, isLoading } = useChecklists();
  const queryClient = useQueryClient();
  const router = useRouter();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteChecklist(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
      setDeleteTarget(null);
    },
  });

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <p style={{ color: "var(--text-secondary)" }}>Loading checklists...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={() => router.push("/checklists")}>
          Manage Checklists
        </Button>
      </div>

      {checklists && checklists.length === 0 && (
        <Card>
          <CardContent className="py-16 text-center">
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
              Create or import checklists to build a reusable evaluation library.
            </p>
            <Button onClick={() => router.push("/checklists")}>
              Go to Checklists
            </Button>
          </CardContent>
        </Card>
      )}

      {checklists && checklists.length > 0 && (
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
                  <th className="text-left px-5 py-3 font-medium">Class</th>
                  <th className="text-left px-5 py-3 font-medium">Level</th>
                  <th className="text-center px-5 py-3 font-medium">Items</th>
                  <th className="text-left px-5 py-3 font-medium">Created</th>
                  <th className="text-right px-5 py-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {checklists.map((cl: ChecklistSummary) => (
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
                      {(() => {
                        const cls = Object.entries(GENERATOR_CLASSES).find(
                          ([, info]) => info.methods.includes(cl.method)
                        );
                        return cls ? (
                          <Badge variant="info">{cls[1].label}</Badge>
                        ) : (
                          <span style={{ color: "var(--text-tertiary)" }}>—</span>
                        );
                      })()}
                    </td>
                    <td className="px-5 py-3">
                      <Badge variant="default">{cl.level}</Badge>
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
                          setDeleteTarget(cl.id);
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

      {/* Dialogs omitted for brevity - users will use the existing /checklists routes */}
      <Dialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}>
        <DialogContent title="Delete Checklist" description="This action cannot be undone.">
          <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>
            Are you sure you want to permanently delete this checklist?
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>Cancel</Button>
            <Button
              className="bg-[var(--error)] hover:bg-[var(--error)] text-white"
              loading={deleteMutation.isPending}
              onClick={() => { if (deleteTarget) deleteMutation.mutate(deleteTarget); }}
            >
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default function LibraryPage() {
  const [activeTab, setActiveTab] = useState("checklists");

  return (
    <div>
      <PageHeader description="Manage saved checklists, prompt templates, and custom pipelines." />
      <Tabs items={TAB_ITEMS} value={activeTab} onValueChange={setActiveTab}>
        <TabContent value="prompts">
          <PromptTemplateList />
        </TabContent>
        <TabContent value="checklists">
          <ChecklistsTab />
        </TabContent>
        <TabContent value="pipelines">
          <PipelineList />
        </TabContent>
      </Tabs>
    </div>
  );
}
