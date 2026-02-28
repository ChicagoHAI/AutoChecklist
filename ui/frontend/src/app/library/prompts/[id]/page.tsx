"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { usePromptTemplate } from "@/lib/hooks";
import { updatePromptTemplate, deletePromptTemplate } from "@/lib/api";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Card, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { PromptEditor } from "@/components/playground/PromptEditor";

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
function PromptForm({ template, id }: { template: any; id: string }) {
  const router = useRouter();
  const queryClient = useQueryClient();

  const [name, setName] = useState(template.name);
  const [description, setDescription] = useState(template.description);
  const [promptText, setPromptText] = useState(template.prompt_text);
  const [hasChanges, setHasChanges] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const saveMutation = useMutation({
    mutationFn: () =>
      updatePromptTemplate(id, {
        name,
        description,
        prompt_text: promptText,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-template", id] });
      queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
      setHasChanges(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deletePromptTemplate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
      router.push("/library");
    },
  });

  const handleFieldChange = (
    setter: (v: string) => void,
    value: string
  ) => {
    setter(value);
    setHasChanges(true);
  };

  return (
    <div>
      <PageHeader title={template.name}>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => router.push("/library")}
          >
            Back
          </Button>
          <Button
            size="sm"
            onClick={() => saveMutation.mutate()}
            disabled={!hasChanges}
            loading={saveMutation.isPending}
          >
            {saveMutation.isSuccess && !hasChanges ? "Saved" : "Save"}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setDeleteOpen(true)}
            className="text-[var(--error)]"
          >
            Delete
          </Button>
        </div>
      </PageHeader>

      <div className="space-y-6">
        {/* Metadata */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Name"
            value={name}
            onChange={(e) => handleFieldChange(setName, e.target.value)}
          />
          <Input
            label="Description"
            value={description}
            onChange={(e) => handleFieldChange(setDescription, e.target.value)}
            placeholder="Brief description..."
          />
        </div>

        {/* Placeholders */}
        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{ color: "var(--text-secondary)" }}
          >
            Detected Placeholders
          </label>
          <div className="flex gap-2">
            {template.placeholders.length > 0 ? (
              template.placeholders.map((p: string) => (
                <Badge key={p} variant="info">
                  {`{${p}}`}
                </Badge>
              ))
            ) : (
              <span
                className="text-sm"
                style={{ color: "var(--text-tertiary)" }}
              >
                No placeholders detected. Use {"{input}"}, {"{target}"}, or{" "}
                {"{reference}"} in your prompt.
              </span>
            )}
          </div>
        </div>

        {/* Prompt Editor */}
        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{ color: "var(--text-secondary)" }}
          >
            Prompt Template
          </label>
          <PromptEditor
            value={promptText}
            onChange={(v) => handleFieldChange(setPromptText, v)}
            minHeight={400}
          />
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3" />
      </div>

      {/* Delete dialog */}
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent
          title="Delete Prompt Template"
          description="This action cannot be undone."
        >
          <p
            className="text-sm mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Are you sure you want to permanently delete &ldquo;{template.name}
            &rdquo;?
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

export default function PromptTemplateDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const router = useRouter();

  const { data: template, isLoading, error } = usePromptTemplate(id);

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Loading..." />
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--text-secondary)" }}>
              Loading prompt template...
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !template) {
    return (
      <div>
        <PageHeader title="Not Found" />
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--error)" }}>
              Prompt template not found.
            </p>
            <Button
              variant="outline"
              className="mt-4"
              onClick={() => router.push("/library")}
            >
              Back to Library
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <PromptForm key={template.id ?? id} template={template} id={id} />;
}
