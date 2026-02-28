"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { usePipeline } from "@/lib/hooks";
import { updatePipeline, deletePipeline } from "@/lib/api";
import { ScorerConfig, DEFAULT_SCORER_CONFIG } from "@/lib/types";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Card, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { PromptEditor } from "@/components/playground/PromptEditor";
import { ScorerConfigPanel } from "@/components/playground/ScorerConfigPanel";

const GENERATOR_CLASS_OPTIONS = [
  { value: "direct", label: "Direct" },
  { value: "contrastive", label: "Contrastive" },
];

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
function PipelineForm({ pipeline, id }: { pipeline: any; id: string }) {
  const router = useRouter();
  const queryClient = useQueryClient();

  const [name, setName] = useState(pipeline.name);
  const [description, setDescription] = useState(pipeline.description);
  const [generatorClass, setGeneratorClass] = useState(pipeline.generator_class);
  const [generatorPrompt, setGeneratorPrompt] = useState(pipeline.generator_prompt);
  const [scorerConfig, setScorerConfig] = useState<ScorerConfig>(
    pipeline.scorer_config ?? DEFAULT_SCORER_CONFIG
  );
  const [scorerPrompt, setScorerPrompt] = useState(pipeline.scorer_prompt);
  // Output format is stored for save but not yet editable in UI
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [outputFormat, setOutputFormat] = useState(pipeline.output_format);
  const [hasChanges, setHasChanges] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [promptTab, setPromptTab] = useState<"generator" | "scorer">("generator");

  const saveMutation = useMutation({
    mutationFn: () =>
      updatePipeline(id, {
        name,
        description,
        generator_class: generatorClass,
        generator_prompt: generatorPrompt,
        scorer_config: scorerConfig,
        scorer_prompt: scorerPrompt,
        output_format: outputFormat,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline", id] });
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      setHasChanges(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deletePipeline(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      router.push("/library");
    },
  });

  const markChanged = () => setHasChanges(true);

  return (
    <div>
      <PageHeader title={pipeline.name}>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => router.push("/library")}>
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
            onChange={(e) => { setName(e.target.value); markChanged(); }}
          />
          <Input
            label="Description"
            value={description}
            onChange={(e) => { setDescription(e.target.value); markChanged(); }}
            placeholder="Brief description..."
          />
        </div>

        {/* Configuration badges */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
              Generator
            </span>
            <Badge variant="info">
              {generatorClass === "direct" ? "Direct" : "Contrastive"}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium" style={{ color: "var(--text-tertiary)" }}>
              Scorer
            </span>
            <Badge variant="default">{scorerConfig.mode} / {scorerConfig.primary_metric}</Badge>
          </div>
        </div>

        {/* Generator / Scorer settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Select
            label="Generator Class"
            value={generatorClass}
            onChange={(e) => { setGeneratorClass(e.target.value); markChanged(); }}
            options={GENERATOR_CLASS_OPTIONS}
          />
          <div>
            <label
              className="block text-sm font-medium mb-2"
              style={{ color: "var(--text-secondary)" }}
            >
              Scorer Config
            </label>
            <ScorerConfigPanel
              value={scorerConfig}
              onChange={(c) => { setScorerConfig(c); markChanged(); }}
            />
          </div>
        </div>

        {/* Prompt Tab Toggle */}
        <div>
          <div
            className="inline-flex rounded-md p-0.5 mb-3"
            style={{ backgroundColor: "var(--surface-elevated)" }}
          >
            <button
              onClick={() => setPromptTab("generator")}
              className="px-4 py-1.5 text-sm font-medium rounded transition-all"
              style={{
                backgroundColor: promptTab === "generator" ? "white" : "transparent",
                color: promptTab === "generator" ? "var(--accent-primary)" : "var(--text-secondary)",
                boxShadow: promptTab === "generator" ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
              }}
            >
              Generator Prompt
            </button>
            <button
              onClick={() => setPromptTab("scorer")}
              className="px-4 py-1.5 text-sm font-medium rounded transition-all"
              style={{
                backgroundColor: promptTab === "scorer" ? "white" : "transparent",
                color: promptTab === "scorer" ? "var(--accent-primary)" : "var(--text-secondary)",
                boxShadow: promptTab === "scorer" ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
              }}
            >
              Scorer Prompt
            </button>
          </div>

          {promptTab === "generator" && (
            <PromptEditor
              value={generatorPrompt}
              onChange={(v) => { setGeneratorPrompt(v); markChanged(); }}
              minHeight={300}
            />
          )}
          {promptTab === "scorer" && (
            <div className="space-y-3">
              <PromptEditor
                value={scorerPrompt}
                onChange={(v) => { setScorerPrompt(v); markChanged(); }}
                minHeight={200}
              />
              <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                Leave empty to use the default scorer prompt.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Delete dialog */}
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent title="Delete Pipeline" description="This action cannot be undone.">
          <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>
            Are you sure you want to permanently delete &ldquo;{pipeline.name}&rdquo;?
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

export default function PipelineDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const router = useRouter();

  const { data: pipeline, isLoading, error } = usePipeline(id);

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Loading..." />
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--text-secondary)" }}>Loading pipeline...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !pipeline) {
    return (
      <div>
        <PageHeader title="Not Found" />
        <Card>
          <CardContent className="py-12 text-center">
            <p style={{ color: "var(--error)" }}>Pipeline not found.</p>
            <Button variant="outline" className="mt-4" onClick={() => router.push("/library")}>
              Back to Library
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <PipelineForm key={pipeline.id ?? id} pipeline={pipeline} id={id} />;
}
