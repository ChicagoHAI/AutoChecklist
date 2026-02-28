"use client";

import { useState, useCallback, useRef, useEffect, KeyboardEvent, ChangeEvent } from "react";
import Editor from "@monaco-editor/react";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { ModelSelector } from "@/components/layout/ModelSelector";
import { useDefaultModel, useDimensionGenerate } from "@/lib/hooks";
import { createChecklist } from "@/lib/api";
import type { DimensionInput, Checklist } from "@/lib/types";

const AUGMENTATION_MODES = [
  { value: "seed", label: "Seed", description: "1-3 questions per sub-dimension (fast)" },
  { value: "elaboration", label: "Elaboration", description: "5+ detailed questions per sub-dimension" },
  { value: "diversification", label: "Diversification", description: "Alternative framings of criteria" },
  { value: "combined", label: "Combined", description: "All modes merged and deduped (thorough)" },
];

function emptyDimension(): DimensionInput {
  return { name: "", definition: "", sub_dimensions: [] };
}

export function DimensionForm() {
  const { provider, setProvider, model, setModel } = useDefaultModel();
  const dimensionMutation = useDimensionGenerate();

  const [taskType, setTaskType] = useState("general");
  const [augmentationMode, setAugmentationMode] = useState("seed");
  const [dimensions, setDimensions] = useState<DimensionInput[]>([emptyDimension()]);
  const [subDimInput, setSubDimInput] = useState<Record<number, string>>({});

  const [generatedChecklist, setGeneratedChecklist] = useState<Checklist | null>(null);
  const [scriptDialogOpen, setScriptDialogOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dimension CRUD
  const updateDimension = useCallback((index: number, updates: Partial<DimensionInput>) => {
    setDimensions((prev) => prev.map((d, i) => (i === index ? { ...d, ...updates } : d)));
  }, []);

  const removeDimension = useCallback((index: number) => {
    setDimensions((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const addSubDimension = useCallback((dimIndex: number) => {
    const text = (subDimInput[dimIndex] || "").trim();
    if (!text) return;
    setDimensions((prev) =>
      prev.map((d, i) =>
        i === dimIndex && !d.sub_dimensions.includes(text)
          ? { ...d, sub_dimensions: [...d.sub_dimensions, text] }
          : d
      )
    );
    setSubDimInput((prev) => ({ ...prev, [dimIndex]: "" }));
  }, [subDimInput]);

  const removeSubDimension = useCallback((dimIndex: number, subIndex: number) => {
    setDimensions((prev) =>
      prev.map((d, i) =>
        i === dimIndex
          ? { ...d, sub_dimensions: d.sub_dimensions.filter((_, si) => si !== subIndex) }
          : d
      )
    );
  }, []);

  const handleSubDimKeyDown = useCallback(
    (dimIndex: number, e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        addSubDimension(dimIndex);
      }
    },
    [addSubDimension]
  );

  // Upload JSON
  const handleUpload = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);

    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const raw = JSON.parse(ev.target?.result as string);
        const arr: unknown[] = Array.isArray(raw) ? raw : Array.isArray(raw?.dimensions) ? raw.dimensions : null as never;
        if (!arr) {
          setUploadError("Expected an array or an object with a \"dimensions\" array.");
          return;
        }
        const parsed: DimensionInput[] = arr.map((item: unknown, i: number) => {
          const d = item as Record<string, unknown>;
          if (typeof d.name !== "string" || !d.name.trim()) throw new Error(`Dimension ${i + 1}: missing "name"`);
          if (typeof d.definition !== "string" || !d.definition.trim()) throw new Error(`Dimension ${i + 1}: missing "definition"`);
          return {
            name: d.name.trim(),
            definition: d.definition.trim(),
            sub_dimensions: Array.isArray(d.sub_dimensions) ? d.sub_dimensions.filter((s): s is string => typeof s === "string") : [],
          };
        });
        if (parsed.length === 0) { setUploadError("No dimensions found in file."); return; }
        setDimensions(parsed);
        setSubDimInput({});
      } catch (err) {
        setUploadError(err instanceof SyntaxError ? "Invalid JSON file." : (err as Error).message);
      }
    };
    reader.readAsText(file);
    // Reset so re-uploading the same file triggers onChange
    e.target.value = "";
  }, []);

  // Generate
  const validDimensions = dimensions.filter((d) => d.name.trim() && d.definition.trim());

  const handleGenerate = useCallback(() => {
    if (validDimensions.length === 0) return;
    dimensionMutation.mutate(
      {
        dimensions: validDimensions,
        task_type: taskType,
        augmentation_mode: augmentationMode,
        provider,
        model,
      },
      {
        onSuccess: (data) => {
          setGeneratedChecklist(data.checklist);
          setSaveStatus("idle");
        },
      }
    );
  }, [validDimensions, taskType, augmentationMode, provider, model, dimensionMutation]);

  // Elapsed timer during generation
  const [elapsed, setElapsed] = useState(0);
  const startTimeRef = useRef(0);
  useEffect(() => {
    if (!dimensionMutation.isPending) return;
    startTimeRef.current = Date.now();
    const t = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => { clearInterval(t); setElapsed(0); };
  }, [dimensionMutation.isPending]);

  // Group items by category
  const groupedItems = generatedChecklist
    ? (() => {
        const categories: string[] =
          (generatedChecklist.metadata?.item_categories as string[]) || [];
        const groups: Record<string, Array<{ question: string; weight: number }>> = {};
        generatedChecklist.items.forEach((item, i) => {
          const cat = categories[i] || "uncategorized";
          if (!groups[cat]) groups[cat] = [];
          groups[cat].push(item);
        });
        return groups;
      })()
    : null;

  // Export script
  const buildScript = useCallback(() => {
    const dims = validDimensions
      .map((d) => {
        const subDims = d.sub_dimensions.length > 0
          ? `\n        sub_dimensions=${JSON.stringify(d.sub_dimensions)},`
          : "";
        return `    DimensionInput(\n        name="${d.name}",\n        definition="${d.definition}",${subDims}\n    ),`;
      })
      .join("\n");

    return `from autochecklist import DimensionGenerator, DimensionInput

gen = DimensionGenerator(
    model="${model}",
    provider="${provider}",
    augmentation_mode="${augmentationMode}",
    task_type="${taskType}",
)

dimensions = [
${dims}
]

checklist = gen.generate(dimensions=dimensions)
checklist.save("checklist.json")
print(f"Generated {len(checklist.items)} items across {len(set(i.category for i in checklist.items))} dimensions")`;
  }, [validDimensions, model, provider, augmentationMode, taskType]);

  const handleExportScript = useCallback(() => {
    setScriptDialogOpen(true);
    setCopied(false);
  }, []);

  const handleCopyScript = useCallback(() => {
    navigator.clipboard.writeText(buildScript());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [buildScript]);

  // Save to library
  const handleSave = useCallback(async () => {
    if (!generatedChecklist) return;
    setSaveStatus("saving");
    try {
      const dimNames = validDimensions.map((d) => d.name).join(", ");
      await createChecklist({
        name: `${taskType} — ${dimNames}`,
        method: "checkeval",
        level: "corpus",
        model,
        items: generatedChecklist.items.map((item) => ({
          question: item.question,
          weight: item.weight,
        })),
        metadata: {
          task_type: taskType,
          augmentation_mode: augmentationMode,
          dimensions: validDimensions,
          ...generatedChecklist.metadata,
        },
      });
      setSaveStatus("saved");
    } catch {
      setSaveStatus("error");
    }
  }, [generatedChecklist, validDimensions, taskType, augmentationMode, model]);

  const scriptSnippet = buildScript();

  return (
    <>
      <div className="space-y-4">
        {/* Settings */}
        <Card>
          <CardHeader>
            <h3
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Settings
            </h3>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label
                  className="block text-xs font-medium mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Task Type
                </label>
                <input
                  type="text"
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value)}
                  placeholder="e.g. summarization, dialog, general"
                  className="w-full px-2 py-1.5 text-xs rounded-md border bg-white"
                  style={{
                    borderColor: "var(--border-strong)",
                    color: "var(--text-primary)",
                    fontFamily: "var(--font-sans)",
                  }}
                />
              </div>
              <div>
                <label
                  className="block text-xs font-medium mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Augmentation Mode
                </label>
                <select
                  value={augmentationMode}
                  onChange={(e) => setAugmentationMode(e.target.value)}
                  className="w-full px-2 py-1.5 text-xs rounded-md border bg-white"
                  style={{
                    borderColor: "var(--border-strong)",
                    color: "var(--text-secondary)",
                    fontFamily: "var(--font-sans)",
                  }}
                >
                  {AUGMENTATION_MODES.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label} — {m.description}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label
                  className="block text-xs font-medium mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Model
                </label>
                <ModelSelector
                  provider={provider}
                  model={model}
                  onProviderChange={setProvider}
                  onModelChange={setModel}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Dimensions */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3
                  className="text-sm font-semibold"
                  style={{ color: "var(--text-primary)" }}
                >
                  Dimensions
                </h3>
                <Badge>{dimensions.length}</Badge>
              </div>
              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  onChange={handleUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex items-center gap-1.5 px-2 py-1 text-xs font-medium rounded-md border hover:bg-[var(--surface-elevated)]"
                  style={{
                    borderColor: "var(--border-strong)",
                    color: "var(--text-secondary)",
                  }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  Upload JSON
                </button>
              </div>
            </div>
            {uploadError && (
              <p className="mt-1 text-xs" style={{ color: "var(--error)" }}>
                {uploadError}
              </p>
            )}
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {dimensions.map((dim, i) => (
                <div
                  key={i}
                  className="rounded-md p-3"
                  style={{
                    border: "1px solid var(--border)",
                    backgroundColor: "var(--surface-manila)",
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className="text-xs font-medium"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      Dimension {i + 1}
                    </span>
                    {dimensions.length > 1 && (
                      <button
                        onClick={() => removeDimension(i)}
                        className="p-0.5 rounded hover:bg-[var(--surface-elevated)]"
                        style={{ color: "var(--text-tertiary)" }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    )}
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
                    <input
                      type="text"
                      value={dim.name}
                      onChange={(e) => updateDimension(i, { name: e.target.value })}
                      placeholder="Name (e.g. coherence)"
                      className="w-full px-2 py-1.5 text-xs rounded-md border bg-white"
                      style={{
                        borderColor: "var(--border-strong)",
                        color: "var(--text-primary)",
                        fontFamily: "var(--font-mono)",
                      }}
                    />
                    <input
                      type="text"
                      value={dim.definition}
                      onChange={(e) => updateDimension(i, { definition: e.target.value })}
                      placeholder="Definition (e.g. The response should maintain logical flow)"
                      className="w-full px-2 py-1.5 text-xs rounded-md border bg-white"
                      style={{
                        borderColor: "var(--border-strong)",
                        color: "var(--text-primary)",
                        fontFamily: "var(--font-sans)",
                      }}
                    />
                  </div>
                  {/* Sub-dimensions */}
                  <div>
                    <div className="flex items-center gap-1 flex-wrap mb-1">
                      {dim.sub_dimensions.map((sub, si) => (
                        <span
                          key={si}
                          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs"
                          style={{
                            backgroundColor: "var(--accent-light)",
                            color: "var(--accent-primary)",
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {sub}
                          <button
                            onClick={() => removeSubDimension(i, si)}
                            className="hover:opacity-70"
                          >
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                              <line x1="18" y1="6" x2="6" y2="18" />
                              <line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                          </button>
                        </span>
                      ))}
                    </div>
                    <input
                      type="text"
                      value={subDimInput[i] || ""}
                      onChange={(e) => setSubDimInput((prev) => ({ ...prev, [i]: e.target.value }))}
                      onKeyDown={(e) => handleSubDimKeyDown(i, e)}
                      onBlur={() => addSubDimension(i)}
                      placeholder="Add sub-dimension (press Enter)"
                      className="w-full px-2 py-1 text-xs rounded-md border bg-white"
                      style={{
                        borderColor: "var(--border)",
                        color: "var(--text-primary)",
                        fontFamily: "var(--font-sans)",
                      }}
                    />
                  </div>
                </div>
              ))}

              <button
                onClick={() => setDimensions((prev) => [...prev, emptyDimension()])}
                className="w-full py-2 text-xs font-medium rounded-md border border-dashed hover:bg-[var(--surface-elevated)]"
                style={{
                  borderColor: "var(--border-strong)",
                  color: "var(--text-secondary)",
                }}
              >
                + Add Dimension
              </button>
            </div>
          </CardContent>
        </Card>

        {/* Action buttons */}
        <div className="flex items-center gap-2">
          <Button
            onClick={handleGenerate}
            loading={dimensionMutation.isPending}
            disabled={validDimensions.length === 0}
          >
            Generate Checklist
          </Button>
          <Button variant="outline" onClick={handleExportScript} disabled={validDimensions.length === 0}>
            Export Script
          </Button>
        </div>

        {/* Error */}
        {dimensionMutation.isError && (
          <div
            className="rounded-md px-3 py-2 text-xs"
            style={{
              backgroundColor: "var(--error-light)",
              color: "var(--error)",
              border: "1px solid var(--error)",
            }}
          >
            {dimensionMutation.error?.message || "Generation failed"}
          </div>
        )}

        {/* Generation progress */}
        {dimensionMutation.isPending && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="animate-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M12 2a10 10 0 0 1 10 10" style={{ color: "var(--accent-primary)" }} />
                  </svg>
                  <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    Generating Checklist
                  </h3>
                </div>
                <span className="text-xs tabular-nums" style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}>
                  {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
                </span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {validDimensions.map((dim, i) => (
                  <div
                    key={i}
                    className="rounded-md p-2.5 animate-pulse"
                    style={{
                      border: "1px solid var(--border)",
                      backgroundColor: "var(--surface-manila)",
                    }}
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-semibold" style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                        {dim.name}
                      </span>
                      {dim.sub_dimensions.length > 0 && (
                        <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                          {dim.sub_dimensions.length} sub-dimensions
                        </span>
                      )}
                    </div>
                    <div className="space-y-1 ml-3">
                      {Array.from({ length: Math.max(dim.sub_dimensions.length, 2) }, (_, j) => (
                        <div
                          key={j}
                          className="rounded"
                          style={{
                            height: "8px",
                            width: `${60 + Math.random() * 30}%`,
                            backgroundColor: "var(--border)",
                          }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
                Generating checklist items for {validDimensions.length} dimension{validDimensions.length !== 1 ? "s" : ""} using {augmentationMode} mode...
              </p>
            </CardContent>
          </Card>
        )}

        {/* Generated checklist */}
        {generatedChecklist && groupedItems && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <h3
                  className="text-sm font-semibold"
                  style={{ color: "var(--text-primary)" }}
                >
                  Generated Checklist
                  <span
                    className="ml-2 text-xs font-normal"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {generatedChecklist.items.length} items across{" "}
                    {Object.keys(groupedItems).length} dimensions
                  </span>
                </h3>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleSave}
                    disabled={saveStatus === "saving" || saveStatus === "saved"}
                  >
                    {saveStatus === "saving"
                      ? "Saving..."
                      : saveStatus === "saved"
                        ? "Saved!"
                        : saveStatus === "error"
                          ? "Retry Save"
                          : "Save to Library"}
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleExportScript}>
                    Export Script
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(groupedItems).map(([category, items]) => (
                  <DimensionGroup key={category} category={category} items={items} />
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Export Script Dialog */}
      <Dialog open={scriptDialogOpen} onOpenChange={setScriptDialogOpen}>
        <DialogContent
          title="Python Script"
          description="Copy this snippet to run the same generation as a script."
        >
          <div className="relative">
            <div
              className="rounded-md overflow-hidden"
              style={{ border: "1px solid var(--border)" }}
            >
              <Editor
                height={`${Math.min(Math.max(scriptSnippet.split("\n").length * 19 + 24, 120), 400)}px`}
                defaultLanguage="python"
                value={scriptSnippet}
                options={{
                  readOnly: true,
                  minimap: { enabled: false },
                  lineNumbers: "off",
                  glyphMargin: false,
                  folding: false,
                  wordWrap: "on",
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  fontSize: 13,
                  fontFamily: "var(--font-mono), monospace",
                  padding: { top: 12, bottom: 12 },
                  renderLineHighlight: "none",
                  overviewRulerLanes: 0,
                  hideCursorInOverviewRuler: true,
                  scrollbar: { vertical: "auto", horizontal: "hidden", verticalScrollbarSize: 8 },
                }}
                theme="vs"
              />
            </div>
            <div className="flex justify-end mt-4">
              <Button onClick={handleCopyScript} size="sm">
                {copied ? "Copied!" : "Copy to Clipboard"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Collapsible group of checklist items for a single dimension. */
function DimensionGroup({
  category,
  items,
}: {
  category: string;
  items: Array<{ question: string; weight: number }>;
}) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left py-1"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          style={{
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.15s",
            color: "var(--text-tertiary)",
          }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <span
          className="text-xs font-semibold"
          style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
        >
          {category}
        </span>
        <Badge>{items.length}</Badge>
      </button>
      {expanded && (
        <div className="ml-5 space-y-1 mt-1">
          {items.map((item, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs py-0.5"
              style={{ color: "var(--text-secondary)" }}
            >
              <span style={{ color: "var(--text-tertiary)" }}>-</span>
              <span>{item.question}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
