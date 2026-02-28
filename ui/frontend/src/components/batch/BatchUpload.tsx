"use client";

import { useState, useCallback, useRef } from "react";
import Editor from "@monaco-editor/react";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import { Input } from "@/components/ui/Input";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { ModelSelector } from "@/components/layout/ModelSelector";
import { useGenerators, useChecklists, usePipelines, useDefaultModel } from "@/lib/hooks";
import { getPipeline } from "@/lib/api";
import type { PipelineMode, PipelineConfig, ScorerConfig } from "@/lib/types";
import { DEFAULT_SCORER_CONFIG } from "@/lib/types";
import { ScorerConfigPanel } from "@/components/playground/ScorerConfigPanel";

type PipelineSource = "builtin" | "pipeline";

type InputMode = "upload" | "path";

export interface BatchStartParams {
  file?: File;
  filePath?: string;
  method: string;
  scorer?: string;
  scorerConfig?: ScorerConfig;
  provider: string;
  generatorModel: string;
  scorerModel: string;
  candidateModel?: string;
  pipelineMode: PipelineMode;
  checklistId?: string;
  customPrompt?: string;
  pipelineId?: string;
}

const PIPELINE_MODES: { value: PipelineMode; label: string }[] = [
  { value: "full", label: "Full Pipeline" },
  { value: "generate_only", label: "Generate Only" },
  { value: "score_only", label: "Score Only" },
];

function FormatInfo() {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="text-xs font-medium flex items-center gap-1"
        style={{ color: "var(--accent-primary)" }}
      >
        <span>{open ? "▾" : "▸"}</span>
        File format guide
      </button>
      {open && (
        <div
          className="mt-2 rounded-md p-3 text-xs space-y-3"
          style={{ backgroundColor: "var(--surface-elevated)", color: "var(--text-secondary)" }}
        >
          <div>
            <p className="font-medium mb-1" style={{ color: "var(--text-primary)" }}>JSONL (recommended)</p>
            <pre className="rounded p-2 overflow-x-auto" style={{ backgroundColor: "var(--surface)", color: "var(--text-secondary)" }}>
{`{"input": "Write a haiku about autumn", "target": "Leaves fall gently..."}
{"input": "Explain recursion", "target": "Recursion is when..."}`}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1" style={{ color: "var(--text-primary)" }}>JSON (array)</p>
            <pre className="rounded p-2 overflow-x-auto" style={{ backgroundColor: "var(--surface)", color: "var(--text-secondary)" }}>
{`[
  {"input": "Write a haiku", "target": "Leaves fall..."},
  {"input": "Explain recursion", "target": "Recursion is..."}
]`}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1" style={{ color: "var(--text-primary)" }}>CSV</p>
            <pre className="rounded p-2 overflow-x-auto" style={{ backgroundColor: "var(--surface)", color: "var(--text-secondary)" }}>
{`input,target
"Write a haiku about autumn","Leaves fall gently..."
"Explain recursion","Recursion is when..."`}
            </pre>
          </div>
          <p style={{ color: "var(--text-tertiary)" }}>
            <strong>input</strong> = generation instruction/query &nbsp;|&nbsp; <strong>target</strong> = response to evaluate
            &nbsp;|&nbsp; Optional: <strong>reference</strong> = gold reference answer
          </p>
        </div>
      )}
    </div>
  );
}

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-sm font-medium border transition-all ${
        active
          ? "border-[var(--accent-primary)] bg-[var(--accent-light)]"
          : "border-[var(--border)] bg-white hover:border-[var(--border-strong)]"
      }`}
      style={{
        color: active ? "var(--accent-primary)" : "var(--text-secondary)",
      }}
    >
      {children}
    </button>
  );
}

interface BatchUploadProps {
  onUploadAndStart: (params: BatchStartParams) => void;
  onPathAndStart: (params: BatchStartParams) => void;
  loading?: boolean;
}

export function BatchUpload({ onUploadAndStart, onPathAndStart, loading }: BatchUploadProps) {
  const [inputMode, setInputMode] = useState<InputMode>("upload");
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>("full");
  const [file, setFile] = useState<File | null>(null);
  const [filePath, setFilePath] = useState("");
  const [pipelineSource, setPipelineSource] = useState<PipelineSource>("builtin");
  const [method, setMethod] = useState("tick");
  const [scorerMode, setScorerMode] = useState<"auto" | "manual">("auto");
  const [scorerConfig, setScorerConfig] = useState<ScorerConfig>(DEFAULT_SCORER_CONFIG);
  const [checklistId, setChecklistId] = useState("");
  const [selectedPipelineId, setSelectedPipelineId] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scriptDialogOpen, setScriptDialogOpen] = useState(false);
  const [scriptSnippet, setScriptSnippet] = useState("");
  const [copied, setCopied] = useState(false);
  const scriptRef = useRef("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Separate model selectors for generator, scorer, and candidate
  const generatorModel = useDefaultModel();
  const scorerModel = useDefaultModel();
  const candidateModel = useDefaultModel();

  const { generators } = useGenerators();
  const { data: savedChecklists } = useChecklists();
  const { data: savedPipelines } = usePipelines();

  const generatorOptions = generators
    .filter((g) => g.level === "instance")
    .map((g) => ({ value: g.name, label: `${g.name.toUpperCase()} - ${g.description}` }));

  const checklistOptions = [
    { value: "", label: "Select a checklist..." },
    ...(savedChecklists || []).map((cl) => ({
      value: cl.id,
      label: `${cl.name} (${cl.item_count} items)`,
    })),
  ];

  const pipelineOptions = [
    { value: "", label: "Select a pipeline..." },
    ...(savedPipelines || []).map((p) => ({
      value: p.id,
      label: p.name,
    })),
  ];

  const showMethod = pipelineMode !== "score_only";
  const showScorer = pipelineMode !== "generate_only" && (pipelineMode === "score_only" || pipelineSource !== "pipeline");
  const showChecklist = pipelineMode === "score_only";

  // Is the selected method contrastive?
  const selectedGenerator = generators.find((g) => g.name === method);
  const isContrastive = pipelineSource === "builtin"
    ? selectedGenerator?.generator_class === "contrastive"
    : false;
  // For saved pipeline, check if it's contrastive
  const selectedPipeline = savedPipelines?.find((p) => p.id === selectedPipelineId);
  const isPipelineContrastive = selectedPipeline?.generator_class === "contrastive";
  const showCandidateModel = isContrastive || (pipelineSource === "pipeline" && isPipelineContrastive);

  const validateFile = useCallback((f: File): string | null => {
    const validTypes = ["application/json", "text/csv", "text/plain"];
    const validExtensions = [".json", ".csv", ".jsonl"];
    const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));

    if (!validTypes.includes(f.type) && !validExtensions.includes(ext)) {
      return "Invalid file type. Please upload a JSON, JSONL, or CSV file.";
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (f: File) => {
      const validationError = validateFile(f);
      if (validationError) {
        setError(validationError);
        setFile(null);
        return;
      }
      setError(null);
      setFile(f);
    },
    [validateFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) handleFile(droppedFile);
    },
    [handleFile]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (selectedFile) handleFile(selectedFile);
    },
    [handleFile]
  );

  const buildParams = (): BatchStartParams | null => {
    if (showChecklist && !checklistId) {
      setError("Please select a checklist for Score Only mode.");
      return null;
    }
    if (showMethod && pipelineSource === "pipeline" && !selectedPipelineId) {
      setError("Please select a pipeline.");
      return null;
    }

    const selectedChecklist = savedChecklists?.find((cl) => cl.id === checklistId);
    const effectiveMethod = showMethod ? method : (selectedChecklist?.method || "score_only");
    const effectiveScorerConfig = showScorer && (pipelineMode === "score_only" || scorerMode === "manual") ? scorerConfig : undefined;
    const effectiveChecklistId = showChecklist ? checklistId : undefined;
    const effectivePipelineId = showMethod && pipelineSource === "pipeline" ? selectedPipelineId || undefined : undefined;

    return {
      method: effectiveMethod,
      scorerConfig: effectiveScorerConfig,
      provider: generatorModel.provider,
      generatorModel: generatorModel.model,
      scorerModel: scorerModel.model,
      candidateModel: showCandidateModel ? candidateModel.model : undefined,
      pipelineMode,
      checklistId: effectiveChecklistId,
      pipelineId: effectivePipelineId,
    };
  };

  const handleSubmit = () => {
    const params = buildParams();
    if (!params) return;

    if (inputMode === "upload") {
      if (!file) {
        setError("Please select a file first.");
        return;
      }
      onUploadAndStart({ ...params, file });
    } else {
      const trimmed = filePath.trim();
      if (!trimmed) {
        setError("Please enter a file path.");
        return;
      }
      if (!/\.(json|jsonl|csv)$/i.test(trimmed)) {
        setError("File must be .json, .jsonl, or .csv");
        return;
      }
      setError(null);
      onPathAndStart({ ...params, filePath: trimmed });
    }
  };

  const handleExportScript = async () => {
    const genModel = generatorModel.model;
    const scrModel = scorerModel.model;
    const provider = generatorModel.provider;
    const fileName = file?.name || filePath.trim().split("/").pop() || "your_file.jsonl";

    let snippet = "";

    if (pipelineMode === "score_only") {
      // Score-only with saved checklist — use scorer directly
      const clName = savedChecklists?.find((cl) => cl.id === checklistId)?.name || "my_checklist";
      snippet = [
        `from autochecklist import BatchScorer, Checklist`,
        ``,
        `scorer = BatchScorer(`,
        `    model="${scrModel}",`,
        `    provider="${provider}",`,
        `)`,
        ``,
        `checklist = Checklist.load("${clName}.json")`,
        ``,
        `# Score each row against the checklist`,
        `import json`,
        ``,
        `with open("${fileName}") as f:`,
        `    data = [json.loads(line) for line in f]`,
        ``,
        `for row in data:`,
        `    result = scorer.score(checklist, target=row["target"], input=row.get("input"))`,
        `    print(f"Pass rate: {result.pass_rate:.1%}")`,
      ].join("\n");
    } else if (pipelineSource === "pipeline" && selectedPipelineId) {
      // Saved pipeline — fetch config for prompts and class info
      try {
        const config: PipelineConfig = await getPipeline(selectedPipelineId);
        const genClass = config.generator_class === "contrastive" ? "ContrastiveGenerator" : "DirectGenerator";
        const scorerCls = {
          batch: "BatchScorer",
          item: "ItemScorer",
          weighted: "WeightedScorer",
          normalized: "NormalizedScorer",
        }[config.scorer_class] || "BatchScorer";

        const candidateLine = config.generator_class === "contrastive" && showCandidateModel
          ? `\n    candidate_models=["${candidateModel.model}"],`
          : "";

        if (pipelineMode === "generate_only") {
          snippet = [
            `from autochecklist import ${genClass}, ChecklistPipeline`,
            ``,
            `gen = ${genClass}(`,
            `    custom_prompt="""${config.generator_prompt}""",`,
            `    model="${genModel}",`,
            `    provider="${provider}",${candidateLine}`,
            `)`,
            ``,
            `pipe = ChecklistPipeline(generator=gen)`,
            `checklists = pipe.generate_batch("${fileName}", output_path="checklists.jsonl", show_progress=True)`,
            ``,
            `print(f"Generated {len(checklists)} checklists")`,
          ].join("\n");
        } else {
          snippet = [
            `from autochecklist import ${genClass}, ${scorerCls}, ChecklistPipeline`,
            ``,
            `gen = ${genClass}(`,
            `    custom_prompt="""${config.generator_prompt}""",`,
            `    model="${genModel}",`,
            `    provider="${provider}",${candidateLine}`,
            `)`,
            ``,
            `scorer = ${scorerCls}(`,
            `    model="${scrModel}",`,
            `    provider="${provider}",`,
            `)`,
            ``,
            `pipe = ChecklistPipeline(generator=gen, scorer=scorer)`,
            `result = pipe.run_batch("${fileName}", output_path="results.jsonl", show_progress=True)`,
            ``,
            `print(f"Macro pass rate: {result.macro_pass_rate:.1%}")`,
          ].join("\n");
        }
      } catch {
        snippet = "# Error loading pipeline config";
      }
    } else {
      // Built-in method
      const effectiveScorer = scorerMode === "manual" ? `\n    scorer="${scorerConfig.mode}",` : "";
      const candidateKwargs = isContrastive && showCandidateModel
        ? `\n    generator_kwargs={"candidate_models": ["${candidateModel.model}"]},`
        : "";

      if (pipelineMode === "generate_only") {
        snippet = [
          `from autochecklist import pipeline`,
          ``,
          `pipe = pipeline(`,
          `    "${method}",`,
          `    provider="${provider}",`,
          `    generator_model="${genModel}",${candidateKwargs}`,
          `)`,
          ``,
          `checklists = pipe.generate_batch("${fileName}", output_path="checklists.jsonl", show_progress=True)`,
          ``,
          `print(f"Generated {len(checklists)} checklists")`,
        ].join("\n");
      } else {
        snippet = [
          `from autochecklist import pipeline`,
          ``,
          `pipe = pipeline(`,
          `    "${method}",`,
          `    provider="${provider}",`,
          `    generator_model="${genModel}",`,
          `    scorer_model="${scrModel}",${effectiveScorer}${candidateKwargs}`,
          `)`,
          ``,
          `result = pipe.run_batch("${fileName}", output_path="results.jsonl", show_progress=True)`,
          ``,
          `print(f"Macro pass rate: {result.macro_pass_rate:.1%}")`,
        ].join("\n");
      }
    }

    scriptRef.current = snippet;
    setScriptSnippet(snippet);
    setCopied(false);
    setScriptDialogOpen(true);
  };

  const dialogRef = useRef<HTMLDivElement>(null);

  const handleCopyScript = () => {
    const text = scriptRef.current;
    // Append textarea inside the dialog so Radix focus trap doesn't block it
    const container = dialogRef.current || document.body;
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "absolute";
    ta.style.left = "-9999px";
    ta.style.opacity = "0";
    container.appendChild(ta);
    ta.focus();
    ta.select();
    try { document.execCommand("copy"); } catch { /* ignore */ }
    container.removeChild(ta);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <>
      <Card>
        <CardHeader>
          <h2
            className="text-base font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            Upload Dataset
          </h2>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
            Upload a JSON, JSONL, or CSV file with generation input and evaluation target pairs.
          </p>
          <FormatInfo />
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Input mode toggle */}
          <div className="flex gap-2">
            <ToggleButton active={inputMode === "upload"} onClick={() => { setInputMode("upload"); setError(null); }}>
              File Upload
            </ToggleButton>
            <ToggleButton active={inputMode === "path"} onClick={() => { setInputMode("path"); setError(null); }}>
              File Path
            </ToggleButton>
          </div>

          {inputMode === "upload" ? (
            /* Drop zone */
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="relative rounded-md border-2 border-dashed px-6 py-4 text-center cursor-pointer transition-colors"
              style={{
                borderColor: dragActive
                  ? "var(--accent-primary)"
                  : file
                    ? "var(--success)"
                    : "var(--border)",
                backgroundColor: dragActive
                  ? "var(--accent-light)"
                  : file
                    ? "var(--success-light)"
                    : "transparent",
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,.jsonl,.csv"
                onChange={handleFileInput}
                className="hidden"
              />

              {file ? (
                <div>
                  <svg
                    width="20" height="20" viewBox="0 0 24 24" fill="none"
                    stroke="var(--success)" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round"
                    className="mx-auto mb-1.5"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <polyline points="9 15 11 17 15 13" />
                  </svg>
                  <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    {file.name}
                  </p>
                  <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                    {formatFileSize(file.size)}
                  </p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                      setError(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                    className="text-xs mt-2 underline"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div>
                  <svg
                    width="20" height="20" viewBox="0 0 24 24" fill="none"
                    stroke="var(--text-tertiary)" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round"
                    className="mx-auto mb-1.5"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <p className="text-xs font-medium" style={{ color: "var(--text-primary)" }}>
                    Drop your file here, or{" "}
                    <span style={{ color: "var(--accent-primary)" }}>browse</span>
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-tertiary)" }}>
                    JSON, JSONL, CSV
                  </p>
                </div>
              )}
            </div>
          ) : (
            /* File path input */
            <div>
              <Input
                label="Path to file on this machine"
                placeholder="/path/to/dataset.jsonl"
                value={filePath}
                onChange={(e) => { setFilePath(e.target.value); setError(null); }}
              />
              <p className="text-xs mt-1.5" style={{ color: "var(--text-tertiary)" }}>
                Absolute or relative path to a .json, .jsonl, or .csv file. No size limit.
              </p>
            </div>
          )}

          {error && (
            <p className="text-xs" style={{ color: "var(--error)" }}>
              {error}
            </p>
          )}

          {/* Settings expander — open by default */}
          <details className="group" open>
            <summary
              className="flex items-center gap-1.5 text-sm font-medium cursor-pointer select-none list-none"
              style={{ color: "var(--text-secondary)" }}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                className="transition-transform group-open:rotate-90"
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
              Settings
            </summary>
            <div className="space-y-4 pt-3">
              {/* Pipeline mode toggle */}
              <div>
                <label
                  className="block text-sm font-medium mb-2"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Evaluation Mode
                </label>
                <div className="flex gap-2">
                  {PIPELINE_MODES.map((mode) => (
                    <ToggleButton
                      key={mode.value}
                      active={pipelineMode === mode.value}
                      onClick={() => { setPipelineMode(mode.value); setError(null); }}
                    >
                      {mode.label}
                    </ToggleButton>
                  ))}
                </div>
              </div>

              {/* Pipeline Source — shown when method is relevant */}
              {showMethod && (
                <div>
                  <label
                    className="block text-sm font-medium mb-2"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Pipeline Source
                  </label>
                  <div className="flex gap-2 mb-3">
                    <ToggleButton
                      active={pipelineSource === "builtin"}
                      onClick={() => setPipelineSource("builtin")}
                    >
                      Built-in
                    </ToggleButton>
                    <ToggleButton
                      active={pipelineSource === "pipeline"}
                      onClick={() => setPipelineSource("pipeline")}
                    >
                      From Library
                    </ToggleButton>
                  </div>

                  {pipelineSource === "builtin" ? (
                    <Select
                      label=""
                      options={
                        generatorOptions.length > 0
                          ? generatorOptions
                          : [{ value: "tick", label: "TICK" }]
                      }
                      value={method}
                      onChange={(e) => setMethod(e.target.value)}
                    />
                  ) : (
                    <Select
                      label=""
                      options={pipelineOptions}
                      value={selectedPipelineId}
                      onChange={(e) => setSelectedPipelineId(e.target.value)}
                    />
                  )}
                </div>
              )}

              {/* Checklist selector — score_only mode */}
              {showChecklist && (
                <Select
                  label="Checklist"
                  options={checklistOptions}
                  value={checklistId}
                  onChange={(e) => setChecklistId(e.target.value)}
                />
              )}

              {/* Scorer config — hidden in generate_only and when using saved pipeline */}
              {showScorer && (
                <div>
                  <label
                    className="block text-sm font-medium mb-2"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Scorer
                  </label>
                  {pipelineMode === "score_only" ? (
                    <ScorerConfigPanel
                      value={scorerConfig}
                      onChange={setScorerConfig}
                    />
                  ) : (
                    <>
                      <div className="flex gap-2 mb-2">
                        <ToggleButton
                          active={scorerMode === "auto"}
                          onClick={() => setScorerMode("auto")}
                        >
                          Auto
                        </ToggleButton>
                        <ToggleButton
                          active={scorerMode === "manual"}
                          onClick={() => setScorerMode("manual")}
                        >
                          Custom
                        </ToggleButton>
                      </div>
                      {scorerMode === "manual" && (
                        <ScorerConfigPanel
                          value={scorerConfig}
                          onChange={setScorerConfig}
                        />
                      )}
                      {scorerMode === "auto" && (
                        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                          Uses the default scorer config for the selected method.
                        </p>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* Model Settings — nested collapsible */}
              <details className="group/models">
                <summary
                  className="flex items-center gap-1.5 text-sm font-medium cursor-pointer select-none list-none"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="transition-transform group-open/models:rotate-90"
                  >
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                  Model Settings
                </summary>
                <div className="space-y-3 pt-3">
                  {pipelineMode !== "score_only" && (
                    <div>
                      <label
                        className="block text-xs font-medium mb-1"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Generator Model
                      </label>
                      <ModelSelector
                        provider={generatorModel.provider}
                        model={generatorModel.model}
                        onProviderChange={generatorModel.setProvider}
                        onModelChange={generatorModel.setModel}
                      />
                    </div>
                  )}

                  {showCandidateModel && (
                    <div>
                      <label
                        className="block text-xs font-medium mb-1"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Candidate Model
                      </label>
                      <ModelSelector
                        provider={candidateModel.provider}
                        model={candidateModel.model}
                        onProviderChange={candidateModel.setProvider}
                        onModelChange={candidateModel.setModel}
                      />
                    </div>
                  )}

                  {pipelineMode !== "generate_only" && (
                    <div>
                      <label
                        className="block text-xs font-medium mb-1"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Scorer Model
                      </label>
                      <ModelSelector
                        provider={scorerModel.provider}
                        model={scorerModel.model}
                        onProviderChange={scorerModel.setProvider}
                        onModelChange={scorerModel.setModel}
                      />
                    </div>
                  )}
                </div>
              </details>
            </div>
          </details>

          {/* Submit + Export buttons */}
          <div className="flex gap-2">
            <Button
              onClick={handleSubmit}
              disabled={(inputMode === "upload" ? !file : !filePath.trim()) || loading}
              loading={loading}
              size="lg"
              className="flex-1"
            >
              {inputMode === "upload" ? "Upload & Start Evaluation" : "Start Evaluation"}
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={handleExportScript}
            >
              Export Script
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Export Script Dialog */}
      <Dialog open={scriptDialogOpen} onOpenChange={setScriptDialogOpen}>
        <DialogContent title="Python Script" description="Copy this snippet to run the same evaluation as a script.">
          <div ref={dialogRef} className="relative">
          <div className="rounded-md overflow-hidden" style={{ border: "1px solid var(--border)" }}>
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
