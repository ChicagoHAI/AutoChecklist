"use client";

import { useState } from "react";
import {
  MethodResult,
  MethodName,
  METHOD_LABELS,
  METHOD_DESCRIPTIONS,
  REQUIRES_REFERENCE,
  Checklist,
  ScoreResult,
  mergeScoreIntoItems,
} from "@/lib/types";
import { createChecklist } from "@/lib/api";
import { Card, CardContent } from "./ui/Card";
import { ChecklistDisplay } from "./ChecklistDisplay";
import { Badge } from "./ui/Badge";

// Partial result for streaming
interface PartialMethodResult {
  checklist?: Checklist;
  score?: ScoreResult;
  loading?: "checklist" | "score";
  error?: string;
}

interface MethodCardProps {
  method: string;
  result?: MethodResult | null;
  partialResult?: PartialMethodResult;
  hasReference: boolean;
  loading?: boolean;
  scoringLoading?: boolean;
  figureNumber?: number;
  customLabel?: string;
}

export function MethodCard({
  method,
  result,
  partialResult,
  hasReference,
  loading = false,
  scoringLoading = false,
  figureNumber,
  customLabel,
}: MethodCardProps) {
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  const label = customLabel || METHOD_LABELS[method as MethodName] || method;
  const description = METHOD_DESCRIPTIONS[method as MethodName] || "";
  const requiresRef = REQUIRES_REFERENCE[method as MethodName];
  const isDisabled = requiresRef && !hasReference;
  const showWeights = method.includes("rlcf");

  // Use partial result if no full result
  const checklist = result?.checklist || partialResult?.checklist;
  const score = result?.score || partialResult?.score;
  const hasChecklist = !!checklist;
  const hasScore = !!score;
  const hasError = !!partialResult?.error;

  // Always show pass count as fraction; weighted/normalized percentage shown below
  const scoreFraction = score
    ? `${Math.round(score.score)}/${Math.round(score.max_score)}`
    : null;
  const showPrimaryPercent = score?.primary_metric && score.primary_metric !== "pass";

  const handleSaveToLibrary = async () => {
    if (!checklist) return;
    setSaveStatus("saving");
    try {
      await createChecklist({
        name: `${label} - ${new Date().toLocaleDateString()}`,
        method: method,
        level: "instance",
        items: checklist.items.map((item) => ({
          question: item.question,
          weight: item.weight ?? 1,
        })),
        metadata: checklist.metadata || {},
      });
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus("idle"), 2000);
    } catch {
      setSaveStatus("error");
      setTimeout(() => setSaveStatus("idle"), 2000);
    }
  };

  return (
    <Card
      className={`min-w-[280px] max-w-[340px] flex-shrink-0 ${
        isDisabled ? "opacity-50" : ""
      }`}
    >
      {/* Header */}
      <div
        className="px-5 py-4 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center justify-between mb-1">
          {figureNumber && (
            <span
              className="text-xs font-medium uppercase tracking-wide"
              style={{ color: "var(--text-tertiary)" }}
            >
              Method {figureNumber}
            </span>
          )}
          {hasScore && !loading && !scoringLoading && (
            <Badge variant="success">Complete</Badge>
          )}
          {hasChecklist && scoringLoading && (
            <Badge variant="warning">Scoring...</Badge>
          )}
          {loading && (
            <Badge variant="default">Generating...</Badge>
          )}
          {hasError && !loading && (
            <Badge variant="error">Failed</Badge>
          )}
          {isDisabled && !loading && (
            <Badge variant="warning">Needs Reference</Badge>
          )}
        </div>
        <h3
          className="text-base font-semibold"
          style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
        >
          {label}
        </h3>
        <p
          className="text-xs mt-0.5"
          style={{ color: "var(--text-tertiary)" }}
        >
          {description}
        </p>
      </div>

      <CardContent>
        {loading ? (
          <div className="space-y-3">
            <div
              className="h-4 rounded animate-skeleton"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            />
            <div
              className="h-4 rounded animate-skeleton w-3/4"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            />
            <div
              className="h-4 rounded animate-skeleton w-1/2"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            />
          </div>
        ) : isDisabled ? (
          <p
            className="text-sm"
            style={{ color: "var(--text-tertiary)" }}
          >
            Provide a reference target to enable this method.
          </p>
        ) : hasError ? (
          <div
            className="p-3 rounded-md border"
            style={{
              backgroundColor: "var(--error-light)",
              borderColor: "var(--error)",
            }}
          >
            <div className="flex items-start gap-2">
              <svg
                className="w-4 h-4 flex-shrink-0 mt-0.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                style={{ color: "var(--error)" }}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div>
                <p
                  className="text-xs font-medium"
                  style={{ color: "var(--error)" }}
                >
                  Evaluation failed
                </p>
                <p
                  className="text-xs mt-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {partialResult?.error}
                </p>
              </div>
            </div>
          </div>
        ) : hasChecklist ? (
          <div className="space-y-4">
            {/* Score display */}
            <div
              className="flex items-center justify-between p-3 rounded-md"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            >
              <span
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Score
              </span>
              {scoringLoading ? (
                <span
                  className="text-sm animate-pulse"
                  style={{
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-tertiary)",
                  }}
                >
                  Scoring...
                </span>
              ) : hasScore ? (
                <div className="flex flex-col items-end">
                  <span
                    className="text-xl font-semibold"
                    style={{
                      fontFamily: "var(--font-mono)",
                      color: "var(--accent-primary)",
                    }}
                  >
                    {scoreFraction}
                  </span>
                  {showPrimaryPercent && (
                    <span
                      className="text-xs"
                      style={{
                        fontFamily: "var(--font-mono)",
                        color: "var(--text-tertiary)",
                      }}
                    >
                      {score!.primary_metric}: {score!.percentage.toFixed(1)}%
                    </span>
                  )}
                </div>
              ) : (
                <span
                  className="text-sm"
                  style={{
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-tertiary)",
                  }}
                >
                  —/{checklist.items.length}
                </span>
              )}
            </div>

            <hr />

            {/* Checklist items */}
            <div>
              <h4
                className="text-xs uppercase tracking-wide mb-3 font-medium"
                style={{ color: "var(--text-tertiary)" }}
              >
                Checklist ({checklist.items.length} items)
              </h4>
              <ChecklistDisplay
                items={mergeScoreIntoItems(checklist.items, score)}
                showWeights={showWeights}
                showPassFail={hasScore}
              />
            </div>

            {/* Save to Library */}
            {hasScore && (
              <button
                onClick={handleSaveToLibrary}
                disabled={saveStatus === "saving" || saveStatus === "saved"}
                className="w-full text-xs py-1.5 rounded-md border transition-all disabled:opacity-50"
                style={{
                  borderColor: saveStatus === "saved" ? "var(--success)" : "var(--border)",
                  color: saveStatus === "saved" ? "var(--success)" : saveStatus === "error" ? "var(--error)" : "var(--text-tertiary)",
                  backgroundColor: saveStatus === "saved" ? "var(--success-light)" : "transparent",
                }}
              >
                {saveStatus === "saving" ? "Saving..." : saveStatus === "saved" ? "Saved to Library" : saveStatus === "error" ? "Save failed" : "Save to Library"}
              </button>
            )}
          </div>
        ) : (
          <p
            className="text-sm"
            style={{ color: "var(--text-tertiary)" }}
          >
            Awaiting evaluation...
          </p>
        )}
      </CardContent>
    </Card>
  );
}
