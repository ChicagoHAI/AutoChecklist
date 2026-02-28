"use client";

import { ChecklistItem } from "@/lib/types";

/**
 * Map NormalizedScorer confidence (P(Yes) probability) to a human-readable label.
 * Bands: <0.2 → High No, 0.2–0.4 → Moderate No, 0.4–0.6 → Unsure,
 *         0.6–0.8 → Moderate Yes, >0.8 → High Yes
 */
function confidenceLabel(confidence: number): { text: string; color: string } {
  if (confidence < 0.2) return { text: "High confidence", color: "var(--error)" };
  if (confidence < 0.4) return { text: "Moderate confidence", color: "var(--error)" };
  if (confidence < 0.6) return { text: "Unsure", color: "var(--text-tertiary)" };
  if (confidence < 0.8) return { text: "Moderate confidence", color: "var(--success)" };
  return { text: "High confidence", color: "var(--success)" };
}

interface ChecklistDisplayProps {
  items: ChecklistItem[];
  showWeights?: boolean;
  showPassFail?: boolean;
}

export function ChecklistDisplay({
  items,
  showWeights = false,
  showPassFail = true,
}: ChecklistDisplayProps) {
  if (!items || items.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        No checklist items generated
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {items.map((item, index) => (
        <li key={index} className="flex items-start gap-2">
          {/* Pass/fail indicator */}
          <span
            className="flex-shrink-0 mt-0.5 w-5 h-5 flex items-center justify-center rounded text-xs"
            style={{
              fontFamily: "var(--font-mono)",
              fontWeight: 600,
              backgroundColor:
                !showPassFail
                  ? "var(--surface-elevated)"
                  : item.passed === true
                  ? "var(--success-light)"
                  : item.passed === false
                  ? "var(--error-light)"
                  : "var(--surface-elevated)",
              color:
                !showPassFail
                  ? "var(--text-tertiary)"
                  : item.passed === true
                  ? "var(--success)"
                  : item.passed === false
                  ? "var(--error)"
                  : "var(--text-tertiary)",
            }}
          >
            {!showPassFail
              ? "\u00A0"
              : item.passed === true
              ? "\u2713"
              : item.passed === false
              ? "\u2717"
              : "\u00A0"}
          </span>

          <div className="flex-1 min-w-0">
            <p
              className="text-sm leading-relaxed"
              style={{ color: "var(--text-primary)" }}
            >
              {item.question}
            </p>

            {item.reasoning && (
              <p
                className="text-xs mt-1 leading-relaxed"
                style={{ color: "var(--text-tertiary)" }}
              >
                {item.reasoning}
              </p>
            )}

            {(showWeights || item.confidence != null) && (
              <div className="flex items-center gap-3 mt-1">
                {showWeights && item.weight !== 1 && (
                  <span
                    className="text-xs"
                    style={{
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    w={item.weight}
                  </span>
                )}
                {item.confidence !== undefined &&
                  item.confidence !== null && (() => {
                    const label = confidenceLabel(item.confidence);
                    return (
                      <span
                        className="text-xs"
                        style={{
                          color: label.color,
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        {label.text}
                      </span>
                    );
                  })()}
              </div>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
