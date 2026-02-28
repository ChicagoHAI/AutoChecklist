"use client";

import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { getBatchExportUrl } from "@/lib/api";
import type { BatchSummary, BatchResult } from "@/lib/types";

interface BatchResultsProps {
  batch: BatchSummary;
  results: BatchResult[];
  onBack: () => void;
}

function buildDistribution(results: BatchResult[]) {
  const buckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}%`,
    count: 0,
    rangeStart: i * 10,
  }));

  for (const r of results) {
    const score = r.primary_score ?? r.pass_rate;
    if (score == null) continue;
    const pct = score * 100;
    const idx = Math.min(Math.floor(pct / 10), 9);
    buckets[idx].count += 1;
  }

  return buckets;
}

/** True when the batch has scoring data (full or score-only pipeline). */
function hasScores(batch: BatchSummary): boolean {
  const mode = batch.config?.pipeline_mode;
  return mode !== "generate_only";
}

interface QuestionStat {
  item_id: string;
  question: string;
  pass_rate: number;
  yes: number;
  no: number;
  total: number;
}

/**
 * Aggregate per-question pass rates across all results.
 * Only applicable for corpus-level batches (shared checklist across all rows).
 */
function buildQuestionStats(
  results: BatchResult[],
): QuestionStat[] | null {
  // Detect shared checklist by checking if results share the same item_ids
  const scoredResults = results.filter(
    (r) => r.item_scores && r.item_scores.length > 0
  );
  if (scoredResults.length < 2) return null;

  const ids1 = new Set(scoredResults[0].item_scores!.map((s) => s.item_id));
  const ids2 = new Set(scoredResults[1].item_scores!.map((s) => s.item_id));
  const shared =
    ids1.size === ids2.size && [...ids1].every((id) => ids2.has(id));
  if (!shared) return null;

  // Aggregate
  const map = new Map<
    string,
    { question: string; yes: number; no: number; na: number; total: number }
  >();

  for (const r of scoredResults) {
    for (const item of r.item_scores!) {
      let entry = map.get(item.item_id);
      if (!entry) {
        entry = { question: item.question || item.item_id, yes: 0, no: 0, na: 0, total: 0 };
        map.set(item.item_id, entry);
      }
      const ans = item.answer.toLowerCase();
      if (ans === "yes") entry.yes++;
      else entry.no++;
      entry.total++;
    }
  }

  const stats: QuestionStat[] = [];
  for (const [item_id, e] of map) {
    stats.push({
      item_id,
      question: e.question,
      pass_rate: e.total > 0 ? e.yes / e.total : 0,
      yes: e.yes,
      no: e.no,
      total: e.total,
    });
  }

  return stats.sort((a, b) => b.pass_rate - a.pass_rate);
}

export function BatchResults({ batch, results, onBack }: BatchResultsProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [distributionOpen, setDistributionOpen] = useState(true);
  const scored = hasScores(batch);
  const distribution = useMemo(() => buildDistribution(results), [results]);
  const questionStats = useMemo(() => buildQuestionStats(results), [results]);

  const handleExport = (format: "json" | "csv") => {
    const url = getBatchExportUrl(batch.batch_id, format);
    window.open(url, "_blank");
  };

  return (
    <div className="space-y-6">
      {/* Summary stats */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h2
                className="text-base font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Results Summary
              </h2>
              {batch.config?.filename && (
                <span
                  className="text-xs"
                  style={{
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {batch.config.filename}
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => handleExport("csv")}>
                Export CSV
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleExport("json")}>
                Export JSON
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className={`grid gap-6 ${scored ? "grid-cols-6" : "grid-cols-3"}`}>
            {scored && (
              <>
                <SummaryStatBlock
                  label="Mean Score"
                  value={
                    batch.mean_score != null
                      ? `${(batch.mean_score * 100).toFixed(1)}%`
                      : "--"
                  }
                  accent
                />
                <SummaryStatBlock
                  label="Macro Pass Rate"
                  value={
                    batch.macro_pass_rate != null
                      ? `${(batch.macro_pass_rate * 100).toFixed(1)}%`
                      : "--"
                  }
                />
                <SummaryStatBlock
                  label="Micro Pass Rate"
                  value={
                    batch.micro_pass_rate != null
                      ? `${(batch.micro_pass_rate * 100).toFixed(1)}%`
                      : "--"
                  }
                />
              </>
            )}
            <SummaryStatBlock label="Total Items" value={String(batch.total)} />
            <SummaryStatBlock
              label="Completed"
              value={String(batch.completed)}
            />
            <SummaryStatBlock
              label="Failed"
              value={String(batch.failed)}
              color={batch.failed > 0 ? "var(--error)" : undefined}
            />
          </div>

          {/* Config info */}
          {batch.config && (
            <div
              className="text-xs flex items-center gap-4 flex-wrap mt-4 pt-4"
              style={{
                color: "var(--text-tertiary)",
                borderTop: "1px solid var(--border)",
              }}
            >
              <span>
                {batch.config.pipeline_name
                  ? "Pipeline"
                  : batch.config.pipeline_mode === "score_only" && batch.config.checklist_name
                  ? "Checklist"
                  : "Method"}
                :{" "}
                <strong style={{ color: "var(--text-secondary)" }}>
                  {batch.config.pipeline_name
                    || batch.config.checklist_name
                    || batch.config.method?.toUpperCase()
                    || "--"}
                </strong>
              </span>
              {batch.config.pipeline_mode !== "score_only" && (
                <span>
                  Generator:{" "}
                  <strong style={{ color: "var(--text-secondary)" }}>
                    {batch.config.generator_model || batch.config.model || "--"}
                  </strong>
                </span>
              )}
              {batch.config.pipeline_mode !== "generate_only" && (
                <span>
                  Scorer:{" "}
                  <strong style={{ color: "var(--text-secondary)" }}>
                    {batch.config.scorer_model || batch.config.model || "--"}
                  </strong>
                </span>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error message for failed batches */}
      {batch.error && (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="2" className="flex-shrink-0 mt-0.5">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" strokeLinecap="round" />
                <line x1="9" y1="9" x2="15" y2="15" strokeLinecap="round" />
              </svg>
              <div>
                <p className="text-sm font-medium" style={{ color: "var(--error)" }}>
                  Batch Failed
                </p>
                <p
                  className="text-sm mt-1 whitespace-pre-wrap"
                  style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}
                >
                  {batch.error}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Distribution chart — only for scored results */}
      {scored && results.length > 0 && (
        <Card>
          <CardHeader>
            <button
              type="button"
              onClick={() => setDistributionOpen((v) => !v)}
              className="flex items-center justify-between w-full text-left"
            >
              <h2
                className="text-base font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Score Distribution
              </h2>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--text-tertiary)"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="transition-transform"
                style={{ transform: distributionOpen ? "rotate(180deg)" : "rotate(0deg)" }}
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          </CardHeader>
          {distributionOpen && (
            <CardContent>
              <div style={{ width: "100%", height: 260 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={distribution}
                    margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="var(--border)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="range"
                      tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                      axisLine={{ stroke: "var(--border)" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                      axisLine={false}
                      tickLine={false}
                      allowDecimals={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "white",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                      formatter={(value) => [value, "Items"]}
                    />
                    <Bar
                      dataKey="count"
                      fill="var(--accent-primary)"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={48}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          )}
        </Card>
      )}

      {/* Question breakdown — only for corpus-level (shared checklist) batches */}
      {questionStats && questionStats.length > 0 && (
        <QuestionBreakdown stats={questionStats} />
      )}

      {/* Results table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h2
              className="text-base font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Individual Results
            </h2>
            <span
              className="text-xs"
              style={{ color: "var(--text-tertiary)" }}
            >
              {results.length} items
            </span>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr
                  style={{
                    borderBottom: "1px solid var(--border)",
                    backgroundColor: "var(--surface-elevated)",
                  }}
                >
                  <th
                    className="text-left px-5 py-3 text-xs font-medium"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    #
                  </th>
                  <th
                    className="text-left px-5 py-3 text-xs font-medium"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Input
                  </th>
                  <th
                    className="text-left px-5 py-3 text-xs font-medium"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Target
                  </th>
                  <th
                    className="text-right px-5 py-3 text-xs font-medium"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    {scored ? "Pass Rate" : "Items"}
                  </th>
                  <th
                    className="text-center px-5 py-3 text-xs font-medium"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Details
                  </th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <ResultRow
                    key={result.index}
                    result={result}
                    scored={scored}
                    expanded={expandedIndex === result.index}
                    onToggle={() =>
                      setExpandedIndex(
                        expandedIndex === result.index ? null : result.index
                      )
                    }
                  />
                ))}
                {results.length === 0 && (
                  <tr>
                    <td
                      colSpan={5}
                      className="px-5 py-12 text-center text-sm"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      No results available.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Back button */}
      <div>
        <Button variant="outline" onClick={onBack}>
          Back to Batch Evaluation
        </Button>
      </div>
    </div>
  );
}

function ResultRow({
  result,
  scored,
  expanded,
  onToggle,
}: {
  result: BatchResult;
  scored: boolean;
  expanded: boolean;
  onToggle: () => void;
}) {
  const displayScore = result.primary_score ?? result.pass_rate;
  const passRateColor =
    displayScore != null
      ? displayScore >= 0.7
        ? "var(--success)"
        : displayScore >= 0.4
          ? "var(--warning)"
          : "var(--error)"
      : "var(--text-tertiary)";

  const truncate = (text: string, maxLen: number) =>
    text.length > maxLen ? text.slice(0, maxLen) + "..." : text;

  return (
    <>
      <tr
        style={{ borderBottom: "1px solid var(--border)" }}
        className="hover:bg-[var(--surface-elevated)] transition-colors"
      >
        <td
          className="px-5 py-3"
          style={{
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-mono)",
            fontSize: "12px",
          }}
        >
          {result.index + 1}
        </td>
        <td
          className="px-5 py-3 max-w-xs"
          style={{ color: "var(--text-primary)" }}
        >
          <span title={result.input}>
            {truncate(result.input, 80)}
          </span>
        </td>
        <td
          className="px-5 py-3 max-w-xs"
          style={{ color: "var(--text-secondary)" }}
        >
          <span title={result.target}>
            {truncate(result.target, 60)}
          </span>
        </td>
        <td
          className="px-5 py-3 text-right font-medium"
          style={{
            color: passRateColor,
            fontFamily: "var(--font-mono)",
          }}
        >
          {scored && displayScore != null
            ? `${(displayScore * 100).toFixed(1)}%`
            : !scored && result.checklist_items
              ? `${result.checklist_items.length} items`
              : "--"}
        </td>
        <td className="px-5 py-3 text-center">
          <button
            onClick={onToggle}
            className="text-xs underline"
            style={{ color: "var(--accent-primary)" }}
          >
            {expanded ? "Hide" : "Show"}
          </button>
        </td>
      </tr>
      {expanded && (
        <tr style={{ borderBottom: "1px solid var(--border)" }}>
          <td colSpan={5} className="px-5 py-4">
            <div
              className="rounded-md p-4 space-y-3"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            >
              <div>
                <p
                  className="text-xs font-medium mb-1"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Full Input
                </p>
                <p
                  className="text-sm whitespace-pre-wrap"
                  style={{ color: "var(--text-primary)" }}
                >
                  {result.input}
                </p>
              </div>
              <div>
                <p
                  className="text-xs font-medium mb-1"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Full Target
                </p>
                <p
                  className="text-sm whitespace-pre-wrap"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {result.target}
                </p>
              </div>

              {/* Scored items (full pipeline or score-only) */}
              {result.item_scores && result.item_scores.length > 0 && (
                <div>
                  <p
                    className="text-xs font-medium mb-2"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Scored Items
                  </p>
                  <div className="space-y-2">
                    {result.item_scores.map((item, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 text-xs"
                      >
                        <Badge
                          variant={
                            item.answer.toLowerCase() === "yes"
                              ? "success"
                              : "error"
                          }
                        >
                          {item.answer}
                        </Badge>
                        <div>
                          <span style={{ color: "var(--text-primary)" }}>
                            {item.question || item.item_id}
                          </span>
                          {item.reasoning && (
                            <p
                              className="mt-0.5"
                              style={{ color: "var(--text-tertiary)" }}
                            >
                              {item.reasoning}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Generated checklist items (generate-only) */}
              {result.checklist_items && result.checklist_items.length > 0 && (
                <div>
                  <p
                    className="text-xs font-medium mb-2"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Generated Checklist ({result.checklist_items.length} items)
                  </p>
                  <div className="space-y-1.5">
                    {result.checklist_items.map((ci, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 text-xs"
                      >
                        <span
                          className="flex-shrink-0 w-5 text-right"
                          style={{
                            color: "var(--text-tertiary)",
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {i + 1}.
                        </span>
                        <span style={{ color: "var(--text-primary)" }}>
                          {ci.question}
                        </span>
                        {ci.weight !== 1 && (
                          <span
                            className="flex-shrink-0"
                            style={{
                              color: "var(--text-tertiary)",
                              fontFamily: "var(--font-mono)",
                            }}
                          >
                            w={ci.weight}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function QuestionBreakdown({ stats }: { stats: QuestionStat[] }) {
  const showSplit = stats.length > 10;
  const top5 = stats.slice(0, 5);
  const bottom5 = [...stats].reverse().slice(0, 5);

  const rateColor = (rate: number) =>
    rate >= 0.7
      ? "var(--success)"
      : rate >= 0.4
        ? "var(--warning)"
        : "var(--error)";

  const QuestionRow = ({ stat }: { stat: QuestionStat }) => (
    <div className="flex items-center gap-3 py-1.5">
      <span
        className="flex-shrink-0 w-14 text-right text-xs font-medium"
        style={{
          color: rateColor(stat.pass_rate),
          fontFamily: "var(--font-mono)",
        }}
      >
        {(stat.pass_rate * 100).toFixed(1)}%
      </span>
      <div className="flex-1 min-w-0">
        <div
          className="h-1.5 rounded-full mb-1"
          style={{ backgroundColor: "var(--border)" }}
        >
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${stat.pass_rate * 100}%`,
              backgroundColor: rateColor(stat.pass_rate),
            }}
          />
        </div>
        <p
          className="text-xs truncate"
          style={{ color: "var(--text-primary)" }}
          title={stat.question}
        >
          {stat.question}
        </p>
      </div>
      <span
        className="flex-shrink-0 text-xs"
        style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}
      >
        {stat.yes}/{stat.yes + stat.no}
      </span>
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h2
            className="text-base font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            Question Breakdown
          </h2>
          <span
            className="text-xs"
            style={{ color: "var(--text-tertiary)" }}
          >
            {stats.length} questions
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {showSplit ? (
          <div className="grid grid-cols-2 gap-8">
            <div>
              <p
                className="text-xs font-medium mb-2"
                style={{ color: "var(--success)" }}
              >
                Top Scored
              </p>
              {top5.map((s) => (
                <QuestionRow key={s.item_id} stat={s} />
              ))}
            </div>
            <div>
              <p
                className="text-xs font-medium mb-2"
                style={{ color: "var(--error)" }}
              >
                Low Scored
              </p>
              {bottom5.map((s) => (
                <QuestionRow key={s.item_id} stat={s} />
              ))}
            </div>
          </div>
        ) : (
          <div>
            {stats.map((s) => (
              <QuestionRow key={s.item_id} stat={s} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SummaryStatBlock({
  label,
  value,
  color,
  accent,
}: {
  label: string;
  value: string;
  color?: string;
  accent?: boolean;
}) {
  return (
    <div>
      <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
        {label}
      </p>
      <p
        className="text-xl font-semibold mt-0.5"
        style={{
          color: color || (accent ? "var(--accent-primary)" : "var(--text-primary)"),
          fontFamily: "var(--font-mono)",
        }}
      >
        {value}
      </p>
    </div>
  );
}
