"use client";

interface ScoreProps {
  value: number;
  max: number;
  size?: "sm" | "md" | "lg";
  showPercentage?: boolean;
  /** When set to "weighted" or "normalized", show primary metric below the fraction. */
  primaryMetric?: string;
  /** Primary score as percentage (0-100). Shown when primaryMetric is weighted/normalized. */
  primaryPercentage?: number;
  className?: string;
}

function getScoreColor(percentage: number): string {
  if (percentage >= 80) return "var(--success)";
  if (percentage >= 60) return "var(--warning)";
  return "var(--error)";
}

function getScoreBg(percentage: number): string {
  if (percentage >= 80) return "var(--success-light)";
  if (percentage >= 60) return "var(--warning-light)";
  return "var(--error-light)";
}

export function Score({
  value,
  max,
  size = "md",
  showPercentage = true,
  primaryMetric,
  primaryPercentage,
  className = "",
}: ScoreProps) {
  const passPercentage = max > 0 ? (value / max) * 100 : 0;
  const displayPercentage = primaryPercentage ?? passPercentage;
  const color = getScoreColor(displayPercentage);
  const bgColor = getScoreBg(displayPercentage);
  const showPrimaryLine = primaryMetric && primaryMetric !== "pass" && primaryPercentage != null;

  const sizes = {
    sm: { text: "text-sm", padding: "px-2 py-1" },
    md: { text: "text-lg", padding: "px-3 py-1.5" },
    lg: { text: "text-2xl", padding: "px-4 py-2" },
  };

  const s = sizes[size];

  return (
    <div
      className={`inline-flex flex-col rounded-md ${s.padding} ${className}`}
      style={{ backgroundColor: bgColor }}
    >
      <div className="flex items-center gap-2">
        <span
          className={`${s.text} font-semibold`}
          style={{ color, fontFamily: "var(--font-mono)", fontVariantNumeric: "tabular-nums" }}
        >
          {Math.round(value)}/{Math.round(max)}
        </span>
        {showPercentage && !showPrimaryLine && (
          <span
            className="text-xs font-medium"
            style={{ color, opacity: 0.8, fontFamily: "var(--font-mono)" }}
          >
            ({Math.round(passPercentage)}%)
          </span>
        )}
      </div>
      {showPrimaryLine && (
        <span
          className="text-xs"
          style={{ color, opacity: 0.7, fontFamily: "var(--font-mono)" }}
        >
          {primaryMetric}: {primaryPercentage!.toFixed(1)}%
        </span>
      )}
    </div>
  );
}

export function ScoreCircle({
  percentage,
  size = 48,
  className = "",
}: {
  percentage: number;
  size?: number;
  className?: string;
}) {
  const color = getScoreColor(percentage);
  const radius = (size - 6) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth="3"
          fill="none"
          style={{ stroke: "var(--surface-manila)" }}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth="3"
          fill="none"
          strokeLinecap="butt"
          style={{
            stroke: color,
            strokeDasharray: circumference,
            strokeDashoffset: offset,
            transition: "stroke-dashoffset 500ms ease-out",
          }}
        />
      </svg>
      <span
        className="absolute text-xs font-semibold"
        style={{ color, fontFamily: "var(--font-mono)", fontVariantNumeric: "tabular-nums" }}
      >
        {Math.round(percentage)}
      </span>
    </div>
  );
}
