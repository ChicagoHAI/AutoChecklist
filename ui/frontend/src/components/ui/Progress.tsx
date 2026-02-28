"use client";

interface ProgressProps {
  value: number;
  max?: number;
  size?: "sm" | "md";
  color?: string;
  className?: string;
  showLabel?: boolean;
}

export function Progress({
  value,
  max = 100,
  size = "md",
  color,
  className = "",
  showLabel = false,
}: ProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const heights = { sm: "h-1.5", md: "h-2.5" };

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-between mb-1">
          <span className="micro-label" style={{ color: "var(--text-secondary)" }}>
            Progress
          </span>
          <span
            className="text-xs font-medium"
            style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
          >
            {Math.round(percentage)}%
          </span>
        </div>
      )}
      <div
        className={`w-full rounded-md overflow-hidden ${heights[size]}`}
        style={{ backgroundColor: "var(--surface-manila)" }}
      >
        <div
          className={`${heights[size]} rounded-md transition-all duration-300`}
          style={{
            width: `${percentage}%`,
            backgroundColor: color || "var(--accent-primary)",
          }}
        />
      </div>
    </div>
  );
}
