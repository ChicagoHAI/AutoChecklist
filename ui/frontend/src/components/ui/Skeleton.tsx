"use client";

interface SkeletonProps {
  className?: string;
  width?: string;
  height?: string;
}

export function Skeleton({ className = "", width, height }: SkeletonProps) {
  return (
    <div
      className={`rounded-md animate-skeleton ${className}`}
      style={{
        backgroundColor: "var(--surface-manila)",
        width: width || "100%",
        height: height || "1rem",
      }}
    />
  );
}

export function SkeletonText({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          width={i === lines - 1 ? "60%" : "100%"}
          height="0.875rem"
        />
      ))}
    </div>
  );
}

export function SkeletonCard() {
  return (
    <div
      className="rounded-lg border p-4 space-y-3"
      style={{ borderColor: "var(--border-strong)", backgroundColor: "white" }}
    >
      <Skeleton height="1.25rem" width="40%" />
      <SkeletonText lines={3} />
      <Skeleton height="2rem" width="30%" />
    </div>
  );
}
