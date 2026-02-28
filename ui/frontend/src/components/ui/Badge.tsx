"use client";

interface BadgeProps {
  variant?: "default" | "success" | "error" | "warning" | "info" | "info2";
  children: React.ReactNode;
  className?: string;
}

const variantStyles: Record<string, { bg: string; color: string }> = {
  default: { bg: "var(--surface-elevated)", color: "var(--text-secondary)" },
  success: { bg: "var(--success-light)", color: "var(--success)" },
  error: { bg: "var(--error-light)", color: "var(--error)" },
  warning: { bg: "var(--warning-light)", color: "var(--warning)" },
  info: { bg: "var(--accent-light)", color: "var(--accent-primary)" },
  info2: { bg: "#c46060", color: "white" },
};

export function Badge({ variant = "default", children, className = "" }: BadgeProps) {
  const style = variantStyles[variant];

  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded-md text-xs font-medium uppercase tracking-wide ${className}`}
      style={{
        backgroundColor: style.bg,
        color: style.color,
        fontFamily: "var(--font-mono)",
        fontSize: "0.6875rem",
      }}
    >
      {children}
    </span>
  );
}
