"use client";

interface TooltipProps {
  content: string;
  children: React.ReactNode;
}

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export function Tooltip({ content, children }: TooltipProps) {
  return (
    <div className="relative group/tooltip inline-block">
      {children}
      <div
        className="fixed z-50 px-3 py-1.5 text-xs rounded-md w-max max-w-56 opacity-0 invisible group-hover/tooltip:opacity-100 group-hover/tooltip:visible pointer-events-none"
        style={{
          backgroundColor: "white",
          color: "var(--text-secondary)",
          fontFamily: "var(--font-mono)",
          boxShadow: "var(--shadow-sm)",
          border: "1px solid var(--border)",
          top: "var(--tt-top)",
          left: "var(--tt-left)",
        }}
        ref={(el) => {
          if (!el) return;
          const parent = el.parentElement;
          if (!parent) return;
          const update = () => {
            const rect = parent.getBoundingClientRect();
            const left = Math.max(8, Math.min(rect.left + rect.width / 2 - el.offsetWidth / 2, window.innerWidth - el.offsetWidth - 8));
            el.style.setProperty("--tt-top", `${rect.bottom + 6}px`);
            el.style.setProperty("--tt-left", `${left}px`);
          };
          parent.addEventListener("mouseenter", update);
          return () => parent.removeEventListener("mouseenter", update);
        }}
      >
        {content}
      </div>
    </div>
  );
}
