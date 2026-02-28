"use client";

import * as RadixDialog from "@radix-ui/react-dialog";

interface DialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
}

export function Dialog({ open, onOpenChange, children }: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      {children}
    </RadixDialog.Root>
  );
}

export function DialogTrigger({ children }: { children: React.ReactNode }) {
  return <RadixDialog.Trigger asChild>{children}</RadixDialog.Trigger>;
}

export function DialogContent({
  title,
  description,
  children,
  className = "",
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <RadixDialog.Portal>
      <RadixDialog.Overlay
        className="fixed inset-0 z-50"
        style={{ backgroundColor: "rgba(0, 0, 0, 0.4)" }}
      />
      <RadixDialog.Content
        className={`fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg border border-[var(--text-primary)] w-full max-w-lg max-h-[85vh] overflow-y-auto p-5 ${className}`}
        style={{ boxShadow: "var(--shadow-lg)" }}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <RadixDialog.Title
              className="text-lg font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {title}
            </RadixDialog.Title>
            {description && (
              <RadixDialog.Description
                className="text-sm mt-1"
                style={{ color: "var(--text-secondary)" }}
              >
                {description}
              </RadixDialog.Description>
            )}
          </div>
          <RadixDialog.Close
            className="p-1 rounded-md hover:bg-[var(--surface-elevated)]"
            style={{ color: "var(--text-tertiary)" }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </RadixDialog.Close>
        </div>
        {children}
      </RadixDialog.Content>
    </RadixDialog.Portal>
  );
}
