"use client";

import * as RadixTabs from "@radix-ui/react-tabs";

interface TabItem {
  value: string;
  label: string;
  disabled?: boolean;
  rightAligned?: boolean;
}

interface TabsProps {
  items: TabItem[];
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}

export function Tabs({ items, value, onValueChange, children }: TabsProps) {
  const hasRightTab = items.some((item) => item.rightAligned);

  return (
    <RadixTabs.Root value={value} onValueChange={onValueChange}>
      <RadixTabs.List
        className="flex gap-0 -mb-px relative z-10"
      >
        {items.map((item) => (
          <RadixTabs.Trigger
            key={item.value}
            value={item.value}
            disabled={item.disabled}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg border border-[var(--border)] border-b-[var(--border-strong)] text-[var(--text-secondary)] bg-[var(--surface-elevated)] disabled:opacity-40 disabled:cursor-not-allowed data-[state=active]:bg-white data-[state=active]:text-[var(--text-primary)] data-[state=active]:border-[var(--border-strong)] data-[state=active]:border-b-white hover:text-[var(--text-primary)] ${item.rightAligned ? "ml-auto" : ""}`}
          >
            {item.label}
          </RadixTabs.Trigger>
        ))}
      </RadixTabs.List>
      <div className={`border border-[var(--border-strong)] rounded-b-lg bg-white p-8 ${hasRightTab ? "" : "rounded-tr-lg"}`}>
        {children}
      </div>
    </RadixTabs.Root>
  );
}

export function TabContent({
  value,
  children,
}: {
  value: string;
  children: React.ReactNode;
}) {
  return (
    <RadixTabs.Content value={value} forceMount className="data-[state=inactive]:hidden">
      {children}
    </RadixTabs.Content>
  );
}
