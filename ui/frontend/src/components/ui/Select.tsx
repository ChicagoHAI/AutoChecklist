"use client";

import { SelectHTMLAttributes, forwardRef } from "react";

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: Array<{ value: string; label: string }>;
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = "", label, options, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, "-");

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium mb-1.5"
            style={{ color: "var(--text-secondary)" }}
          >
            {label}
          </label>
        )}
        <div className="relative">
          <select
            ref={ref}
            id={inputId}
            className={`
              w-full px-3 py-2 rounded-md border text-sm appearance-none
              text-[var(--text-primary)]
              disabled:opacity-50 disabled:cursor-not-allowed
              focus:outline-none focus:ring-2 focus:ring-offset-1
              border-[var(--border-strong)] focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)]
              bg-white
              pr-8
              ${className}
            `}
            style={{ fontFamily: "var(--font-sans)" }}
            {...props}
          >
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <svg
            className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--text-tertiary)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </div>
    );
  }
);

Select.displayName = "Select";

export { Select };
