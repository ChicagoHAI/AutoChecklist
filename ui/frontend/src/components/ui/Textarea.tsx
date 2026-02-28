"use client";

import { TextareaHTMLAttributes, forwardRef } from "react";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
}

const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = "", label, error, id, ...props }, ref) => {
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
            {props.required && (
              <span className="ml-1" style={{ color: "var(--error)" }}>
                *
              </span>
            )}
          </label>
        )}
        <textarea
          ref={ref}
          id={inputId}
          className={`
            w-full px-3 py-2.5 rounded-md border text-sm
            text-[var(--text-primary)]
            placeholder:text-[var(--text-tertiary)]
            disabled:opacity-50 disabled:cursor-not-allowed
            resize-y min-h-[100px]
            focus:outline-none focus:ring-2 focus:ring-offset-1
            ${
              error
                ? "border-[var(--error)] focus:ring-[var(--error)] bg-[var(--error-light)]"
                : "border-[var(--border-strong)] focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)] bg-white"
            }
            ${className}
          `}
          style={{
            fontFamily: "var(--font-sans)",
            lineHeight: "1.6",
          }}
          {...props}
        />
        {error && (
          <p className="mt-1.5 text-xs" style={{ color: "var(--error)" }}>
            {error}
          </p>
        )}
      </div>
    );
  }
);

Textarea.displayName = "Textarea";

export { Textarea };
