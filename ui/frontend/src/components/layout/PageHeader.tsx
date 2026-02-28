interface PageHeaderProps {
  title?: string;
  description?: string;
  children?: React.ReactNode;
}

export function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6 -mt-2">
      <div>
        {title && (
          <h1
            className="text-2xl font-semibold"
            style={{ color: "var(--text-primary)", fontFamily: "var(--font-serif)" }}
          >
            {title}
          </h1>
        )}
        {description && (
          <p
            className={`text-base max-w-2xl${title ? " mt-1" : ""}`}
            style={{ color: "var(--text-secondary)", lineHeight: 1.6 }}
          >
            {description}
          </p>
        )}
      </div>
      {children && <div className="flex items-center gap-2 flex-shrink-0">{children}</div>}
    </div>
  );
}
