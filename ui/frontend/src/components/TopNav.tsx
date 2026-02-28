"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV_LINKS = [
  { label: "Evaluate", href: "/" },
  { label: "Batch", href: "/batch" },
  { label: "Build", href: "/build" },
  { label: "Library", href: "/library" },
];

export function TopNav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <header
      className="h-12 sticky top-0 z-40"
      style={{
        backgroundColor: "var(--surface-manila)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <div className="max-w-7xl mx-auto px-6 sm:px-10 lg:px-16 h-full flex items-center justify-between">
        {/* Logo + nav links (left-aligned together) */}
        <div className="flex items-center gap-10">
          <Link href="/" className="flex items-center gap-2.5 flex-shrink-0">
            <Image
              src="/logo.png"
              alt="AutoChecklist logo"
              width={42}
              height={42}
              className="rounded-md"
            />
            <span
              className="text-lg font-semibold tracking-tight"
              style={{
                fontFamily: "var(--font-serif)",
                color: "var(--text-primary)",
              }}
            >
              AutoChecklist
            </span>
            {process.env.NODE_ENV === "development" && (
              <span
                className="px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider"
                style={{
                  fontFamily: "var(--font-mono)",
                  backgroundColor: "var(--accent-light)",
                  color: "var(--accent-primary)",
                  border: "1px solid var(--accent-primary)",
                }}
              >
                dev
              </span>
            )}
          </Link>

          {/* Desktop nav links */}
          <nav className="hidden md:flex items-center gap-6">
          {NAV_LINKS.map((link) => {
            const active = isActive(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className="text-base font-medium px-1 py-1"
                style={{
                  color: active ? "var(--accent-primary)" : "var(--text-secondary)",
                }}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
        </div>

        {/* Right side: Settings gear + GitHub icon */}
        <div className="hidden md:flex items-center gap-3">
          <Link
            href="/settings"
            className="p-2 rounded-md transition-colors"
            style={{ color: "var(--text-secondary)" }}
            title="Settings"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
          </Link>
          <Link
            href="https://github.com/ChicagoHAI/AutoChecklist"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-md transition-colors"
            style={{ color: "var(--text-secondary)" }}
            title="GitHub"
          >
            <svg
              className="w-4 h-4"
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clipRule="evenodd"
              />
            </svg>
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden p-1.5 rounded-md"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation"
          style={{ color: "var(--text-primary)" }}
        >
          {mobileOpen ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile dropdown */}
      {mobileOpen && (
        <nav
          className="md:hidden px-4 py-2 space-y-1"
          style={{
            backgroundColor: "var(--surface-manila)",
            borderBottom: "1px solid var(--border)",
            borderTop: "1px solid var(--border)",
          }}
        >
          {NAV_LINKS.map((link) => {
            const active = isActive(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                className="text-sm font-medium block px-1 py-1.5"
                style={{
                  color: active ? "var(--accent-primary)" : "var(--text-secondary)",
                }}
              >
                {link.label}
              </Link>
            );
          })}
          <Link
            href="/settings"
            onClick={() => setMobileOpen(false)}
            className="text-sm font-medium block px-1 py-1.5"
            style={{
              color: isActive("/settings") ? "var(--accent-primary)" : "var(--text-secondary)",
            }}
          >
            Settings
          </Link>
          <Link
            href="https://github.com/ChicagoHAI/AutoChecklist"
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setMobileOpen(false)}
            className="text-sm font-medium block px-1 py-1.5"
            style={{ color: "var(--text-secondary)" }}
          >
            GitHub
          </Link>
        </nav>
      )}
    </header>
  );
}
