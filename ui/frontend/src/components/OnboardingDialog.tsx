"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { Button } from "@/components/ui/Button";

const DISMISSED_KEY = "checklisteval-onboarding-dismissed";

export function OnboardingDialog() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    // Check localStorage first
    if (typeof window === "undefined") return;
    const dismissed = localStorage.getItem(DISMISSED_KEY);
    if (dismissed === "true") return;

    // Check if settings have an API key configured
    const checkSettings = async () => {
      try {
        const { protocol, hostname } = window.location;
        const apiBase = process.env.NEXT_PUBLIC_API_URL || `${protocol}//${hostname}:7771`;
        const res = await fetch(`${apiBase}/api/settings`);
        if (!res.ok) {
          setOpen(true);
          return;
        }
        const settings = await res.json();
        const hasKey =
          (settings.openrouter_api_key && settings.openrouter_api_key !== "" && !settings.openrouter_api_key.startsWith("****")) ||
          (settings.openai_api_key && settings.openai_api_key !== "" && !settings.openai_api_key.startsWith("****"));
        if (!hasKey) {
          setOpen(true);
        }
      } catch {
        // Backend not available, show onboarding
        setOpen(true);
      }
    };

    checkSettings();
  }, []);

  const handleDismiss = () => {
    setOpen(false);
    if (typeof window !== "undefined") {
      localStorage.setItem(DISMISSED_KEY, "true");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(val) => { if (!val) handleDismiss(); }}>
      <DialogContent
        title="Welcome to AutoChecklist"
        description="Automatically evaluate outputs using LLM-based checklist methods."
      >
        <div className="space-y-4">
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            To get started, you need to configure an API key for your LLM provider
            (OpenRouter, OpenAI, or vLLM).
          </p>
          <div
            className="p-3 rounded-md border text-sm"
            style={{
              backgroundColor: "var(--accent-light)",
              borderColor: "var(--accent-primary)",
              color: "var(--text-primary)",
            }}
          >
            Head to <strong>Settings</strong> to add your API key and select a default model.
          </div>
          <div className="flex items-center justify-end gap-3 pt-2">
            <Button variant="ghost" size="sm" onClick={handleDismiss}>
              Dismiss
            </Button>
            <Link href="/settings">
              <Button size="sm" onClick={handleDismiss}>
                Go to Settings
              </Button>
            </Link>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
