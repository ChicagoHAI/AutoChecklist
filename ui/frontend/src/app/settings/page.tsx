"use client";

import { useState, useEffect } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Badge } from "@/components/ui/Badge";
import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { useSettings, useUpdateSettings, useTestConnection, useHealth } from "@/lib/hooks";
import { clearAllData, seedDefaultPrompts } from "@/lib/api";

const PROVIDER_OPTIONS = [
  { value: "openrouter", label: "OpenRouter" },
  { value: "openai", label: "OpenAI" },
  { value: "vllm", label: "vLLM" },
];

export default function SettingsPage() {
  const { data: settings, isLoading: settingsLoading, error: settingsError } = useSettings();
  const updateSettings = useUpdateSettings();
  const testConn = useTestConnection();
  const testVllm = useTestConnection();
  const { data: health } = useHealth();

  // Local form state
  const [openrouterKey, setOpenrouterKey] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [defaultProvider, setDefaultProvider] = useState("openrouter");
  const [defaultModel, setDefaultModel] = useState("openai/gpt-5-mini");
  const [vllmBaseUrl, setVllmBaseUrl] = useState("http://localhost:8000/v1");
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  // Seed prompts state
  const [seeding, setSeeding] = useState(false);
  const [seedMessage, setSeedMessage] = useState<string | null>(null);

  // Clear data state
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearMessage, setClearMessage] = useState<string | null>(null);

  // Sync settings to local state when loaded
  useEffect(() => {
    if (settings) {
      setOpenrouterKey(settings.openrouter_api_key || "");
      setOpenaiKey(settings.openai_api_key || "");
      setDefaultProvider(settings.default_provider || "openrouter");
      setDefaultModel(settings.default_model || "openai/gpt-5-mini");
      setVllmBaseUrl(settings.vllm_base_url || "http://localhost:8000/v1");
    }
  }, [settings]);

  const handleSave = async () => {
    setSaveMessage(null);
    try {
      // Only send API keys if user entered new ones (not masked values from settings)
      const updates: Record<string, string> = {
        default_provider: defaultProvider,
        default_model: defaultModel,
        vllm_base_url: vllmBaseUrl,
      };
      if (openrouterKey && !openrouterKey.includes("...")) {
        updates.openrouter_api_key = openrouterKey;
      }
      if (openaiKey && !openaiKey.includes("...")) {
        updates.openai_api_key = openaiKey;
      }
      await updateSettings.mutateAsync(updates);
      setSaveMessage("Settings saved successfully.");
      setTimeout(() => setSaveMessage(null), 3000);
    } catch {
      setSaveMessage("Failed to save settings. Is the backend running?");
    }
  };

  const handleTestConnection = () => {
    // Only send API key if user entered a new one (not masked from settings)
    const rawKey = defaultProvider === "openrouter" ? openrouterKey : defaultProvider === "openai" ? openaiKey : undefined;
    const apiKey = rawKey && !rawKey.includes("...") ? rawKey : undefined;
    const baseUrl = defaultProvider === "vllm" ? vllmBaseUrl : undefined;
    testConn.mutate({ provider: defaultProvider, model: defaultModel, apiKey, baseUrl });
  };

  const handleTestVllm = () => {
    testVllm.mutate({ provider: "vllm", model: "default", baseUrl: vllmBaseUrl });
  };

  const handleSeedPrompts = async () => {
    setSeeding(true);
    setSeedMessage(null);
    try {
      const result = await seedDefaultPrompts();
      setSeedMessage(`Seeded ${result.total} prompt templates (${result.seeded.join(", ")}).`);
      setTimeout(() => setSeedMessage(null), 5000);
    } catch {
      setSeedMessage("Failed to seed prompts. Is the backend running?");
    } finally {
      setSeeding(false);
    }
  };

  const handleClearAllData = async () => {
    setClearing(true);
    setClearMessage(null);
    try {
      const result = await clearAllData();
      const totalDeleted = result.details.reduce((sum, d) => sum + d.files_deleted, 0);
      setClearMessage(`Cleared ${totalDeleted} files across ${result.details.length} directories.`);
      setClearDialogOpen(false);
      setTimeout(() => setClearMessage(null), 5000);
    } catch {
      setClearMessage("Failed to clear data. Is the backend running?");
    } finally {
      setClearing(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Settings"
        description="Configure API keys, default model, and connection settings."
      >
        <div className="flex items-center gap-2">
          {health ? (
            <Badge variant="success">Backend Connected</Badge>
          ) : (
            <Badge variant="error">Backend Offline</Badge>
          )}
        </div>
      </PageHeader>

      <div className="space-y-6 max-w-2xl">
        {/* API Keys */}
        <Card>
          <CardHeader>
            <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              API Keys
            </h2>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
              Keys are stored in the backend session and never sent to the frontend.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              label="OpenRouter API Key"
              type="password"
              value={openrouterKey}
              onChange={(e) => setOpenrouterKey(e.target.value)}
              placeholder="sk-or-..."
            />
            <Input
              label="OpenAI API Key"
              type="password"
              value={openaiKey}
              onChange={(e) => setOpenaiKey(e.target.value)}
              placeholder="sk-..."
            />
          </CardContent>
        </Card>

        {/* Default Model */}
        <Card>
          <CardHeader>
            <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              Default Model
            </h2>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
              Used as the default for evaluations unless overridden per-request.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select
              label="Provider"
              options={PROVIDER_OPTIONS}
              value={defaultProvider}
              onChange={(e) => setDefaultProvider(e.target.value)}
            />
            <Input
              label="Model Name"
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              placeholder="openai/gpt-5-mini"
            />
          </CardContent>
        </Card>

        {/* vLLM Server */}
        <Card>
          <CardHeader>
            <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              vLLM Server
            </h2>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
              Configure a vLLM server for local or remote inference. Used when vLLM is selected as the provider in any evaluation.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              label="Base URL"
              value={vllmBaseUrl}
              onChange={(e) => setVllmBaseUrl(e.target.value)}
              placeholder="http://localhost:8000/v1"
            />
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={handleTestVllm}
                loading={testVllm.isPending}
              >
                Test Connection
              </Button>
              {testVllm.data && (
                <span
                  className="text-xs font-medium"
                  style={{
                    color: testVllm.data.success ? "var(--success)" : "var(--error)",
                  }}
                >
                  {testVllm.data.success
                    ? `Connected${testVllm.data.latency_ms ? ` (${testVllm.data.latency_ms}ms)` : ""}`
                    : testVllm.data.message || "Connection failed"}
                </span>
              )}
              {testVllm.error && (
                <span className="text-xs font-medium" style={{ color: "var(--error)" }}>
                  Could not reach backend
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Test Connection */}
        <Card>
          <CardHeader>
            <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              Connection Test
            </h2>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              Test your current configuration by sending a simple request to the selected provider.
            </p>
            <Button
              variant="outline"
              onClick={handleTestConnection}
              loading={testConn.isPending}
            >
              Test Connection
            </Button>
            {testConn.data && (
              <div
                className="p-3 rounded-md border text-sm"
                style={{
                  backgroundColor: testConn.data.success ? "var(--success-light)" : "var(--error-light)",
                  borderColor: testConn.data.success ? "var(--success)" : "var(--error)",
                  color: "var(--text-primary)",
                }}
              >
                <div className="flex items-center gap-2 mb-1">
                  {testConn.data.success ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2">
                      <path d="M22 11.08V12a10 10 0 11-5.93-9.14" strokeLinecap="round" strokeLinejoin="round" />
                      <polyline points="22 4 12 14.01 9 11.01" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="15" y1="9" x2="9" y2="15" strokeLinecap="round" />
                      <line x1="9" y1="9" x2="15" y2="15" strokeLinecap="round" />
                    </svg>
                  )}
                  <span className="font-medium">
                    {testConn.data.success ? "Connection successful" : "Connection failed"}
                  </span>
                </div>
                <p style={{ color: "var(--text-secondary)" }}>{testConn.data.message}</p>
                {testConn.data.latency_ms && (
                  <p
                    className="mt-1 text-xs"
                    style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}
                  >
                    Latency: {testConn.data.latency_ms}ms
                  </p>
                )}
              </div>
            )}
            {testConn.error && (
              <div
                className="p-3 rounded-md border text-sm"
                style={{
                  backgroundColor: "var(--error-light)",
                  borderColor: "var(--error)",
                  color: "var(--text-primary)",
                }}
              >
                Failed to reach the backend. Make sure the server is running.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Save */}
        <div className="flex items-center gap-4">
          <Button onClick={handleSave} loading={updateSettings.isPending}>
            Save Settings
          </Button>
          {saveMessage && (
            <span
              className="text-sm"
              style={{
                color: saveMessage.includes("success")
                  ? "var(--success)"
                  : "var(--error)",
              }}
            >
              {saveMessage}
            </span>
          )}
        </div>

        {/* Loading/error states */}
        {settingsLoading && (
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            Loading settings...
          </p>
        )}
        {settingsError && (
          <div
            className="p-3 rounded-md border text-sm"
            style={{
              backgroundColor: "var(--warning-light)",
              borderColor: "var(--warning)",
              color: "var(--text-primary)",
            }}
          >
            Could not load settings from the backend. You can still configure settings
            locally and save when the backend is available.
          </div>
        )}

        {/* Prompt Library */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                  Reset Prompt Library
                </p>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
                  Replace all prompt templates with the latest built-in defaults from the package.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleSeedPrompts}
                loading={seeding}
              >
                Seed Defaults
              </Button>
            </div>
            {seedMessage && (
              <p
                className="mt-3 text-xs"
                style={{
                  color: seedMessage.includes("Failed") ? "var(--error)" : "var(--success)",
                }}
              >
                {seedMessage}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Danger Zone */}
        <div className="pt-6 mt-6 border-t" style={{ borderColor: "var(--border)" }}>
          <h2
            className="text-base font-semibold mb-1"
            style={{ color: "var(--error)" }}
          >
            Danger Zone
          </h2>
          <p className="text-sm mb-4" style={{ color: "var(--text-secondary)" }}>
            Irreversible actions that delete stored data.
          </p>
          <Card
            className="border"
            style={{ borderColor: "var(--error)" } as React.CSSProperties}
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    Clear All Data
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
                    Delete all saved evaluations, checklists, and batch results.
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setClearDialogOpen(true)}
                  className="border-[var(--error)] text-[var(--error)] hover:bg-[var(--error-light)]"
                >
                  Clear All Data
                </Button>
              </div>
              {clearMessage && (
                <p
                  className="mt-3 text-xs"
                  style={{
                    color: clearMessage.includes("Failed") ? "var(--error)" : "var(--success)",
                  }}
                >
                  {clearMessage}
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Clear data confirmation dialog */}
      <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
        <DialogContent title="Clear All Data" description="This action cannot be undone.">
          <div className="space-y-4">
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              This will permanently delete all saved evaluations, checklists, and batch results.
              Your settings and API keys will not be affected.
            </p>
            <div className="flex items-center justify-end gap-3 pt-2">
              <Button variant="ghost" size="sm" onClick={() => setClearDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleClearAllData}
                loading={clearing}
                className="bg-[var(--error)] hover:bg-red-700"
              >
                Delete Everything
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
