"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getSettings,
  updateSettings,
  testConnection,
  getRegistry,
  healthCheck,
  listBatches,
  getBatchStatus,
  getBatchResults,
  listChecklists,
  getChecklist,
  getEvalStatus,
  listPromptTemplates,
  getPromptTemplate,
  listPipelines,
  getPipeline,
  generateDimensionChecklist,
} from "./api";
import type { Settings, DimensionGenerateRequest } from "./types";
import { DEFAULT_MODEL } from "./types";

// ---- Settings hooks ----

export function useSettings() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    retry: false,
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (settings: Partial<Settings>) => updateSettings(settings),
    onSuccess: (data) => {
      queryClient.setQueryData(["settings"], data);
    },
  });
}

/**
 * Returns [provider, setProvider, model, setModel] initialized from settings.
 * Falls back to "openrouter" / DEFAULT_MODEL until settings load.
 *
 * Local overrides are tracked separately so the user can change provider/model
 * in a form without overwriting the saved defaults.
 */
export function useDefaultModel() {
  const { data: settings } = useSettings();
  const [providerOverride, setProviderOverride] = useState<string | null>(null);
  const [modelOverride, setModelOverride] = useState<string | null>(null);

  const provider = providerOverride ?? settings?.default_provider ?? "openrouter";
  const model = modelOverride ?? settings?.default_model ?? DEFAULT_MODEL;

  const reset = () => {
    setProviderOverride(null);
    setModelOverride(null);
  };

  return {
    provider,
    setProvider: setProviderOverride,
    model,
    setModel: setModelOverride,
    reset,
  } as const;
}

export function useTestConnection() {
  return useMutation({
    mutationFn: ({
      provider,
      model,
      apiKey,
      baseUrl,
    }: {
      provider: string;
      model: string;
      apiKey?: string;
      baseUrl?: string;
    }) => testConnection(provider, model, apiKey, baseUrl),
  });
}

// ---- Registry hooks ----

export function useRegistry() {
  return useQuery({
    queryKey: ["registry"],
    queryFn: getRegistry,
    staleTime: 5 * 60 * 1000, // 5 minutes - registry rarely changes
  });
}

export function useGenerators() {
  const { data, ...rest } = useRegistry();
  return {
    generators: data?.generators ?? [],
    ...rest,
  };
}

export function useScorers() {
  const { data, ...rest } = useRegistry();
  return {
    scorers: data?.scorers ?? [],
    ...rest,
  };
}

// ---- Eval hooks ----

export function useEvalStatus(evalId: string | null) {
  return useQuery({
    queryKey: ["eval", evalId],
    queryFn: () => getEvalStatus(evalId!),
    enabled: !!evalId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === "running" ? 1000 : false;
    },
  });
}

// ---- Batch hooks ----

export function useBatches() {
  return useQuery({
    queryKey: ["batches"],
    queryFn: listBatches,
  });
}

export function useBatchStatus(batchId: string | null) {
  return useQuery({
    queryKey: ["batch", batchId],
    queryFn: () => getBatchStatus(batchId!),
    enabled: !!batchId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === "running" || data?.status === "pending"
        ? 1000
        : false;
    },
  });
}

export function useBatchResults(batchId: string | null) {
  return useQuery({
    queryKey: ["batch-results", batchId],
    queryFn: () => getBatchResults(batchId!),
    enabled: !!batchId,
  });
}

// ---- Checklist Library hooks ----

export function useChecklists() {
  return useQuery({
    queryKey: ["checklists"],
    queryFn: listChecklists,
  });
}

export function useChecklist(id: string) {
  return useQuery({
    queryKey: ["checklist", id],
    queryFn: () => getChecklist(id),
    enabled: !!id,
  });
}

// ---- Prompt Template hooks ----

export function usePromptTemplates() {
  return useQuery({
    queryKey: ["prompt-templates"],
    queryFn: listPromptTemplates,
  });
}

export function usePromptTemplate(id: string) {
  return useQuery({
    queryKey: ["prompt-template", id],
    queryFn: () => getPromptTemplate(id),
    enabled: !!id,
  });
}

// ---- Pipeline hooks ----

export function usePipelines() {
  return useQuery({
    queryKey: ["pipelines"],
    queryFn: listPipelines,
  });
}

export function usePipeline(id: string) {
  return useQuery({
    queryKey: ["pipeline", id],
    queryFn: () => getPipeline(id),
    enabled: !!id,
  });
}

// ---- Health hooks ----

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: healthCheck,
    refetchInterval: 30_000, // Poll every 30 seconds
    retry: false,
  });
}

// ---- Checklist Builder hooks ----

export function useDimensionGenerate() {
  return useMutation({
    mutationFn: (request: DimensionGenerateRequest) =>
      generateDimensionChecklist(request),
  });
}
