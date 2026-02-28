"use client";

import { useParams, useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/Button";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { BatchProgress } from "@/components/batch/BatchProgress";
import { BatchResults } from "@/components/batch/BatchResults";
import { useBatchStatus, useBatchResults } from "@/lib/hooks";
import { cancelBatch } from "@/lib/api";

export default function BatchDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const batchId = params.id as string;

  const { data: batch, isLoading: batchLoading } = useBatchStatus(batchId);
  const { data: resultsData, isLoading: resultsLoading } = useBatchResults(
    batch?.status === "completed" ||
      batch?.status === "failed" ||
      batch?.status === "cancelled"
      ? batchId
      : null
  );

  const cancelMutation = useMutation({
    mutationFn: (id: string) => cancelBatch(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["batch", batchId] });
    },
  });

  const handleCancel = () => {
    cancelMutation.mutate(batchId);
  };

  const handleBack = () => {
    router.push("/batch");
  };

  if (batchLoading) {
    return (
      <div>
        <PageHeader title="Batch Evaluation">
          <Button variant="outline" size="sm" onClick={handleBack}>
            Back
          </Button>
        </PageHeader>
        <div className="space-y-4">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  if (!batch) {
    return (
      <div>
        <PageHeader title="Batch Evaluation">
          <Button variant="outline" size="sm" onClick={handleBack}>
            Back
          </Button>
        </PageHeader>
        <div
          className="text-center py-16 text-sm"
          style={{ color: "var(--text-tertiary)" }}
        >
          Batch not found.
        </div>
      </div>
    );
  }

  const isActive = batch.status === "running" || batch.status === "pending";
  const showResults =
    batch.status === "completed" ||
    batch.status === "failed" ||
    batch.status === "cancelled";

  return (
    <div>
      <PageHeader
        title="Batch Evaluation"
        description={batch.config?.filename ?? `Batch ${batchId.slice(0, 8)}`}
      >
        <Button variant="outline" size="sm" onClick={handleBack}>
          Back
        </Button>
      </PageHeader>

      {isActive && (
        <BatchProgress
          batch={batch}
          onCancel={handleCancel}
          onViewResults={() => {}}
          cancelling={cancelMutation.isPending}
        />
      )}

      {showResults && !resultsLoading && (
        <BatchResults
          batch={batch}
          results={resultsData?.results ?? []}
          onBack={handleBack}
        />
      )}

      {showResults && resultsLoading && (
        <div className="space-y-4">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}
    </div>
  );
}
