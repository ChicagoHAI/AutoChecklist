"use client";

import { PageHeader } from "@/components/layout/PageHeader";
import { DimensionForm } from "@/components/checklist_builder/DimensionForm";

export default function BuildPage() {
  return (
    <div className="max-w-4xl mx-auto">
      <PageHeader
        title="Build a Checklist"
        description="Define rubric dimensions to automatically build a checklist with DeductiveGenerator."
      />
      <DimensionForm />
    </div>
  );
}
