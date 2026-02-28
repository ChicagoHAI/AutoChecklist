"use client";

import { useState } from "react";
import { Tabs, TabContent } from "@/components/ui/Tabs";
import { PlaygroundForm } from "@/components/playground/PlaygroundForm";
import { CompareForm } from "@/components/compare/CompareForm";
import { MethodReferenceTable } from "@/components/MethodReferenceTable";
import { PageHeader } from "@/components/layout/PageHeader";

const TAB_ITEMS = [
  { value: "evaluate", label: "Custom Eval" },
  { value: "compare", label: "Compare" },
  { value: "reference", label: "Reference", rightAligned: true },
];

export default function EvaluatePage() {
  const [activeTab, setActiveTab] = useState("evaluate");

  return (
    <div>
      <PageHeader description="Composable pipelines for generating and scoring checklist criteria for evaluation." />
      <Tabs items={TAB_ITEMS} value={activeTab} onValueChange={setActiveTab}>
        <TabContent value="evaluate">
          <PlaygroundForm />
        </TabContent>
        <TabContent value="compare">
          <CompareForm />
        </TabContent>
        <TabContent value="reference">
          <MethodReferenceTable />
        </TabContent>
      </Tabs>
    </div>
  );
}
