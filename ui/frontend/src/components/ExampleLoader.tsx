"use client";

import { Example } from "@/lib/types";
import { Select } from "./ui/Select";

interface ExampleLoaderProps {
  examples: Example[];
  onSelect: (example: Example | null) => void;
  value?: string;
}

export function ExampleLoader({ examples, onSelect, value }: ExampleLoaderProps) {
  const options = [
    { value: "", label: "..." },
    ...examples.map((ex) => ({ value: ex.name, label: ex.name })),
  ];

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedName = e.target.value;
    if (!selectedName) {
      onSelect(null);
      return;
    }
    const example = examples.find((ex) => ex.name === selectedName);
    if (example) {
      onSelect(example);
    }
  };

  return (
    <Select
      options={options}
      value={value}
      onChange={handleChange}
      className="w-36"
    />
  );
}
