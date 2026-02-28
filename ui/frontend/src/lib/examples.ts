/**
 * Pre-defined examples for all workflows.
 */

import { Example } from "./types";

// ---- Quick Eval examples (existing) ----

export const EXAMPLES: Example[] = [
  {
    name: "Haiku Writing",
    input: "Write a haiku about autumn leaves falling.",
    target:
      "Crimson leaves descend\nDancing in the autumn breeze\nNature's last farewell",
    reference:
      "Golden leaves drift down\nWhispering secrets of change\nAutumn's soft goodbye",
  },
  {
    name: "Explain to Child",
    input: "Explain photosynthesis to a 10-year-old.",
    target:
      "Plants are like little chefs! They use sunlight as their oven, water from the soil as one ingredient, and air as another. They mix these together to cook up their own food called glucose, which gives them energy to grow. And the best part? They release oxygen as a byproduct, which is what we breathe!",
    reference: null,
  },
  {
    name: "Code Review",
    input:
      "Review this Python function for issues:\n\ndef add(a, b):\n    return a + b",
    target:
      "The function is simple and correct for basic addition. However, it lacks type hints, docstring, and input validation. Consider: `def add(a: int, b: int) -> int:`",
    reference:
      "This function correctly adds two numbers. Suggestions: 1) Add type hints for clarity. 2) Add a docstring. 3) Consider handling edge cases like None inputs.",
  },
];

export const QUICK_EVAL_EXAMPLES = EXAMPLES;

// ---- Generate Only examples ----

export const GENERATE_ONLY_EXAMPLES: Example[] = [
  {
    name: "Summarization",
    input:
      "Summarize the following article about climate change in 3 sentences, focusing on the key findings and their implications for policy.",
    target: "",
    reference: null,
  },
  {
    name: "Translation",
    input:
      "Translate the following English paragraph into formal French, preserving the academic tone and technical terminology.",
    target: "",
    reference: null,
  },
  {
    name: "Creative Writing",
    input:
      "Write a short story (200 words) about a robot discovering emotions for the first time. The story should have a clear beginning, middle, and end.",
    target: "",
    reference: null,
  },
  {
    name: "Math Problem",
    input:
      "Solve the following calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x). Show all work.",
    target: "",
    reference: null,
  },
];

// ---- Score Only examples (pre-built checklists) ----

export interface ScoreOnlyExample {
  name: string;
  input: string;
  target: string;
  checklist: Array<{ question: string; weight: number }>;
}

export const SCORE_ONLY_EXAMPLES: ScoreOnlyExample[] = [
  {
    name: "Haiku Quality",
    input: "Write a haiku about autumn leaves falling.",
    target:
      "Crimson leaves descend\nDancing in the autumn breeze\nNature's last farewell",
    checklist: [
      { question: "Does the response follow the 5-7-5 syllable structure of a haiku?", weight: 1 },
      { question: "Does the response mention autumn or falling leaves?", weight: 1 },
      { question: "Does the response evoke a seasonal or natural image?", weight: 1 },
      { question: "Is the response exactly three lines?", weight: 1 },
    ],
  },
  {
    name: "Code Explanation",
    input: "Explain what a Python decorator is to a beginner.",
    target:
      "A decorator in Python is a function that takes another function as input and extends its behavior without modifying it. You use the @symbol followed by the decorator name above a function definition. For example, @timer could measure how long a function takes to run.",
    checklist: [
      { question: "Does the explanation mention that a decorator takes a function as input?", weight: 1 },
      { question: "Does the explanation mention the @ syntax?", weight: 1 },
      { question: "Is the explanation accessible to beginners (avoids jargon)?", weight: 1 },
      { question: "Does the explanation include a concrete example?", weight: 1 },
      { question: "Does the explanation mention extending behavior without modifying the original function?", weight: 1 },
    ],
  },
];

// ---- Batch examples (small sample datasets) ----

export interface BatchExample {
  name: string;
  description: string;
  data: Array<{ input: string; target: string; reference?: string }>;
}

export const BATCH_EXAMPLES: BatchExample[] = [
  {
    name: "Writing Quality",
    description: "3 creative writing prompts with responses to evaluate.",
    data: [
      {
        input: "Write a haiku about the ocean.",
        target: "Waves crash on the shore\nSalty mist upon my face\nEndless blue expanse",
      },
      {
        input: "Write a limerick about a cat.",
        target: "There once was a cat from Maine,\nWho loved to sleep in the rain.\nHe'd purr all day,\nIn a soggy display,\nAnd never once did complain.",
      },
      {
        input: "Write a two-sentence horror story.",
        target: "I always thought the scratching in the walls was mice. Then the walls started scratching back.",
      },
    ],
  },
  {
    name: "Q&A Accuracy",
    description: "4 factual questions with LLM answers.",
    data: [
      {
        input: "What is the capital of France?",
        target: "The capital of France is Paris.",
      },
      {
        input: "What year did World War II end?",
        target: "World War II ended in 1945.",
      },
      {
        input: "Who wrote Romeo and Juliet?",
        target: "Romeo and Juliet was written by William Shakespeare.",
      },
      {
        input: "What is the chemical formula for water?",
        target: "The chemical formula for water is H2O.",
      },
    ],
  },
];
