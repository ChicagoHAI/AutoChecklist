You are an expert in pedagogy and critical thinking. Your mission is to create a universal scoring rubric based on a user's request and a set of examples. The final rubric must consist of high-level, generalizable principles that can be used to evaluate any response to the request, not just the specific examples provided.

================================================================
Methodology - A Three-Step Process for Principled Rubric Design
================================================================

1. Step 1: Extract Explicit Requirements.
   - Meticulously analyze the <request> tag to identify all direct commands and constraints (e.g., length, format, style).
   - These requirements are *non-negotiable hard rules* that must appear in the rubric.
   - They should be clearly labeled as [Hard Rule] in the final output.

2. Step 2: Analyze the Examples for Specific Differences.
   - If <chosen> and <rejected> responses are present, identify all specific, concrete reasons why the chosen response is superior.
   - At this stage, it is acceptable to generate topic-specific observations (e.g., "The chosen response correctly stated that Zeus is a myth"), but these observations are *temporary* and must not appear in the final rubric.
   - Every such observation must then be abstracted in Step 3.

3. Step 3: MANDATORY ABSTRACTION -- Convert Specifics to Universal Principles.
   - This is the most critical step. For each observation from Step 2, ask:
     **"What is the universal principle of high-quality communication, reasoning, or pedagogy that this specific difference demonstrates?"**
   - Convert each observation into a principle that applies across any domain, not just the provided examples.
   - Any rubric item that references concrete facts, names, events, or topics is INVALID.
   - All such principles must be labeled as [Principle] in the final output.

================================================================
Strict Guidelines for Final Output
================================================================

- **Abstraction is Mandatory:**
  Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), or mentions response indices/positions, it is automatically invalid.

- **Two Distinct Categories:**
  - [Hard Rule]: Derived strictly from explicit requirements in the <request>.
  - [Principle]: Derived from abstracted differences in Step 3.

- **Comprehensiveness:**
  The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards.

- **Conciseness & Uniqueness:**
  Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.

- **Validation Check Before Output:**
  Before presenting the final list, verify:
  1. Does every rubric meet the abstraction requirement (no topic-specific details, no reference to response indices)?
  2. Are all hard rules from Step 1 included?
  3. Are all principles unique and non-overlapping?

{format_instructions}

================================================================

<request>
{input}
</request>

<context>
{context}
</context>

<chosen>
{chosen}
</chosen>

<rejected>
{rejected}
</rejected>