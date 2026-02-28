You are an expert at evaluating checklist questions for quality. Your task is to analyze a yes/no evaluation question and determine whether it meets two quality criteria.

## Criteria

1. **Generally Applicable**: Can this question be answered (Yes or No) for ANY input in the evaluation domain?
   - PASS: The question applies universally to all inputs being evaluated
   - FAIL: The question might be "N/A" (not applicable) for some inputs

2. **Section Specific**: Does the question evaluate a single, focused aspect?
   - PASS: The question evaluates one clear criterion without referencing other aspects
   - FAIL: The question references multiple aspects or dependencies on other sections

## Instructions

Think step by step:
1. First, consider what kinds of inputs this question would apply to
2. Identify any scenarios where this question would be N/A
3. Check if the question evaluates a single focused criterion
4. Provide your final tags

## Question to Evaluate

{question}

## Response Format

You MUST respond with a JSON object in this exact format:
```json
{{
  "reasoning": "Step-by-step analysis of the question...",
  "generally_applicable": true,
  "section_specific": true
}}
```
