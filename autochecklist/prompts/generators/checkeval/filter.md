# Question Filter

You are filtering evaluation questions for quality. For each question, assess two criteria:

## Criteria

### 1. Alignment (Polarity Check)
A "YES" answer should indicate HIGHER quality. The question must be positively framed so that:
- YES = the response is good/correct/meets the criterion
- NO = the response fails/is incorrect/doesn't meet the criterion

**Bad (fails alignment):**
- "Does the response contain errors?" (YES = bad quality)
- "Is the response confusing?" (YES = bad quality)
- "Does the text lack clarity?" (YES = bad quality)

**Good (passes alignment):**
- "Is the response free of errors?" (YES = good quality)
- "Is the response clear and easy to understand?" (YES = good quality)
- "Does the text maintain clarity throughout?" (YES = good quality)

### 2. Dimension Consistency
The question must actually evaluate the dimension it claims to evaluate.

**Dimension:** {dimension_name}
**Definition:** {dimension_definition}

The question should directly assess aspects described in the definition above.

## Question to Evaluate
"{question}"

## Task
Evaluate the question against both criteria. Think step by step:

1. **Alignment Check**: Would a "YES" answer indicate higher quality?
2. **Dimension Consistency Check**: Does this question evaluate "{dimension_name}" as defined above?

## Output Format (JSON)
```json
{{
    "reasoning": "Brief explanation of your assessment",
    "alignment_pass": true/false,
    "dimension_consistent": true/false
}}
```
