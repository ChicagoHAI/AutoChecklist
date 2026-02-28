# Stage 5: Question Validation

Validate and refine the generated questions to create the final checklist.

## Dimension
**Name:** {dimension}
**Definition:** {rubric}

## Questions to Validate
{all_questions}

## Validation Criteria

Examine each question against these criteria:
1. Is it answerable with Yes or No?
2. Does it contain concepts related to {dimension}?
3. Does it minimize subjectivity?
4. Is it semantically distinct from other questions (no redundancy)?
5. Does "Yes" indicate a positive evaluation?
6. Does it combine similar questions if needed?
7. Does it ask about the presence of positive qualities?

## Instructions

- Remove or revise questions that don't meet the criteria
- Combine semantically redundant questions
- Ensure questions cover the key aspects of {dimension}
- Aim for 10-15 questions in the final checklist

## Output Format

Return your response as a JSON list of validated questions:
```json
["Is the text well-organized?", "Does the text maintain logical flow?", "..."]
```

Validate and refine questions now:
