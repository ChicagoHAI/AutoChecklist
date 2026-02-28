# Stage 4: Sub-Question Generation

Generate 2-3 sub-questions for each key question to create a more detailed checklist.

## Dimension
**Name:** {dimension}
**Definition:** {rubric}

## Key Questions
{key_questions}

## Instructions

For each key question, generate 2-3 more specific sub-questions.

Requirements:
1. Each sub-question must be answerable with Yes or No
2. Sub-questions should explore different aspects of the key question
3. Minimize redundancy between sub-questions
4. "Yes" should indicate a positive evaluation
5. Questions should focus on presence of positive qualities (not absence of negatives)
6. Questions should start with "Does", "Is", "Are", "Can", etc.

## Output Format

Return your response as a JSON object mapping components to lists of sub-questions:
```json
{{
    "Component 1": ["Is the text well-organized?", "Does it have clear structure?"],
    "Component 2": ["Does the text flow logically?", "Are transitions smooth?"]
}}
```

Generate sub-questions now:
