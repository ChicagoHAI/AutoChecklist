# Stage 3: Key Question Generation

Generate one binary (Yes/No) evaluation question for each component.

## Dimension
**Name:** {dimension}
**Definition:** {rubric}

## Components and Attributes
{components_attributes}

## Instructions

For each component, generate ONE key evaluation question.

Requirements:
1. Question must be answerable with Yes or No
2. Question must incorporate the component's core concept
3. Question must minimize subjectivity
4. "Yes" should indicate a positive evaluation
5. Question should start with "Does", "Is", "Are", "Can", etc.

## Output Format

Return your response as a JSON object mapping components to questions:
```json
{{
    "Component 1": "Is the text well-organized?",
    "Component 2": "Does the text maintain logical flow?"
}}
```

Generate key questions now:
