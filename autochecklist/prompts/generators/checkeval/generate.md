# Question Generation

Generate binary yes/no evaluation questions for the given dimension.

## Task Type
{task_type}

## Dimension
**Name:** {dimension_name}
**Definition:** {definition}

## Sub-dimensions
{sub_dimensions}

## Instructions

Generate evaluation questions that can be answered with "Yes" or "No" for each sub-dimension.

For each sub-dimension, create {questions_per_sub} questions that:
1. Are directly tied to the dimension definition
2. Can be answered objectively with Yes or No
3. Start with words like "Does", "Is", "Are", "Can", "Has"
4. Focus on a single, specific criterion
5. Are relevant to evaluating {task_type} quality

## Output Format

Return your response as a JSON object with this structure:
```json
{{
    "questions": [
        {{
            "question": "Does the response maintain a clear logical flow?",
            "sub_aspect": "Logical Flow",
            "dimension": "{dimension_name}"
        }}
    ]
}}
```

Generate questions now:
