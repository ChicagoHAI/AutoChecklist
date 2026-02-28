# Question Augmentation

Augment the given evaluation questions using the {augmentation_mode} strategy.

## Task Type
{task_type}

## Dimension
**Name:** {dimension_name}
**Definition:** {definition}

## Existing Questions
{existing_questions}

## Augmentation Strategy: {augmentation_mode}

{augmentation_instructions}

## Instructions

Based on the existing questions, generate additional questions following the {augmentation_mode} strategy.

Each new question should:
1. Be answerable with "Yes" or "No"
2. Start with words like "Does", "Is", "Are", "Can", "Has"
3. Focus on a single, specific criterion
4. Be relevant to the dimension: {dimension_name}

## Output Format

Return your response as a JSON object with this structure:
```json
{{
    "questions": [
        {{
            "question": "Does the response...",
            "sub_aspect": "Sub-aspect Name",
            "dimension": "{dimension_name}"
        }}
    ]
}}
```

Include both the original questions AND the new augmented questions in your response.

Generate augmented questions now:
