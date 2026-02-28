# Stage 2: Attributes Clustering

Group the given attributes under their corresponding components.

## Components
{components}

## Attributes
{attributes}

## Instructions

For each component, identify which attributes belong to it.

Requirements:
- Each attribute should be assigned to exactly one component
- Group attributes based on semantic similarity to the component
- All attributes should be assigned to a component

## Output Format

Return your response as a JSON object mapping components to attribute lists:
```json
{{
    "Component 1": ["attribute 1", "attribute 2"],
    "Component 2": ["attribute 3", "attribute 4"]
}}
```

Cluster attributes now:
