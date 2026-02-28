# Stage 1: Component Extraction

Given a set of think-aloud attributes about evaluating {dimension}, extract the key recurring components/themes.

## Dimension
**Name:** {dimension}
**Definition:** {rubric}

## Attributes
{attributes}

## Instructions

Identify the most important recurring themes or components from these attributes.

Requirements:
- Extract at most {max_components} unique components
- Each component should represent a distinct evaluation criterion
- Components should be related to {dimension}
- Use concise, descriptive names for components

## Output Format

Return your response as a JSON list of component names:
```json
["Component 1", "Component 2", "Component 3"]
```

Extract components now:
