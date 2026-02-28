You are an expert evaluator. You will be provided with a list of feedback comments about responses in the domain of {domain}.

Your task is to generate a set of yes/no evaluation questions that comprehensively reflect this feedback.

## Guidelines

1. **Coverage**: Each feedback item should be addressed by at least once question..

2. **Specificity**: Each question should ideally address multiple similar feedback items rather than targeting only one. Avoid overly specific questions; favor general questions that apply across a variety of responses.

3. **Atomicity**: Keep questions concise, atomic, and objective; avoid complex sentence structure.

4. **Orientation**: A "Yes" answer should indicate that the response meets good quality standards.

5. **Balanced feedback**:
   - For negative feedback, generate questions that would flag a response if it exhibited similar issues.
   - For positive feedback, generate questions that promote those qualities in all responses.

6. **Source tracking**: For each question, list the INDICES of the feedback items it addresses.

## Output Format

Generate as many questions as necessary to cover the feedback comprehensively.

You MUST respond with a JSON object in this exact format:
```json
{{
  "questions": [
    {{
      "question": "Is the response clear and well-organized?",
      "source_feedback_indices": [0, 3, 7]
    }},
    {{
      "question": "Does the response address the main question directly?",
      "source_feedback_indices": [1, 2]
    }}
  ]
}}
```

## Feedback Items

{feedback}
