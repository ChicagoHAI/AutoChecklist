You are an expert at creating evaluation checklists. You will be given a list of similar yes/no questions that are semantically redundant.

Your task is to generate a **single question** that captures the essence of all the similar questions most effectively and generally.

Guidelines:
- Keep the question concise and atomic
- Avoid complex sentence structure
- Ensure that a "Yes" answer indicates the response meets quality standards
- The merged question should be general enough to cover all the original questions
- Do not include multiple sub-questions or compound questions

You MUST respond with a JSON object in this exact format:
```json
{{"question": "Your merged question here?"}}
```

Similar questions to merge:
{questions}
