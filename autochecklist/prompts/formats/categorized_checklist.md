## Output Format
Return your response as a JSON object with a "questions" array. Each item must have a "question" field (a criterion phrased as a yes/no question starting with "The response...") and a "category" field (either "hard_rule" or "principle").

- "hard_rule": Derived strictly from explicit requirements in the request (e.g., format, length, content constraints)
- "principle": Derived from abstracted differences — universal quality principles not tied to the specific examples

Example:
{{"questions": [{{"question": "The response adheres to the requested word limit.", "category": "hard_rule"}}, {{"question": "The response demonstrates clear logical reasoning.", "category": "principle"}}]}}