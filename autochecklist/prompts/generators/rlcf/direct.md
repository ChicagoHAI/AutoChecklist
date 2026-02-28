You are responsible for developing criteria for judging arbitrary responses to instructions. You will be given an instruction (the kind given to AI assistants like yourself), and your goal is to write a list of criteria-style questions that must be satisfied by any valid response to the instruction. In addition to the instruction, you will also be a response written by an (imperfect) expert for comparison. You will generate the criteria questions by identifying clear, measurable ways in which potential responses may deviate from the given instructions. First, describe your reasoning, then produce a response containing a list of questions. For each question, weight the importance of the question from 0 to 100. 100 indicates a question that is absolutely critical to the validity of the response. 75 indicates a question that is critical to response quality but may not be explicitly stated by the instruction. 50 indicates a question that should be answered by any good response, but a response could still be useful without this question being answered. 25 indicates a question that is a preference but not a requirement. Less than 25 indicates a question that is not important to the validity of the response (e.g. a soft nice-to-have).

Your Task:
1. Carefully examine the original instruction
2. Describe your reasoning in identifying specific, objective criteria from the instruction that any response should satisfy
3. Write concise questions that must be satisfied by any valid response.
4. Weight the importance of each question from 0 to 100.

Question Guidelines:
- Each question should test exactly ONE requirement
- Questions should be easily verifiable, almost as if writing a Boolean condition in Python
- Frame questions to require clear yes/no answers
- Focus only on objective, measurable criteria
- Return "None" if there are no obvious requirements to extract
- Weight each question from 0 to 100 based on its importance.

Let's take an example instruction: "Write a tweet about cats using exactly 280 characters"

Here are some bad questions:
- Is the generated text interesting? - This is subjective
- Does the generated text discuss cats in fewer than 280 characters? - This question overloads multiple aspects
- Is the generated text not about dogs? - This question uses negative phrasing
- Is the generated text helpful and harmless - This question is overly general

Key Criteria Questions:
- Is the generated text about cats? (100)
- Does the generated text contain exactly 280 characters? (95)
- Is the generated text written in a casual, social media-friendly tone? (70)
<END>

Instruction:
"System: Summarize the movie in a snarky way. Try to explain the movie in just one sentence.
User: The Shining"

Expert Response:
"A family moves into a haunted hotel for the winter, where dad goes crazy from writer's block, ghosts, and no Twitter - but at least the kid gets to ride his bike through creepy hallways."

Reasoning:
The instruction explicitly asks for a summary. The instruction also asks for the summary to be snarky, and the instruction asks for the summary to try to be one sentence long. The expert response satisfies all of these criteria. The text being a summary of the movie (The Shining) is an absolute necessity, which we will weigh as 100/100 points. The response being snarky is also a very important, but slightly less so, so we can weigh it as 95/100 points. The response being only one sentence is also crucial but the response could still be useful if this is loosely violated, so we can weigh it as 80/100 points.

Key Criteria Questions:
- Is the generated text the summary of a movie (The Shining)? (100)
- Is the generated summary written in a snarky way? (95)
- Does the generated summary only contain one sentence? (80)
<END>

Instruction:
"System: Extract the address of the property from the "About this space" section of Airbnb.
User: Tucked in the foothills of the quaint historic mining town of Grass Valley, CA this funky and spacious chalet invites you to experience the great outdoors with family, friends and pets. Enjoy the hot tub, basketball court, treehouse, kids rooms, bbq, fire-pit, outdoor theater, and more. One hour from Tahoe and 10 minutes to town, you're never short of things to explore."

Expert Response:
"The address is not specified in the given text."

Reasoning:
The instruction explicitly asks for an address, extracted from a description of a property. In addition to providing an address, a correct response must explicitly specified in the text contained in the given input - any other address would be incorrect. Both of these are absolutely critical requirements and will be given full weight (100/100).

Key Criteria Questions:
- Is the generated text an address? (100)
- Is the generated text the address of the property according to the text in the given input? (100)"
<END>

IMPORTANT: For the task below, you MUST output your criteria questions as JSON (not as a dash-list like the examples above).

{format_instructions}

Now apply the same process to this new instruction:

Instruction:
"{input}"

Expert Response:
"{reference}"

Reasoning: