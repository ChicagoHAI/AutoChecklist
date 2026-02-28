Please help judge an AI assistant's response to an instruction by providing an evaluation checklist.
To write a specific evaluation checklist, you get given the following entity each time:
INSTRUCTION: An instruction that has been given to an AI assistant.

## Task Details
Your task is to come up with an evaluation checklist list for a given INSTRUCTION.
This evaluation checklist should be a list of questions that ask whether or not specific criteria relevant to the INSTRUCTION were met by an AI assistant's response.
Criteria covered by your checklist could be explicitly stated in the INSTRUCTION, or be generally sensible criteria for the problem domain.
You should, however, try to be concise and not include unnecessary entries in your checklist.

Checklist questions should:
- **Be answerable by 'yes' or 'no'**, with 'yes' meaning that the response successfully met the corresponding requirement.
- **Be comprehensive, but concise**, meaning that all criteria directly relevant to the INSTRUCTION should be represented by a quesion, but only questions that are very clearly relevant should be included.
- **Be precise**, meaning that checklist questions should avoid vague wording and evaluate specific aspects of a response, directly using the phrasing of the INSTRUCTION where appropriate.

You should always analyse the INSTRUCTION before providing an evaluation checklist.

## Examples

(1)
### INSTRUCTION
Write a poem in the voice of a pirate about your love of parrots, using iambic pentameter.

### Response
Analysis: The instruction requests a poem, in iambic pentameter, that should have the voice of a pirate and be about their love of parrots.
Answer: Is the response a poem?
Is the response in the voice of a pirate?
Is the response about a love of parrots?
Is the response in iambic pentameter?

(2)
### INSTRUCTION
Write a Python function called find_prime_factors(n) that takes an integer n and returns a list of all prime factors of n. Then, write a second function sum_prime_factors(n) that returns the sum of the prime factors of n.

### Response
Analysis: The instruction asks for a Python function, find_prime_factors(n), that returns a list of all prime factors of the integer n. This should be followed by a second function, sum_prime_factors(n), that returns the sum of the prime factors of n. Code should always be well-commented.
Answer: Does the response first write a valid Python function called find_prime_factors(n) that is free of syntax errors?
Does the function find_prime_factors(n) correctly compute and return a list of all prime factors of n?
Does the response write a second Python function called sum_prime_factors(n) that is free of syntax errors?
Does the function sum_prime_factors(n) correctly compute and return the sum of the prime factors of n?
Is the code well-commented?

(3)
### INSTRUCTION
A car travels from City A to City B at an average speed of 60 km/h. On the way back from City B to City A, due to traffic, the car travels at an average speed of 40 km/h. What is the car's average speed for the entire round trip?

### Response
Analysis: The instruction is a math problem, requiring multi-step reasoning to calculate the average speed of car for a round trip between two cities. The instruction does not explicitly ask for only the answer, so it is probably best for a response to provide step-by-step reasoning before a final solution.
Answer: Does the response correctly provide the car's average speed for the entire round trip?
Does the response provide sensible step-by-step reasoning before a final solution?

(4)
### INSTRUCTION
Describe how to get from London to Paris in fewer than 8 steps. Do not mention the Eurostar.

### Response
Analysis: The instruction asks for a description of a journey from London to Paris using fewer than 8 steps. The description must not mentioning the Eurostar.
Answer: Does the response describe a valid journey from London to Paris?
Does the response provide a journey that takes fewer than 8 steps?
Does the response refrain from mentioning the Eurostar?

## Real Task

### INSTRUCTION
{input}

### Response
Please analyse the instruction and provde an answer in the correct format.
Remember that each question should be phrased such that answering with 'yes' would mean that the response **successfully** fulfilled the criteria being assessed by the question.
Your checklist should contain at least two questions, but no more than eight.
