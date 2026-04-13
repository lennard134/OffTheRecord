import ollama
import pandas as pd 
import numpy as np
from ollama import chat


data = [
    {
        "text": "The dough had that perfect leopard-spotted char you only get from a proper wood-fired oven. Toppings were sparse but deliberate — a few slices of nduja that melted into little puddles of spice. Incredible.",
        "label": "pizza"
    },
    {
        "text": "Smashed thin and cooked hard on a flat-top, the edges were almost lacy with crispiness. Topped with a dijonnaise and thinly sliced gherkins. Simple but executed flawlessly.",
        "label": "burger"
    },
    {
        "text": "Soft, pillowy wrap folded around slow-braised beef cheek with a bright tomatillo sauce. The fresh coriander and diced white onion on top kept it from feeling heavy. Clean, confident flavours.",
        "label": "taco"
    },
    {
        "text": "Deep dish style with a buttery, almost focaccia-like base. The sauce goes on top of the cheese here which felt wrong but tasted completely right. Left me in a carb-induced stupor for the rest of the afternoon.",
        "label": "pizza"
    },
    {
        "text": "The grind on the meat was coarse and you could taste it — properly beefy, not just seasoning doing the heavy lifting. Stacked with aged cheddar and a charred onion jam that was quietly the star of the show.",
        "label": "burger"
    },
    {
        "text": "Shrimp with a smoky al pastor-style marinade, served in a warm corn wrap with pineapple salsa. The sweetness cut through the char beautifully. Ate two before I even sat down.",
        "label": "taco"
    },
    {
        "text": "Sourdough base with a long cold ferment — you could taste the complexity in the crust. Topped with stracciatella and a drizzle of honey that shouldn't work but absolutely does.",
        "label": "pizza"
    },
    {
        "text": "Dry-aged patty with a crust that crackled when you bit through it. The bun was toasted just enough to hold together without going hard. Ate it in the car and have zero regrets.",
        "label": "burger"
    },
    {
        "text": "Crispy fried cauliflower in a flour wrap with a smoky chipotle crema and quick-pickled jalapeños. Vegetarian but punchy enough that nobody at the table missed the meat.",
        "label": "taco"
    },
    {
        "text": "Roman-style, sold by weight, eaten standing at the counter. The potato and rosemary slice was almost too simple — but the olive oil in the dough made every bite rich and satisfying in a way that's hard to explain.",
        "label": "pizza"
    },
]


model = "gemma3:1b"


count = 0
for sample in data:
	prompt = f"""
	Choose if the text is about pizza, burger or taco.
	Text to classify: {sample['text']}
	"""


	TEMPERATURE = 0.0

	ROLE = "user"

	response = chat(
	        model=model,
	        messages=[{'role': ROLE, 'content': prompt}],
	        options={"temperature":TEMPERATURE}

	    )
	out = response.message.content.strip()
	print(f"Response: {out}, Correct answer: {sample['label']}")
	if out.lower() == sample['label'].lower():
		count += 1 
print(f"Final overall performance: {count/10*100}.")






