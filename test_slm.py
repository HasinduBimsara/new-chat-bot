import torch
import tiktoken
from backend.inference import load_slm_model, generate_local_response

model = load_slm_model()

tests = [
    "මට නිතරම කේන්ති යනවා.",
    "මම ගොඩක් දුකින් ඉන්නේ.",
    "මට නිදාගන්න අමාරුයි."
]

for t in tests:
    print("User:", t)
    print("Bot:", generate_local_response("system", t, temperature=0.3))
    print("-" * 50)
