import random
import json
from pathlib import Path

clothing_colors = ["black", "red", "blue", "green", "yellow", "white"]
hair_colors     = ["black", "brown", "blonde", "ginger", "red", "gray"]
eye_colors      = ["brown", "blue", "green", "gray", "hazel"]
genders         = ["man", "woman"]
expressions     = ["no", "normal", "very slight"]
skin_tones      = ["fair", "medium", "dark"]
facial_hairs    = ["no facial hair", "beard", "mustache"]
facial_hair_colors = ["black", "brown", "blonde", "ginger", "red", "gray"]
facial_features_list = ["freckles", "acne", "scars", "birthmarks"]
min_age, max_age = 18, 65

def generate_prompt():
    clothing_color      = random.choice(clothing_colors)
    hair_color          = random.choice(hair_colors)
    eye_color           = random.choice(eye_colors)
    gender              = random.choice(genders)
    skin_tone           = random.choice(skin_tones)
    facial_features     = random.choice(facial_features_list)

    if gender == "man":
        facial_hair       = random.choice(facial_hairs)
        facial_hair_color = random.choice(facial_hair_colors)
    else:
        facial_hair       = "no facial hair"
        facial_hair_color = "n/a"

    age                 = random.randint(min_age, max_age)
    inner_eyebrow_raise = random.choice(expressions)
    brow_lowerer        = random.choice(expressions)
    lip_corner_depressor= random.choice(expressions)

    return (
        f"Create a hyperrealistic headshot of a {gender} with {hair_color} hair and {eye_color} eyes. "
        f"This person is looking depressed and is wearing a {clothing_color} T-shirt. "
        f"They are {age} years old. They have a {inner_eyebrow_raise} inner eyebrow raise, "
        f"a {brow_lowerer} brow lowerer, and a {lip_corner_depressor} lip corner depressor. "
        f"The person's eyes are open and focused. "
        f"The overall facial tone is active as if the person is energetic and well rested. "
        f"The head tilts upward slightly, and the face carries a sense of normal confidence. "
        f"The person's skin tone is {skin_tone}. "
        f"The person has {facial_features}. "
        f"The person has {facial_hair} with {facial_hair_color} color."
    )

def main(n=100, outfile="prompts.jsonl"):
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for i in range(n):
            prompt = generate_prompt()
            record = {"id": i+1, "prompt": prompt}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {n} prompts to {outfile}")

if __name__ == "__main__":
    # change n or outfile as you like
    main(n=100, outfile="prompts.jsonl")