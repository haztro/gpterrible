import json

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Create a dictionary containing both stoi and itos mappings
tokenizer_dict = {'stoi': stoi, 'itos': itos}

# Write to a JSON file
with open('../tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_dict, f, ensure_ascii=False, indent=4)
