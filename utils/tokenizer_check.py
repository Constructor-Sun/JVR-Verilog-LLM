from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-8B")
print(tokenizer.tokenize(" /no_think"))
print(tokenizer.tokenize(" /think "))