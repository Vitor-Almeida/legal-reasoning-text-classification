from datasets import load_dataset

# Load the subset of the 'lex_glue' dataset
dataset = load_dataset("openai/gsm8k", "main")

# Define the path where you want to save the dataset
save_path = "data/gsm8k"

dataset.save_to_disk(save_path)