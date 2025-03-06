from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"
save_path = "D:/TinyLLM-Edge/models/Phi2"  # Change path if needed

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.save_pretrained(save_path)

print(f"Model downloaded and saved to {save_path}")
