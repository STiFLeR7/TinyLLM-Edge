from huggingface_hub import snapshot_download
import os

# Define model repo and destination path
model_repo = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Mistral-7B-GPTQ"))

# Download the model
print(f"📥 Downloading {model_repo} to {model_path} ...")
snapshot_download(repo_id=model_repo, local_dir=model_path, local_dir_use_symlinks=False)
print(f"✅ Model downloaded successfully at: {model_path}")
