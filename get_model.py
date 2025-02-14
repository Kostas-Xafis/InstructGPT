from huggingface_hub import hf_hub_download

# Define the model repository and filename
# repo_id = "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF" 
# filename = "Llama-3.2-3B-Instruct.Q4_K_M.gguf"

repo_id = "bartowski/aya-expanse-8b-GGUF"
filename = "aya-expanse-8b-Q4_K_M.gguf"

local_dir = "models/"
model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

print(f"Model downloaded to: {model_path}")