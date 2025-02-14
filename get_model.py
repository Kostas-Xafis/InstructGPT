import os
from huggingface_hub import hf_hub_download


def get_model(name: str):
    # Define the model repository and filename
    if name == "Llama":
        repo_id = "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF" 
        filename = "Llama-3.2-3B-Instruct.Q4_K_M.gguf"
    elif name == "Aya":
        repo_id = "bartowski/aya-expanse-8b-GGUF"
        filename = "aya-expanse-8b-Q4_K_M.gguf"
    else:
        raise ValueError(f"Model '{name}' not found.")

    local_dir = "models/"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    if not filename in os.listdir(local_dir):
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"Model downloaded to: {model_path}")
    
    return local_dir + filename