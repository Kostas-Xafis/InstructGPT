import os
import numpy as np
import pandas as pd
from llama_cpp import Llama
from get_model import get_model
from parse_args import parse_args
from sklearn.metrics.pairwise import cosine_similarity
args = parse_args()

def _print(llm:str, lang:str, *pargs):
    print(*pargs)
    if args["log"]:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open(f"logs/{llm}_{lang}.txt", "a") as f:
            print(f"{llm} assistant ({lang}):", *pargs, file=f)

def get_llm(model: str, embed=False):
    return Llama(
        model_path=get_model(model),
        n_ctx=256,
        n_threads=12,
        verbose=False,
        embedding=embed
    )

def get_content(response):
    return response['choices'][0]['message']['content']

def simple_llm_prompting(llm: dict[str, Llama]):
    print("\n--- Simple Questions ---")
    # Chat histories
    messages_en = {
        "lang": "English",
        "prompts": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Greece?"},
            {"role": "assistant", "content": "The capital of Greece is Athens."},
            {"role": "user", "content": "Who wrote '20,000 Leagues Under the Sea'?"},
        ]
    }
    messages_gr = {
        "lang": "Greek",
        "prompts": [
        {"role": "system", "content": "Είσαι ένας χρήσιμος βοηθός."},
        {"role": "user", "content": "Ποια είναι η πρωτεύουσα της Ελλάδας;"},
        {"role": "assistant", "content": "Η πρωτεύουσα της Ελλάδας είναι η Αθήνα."},
        {"role": "user", "content": "Ποιος έγραψε το '20,000 Λεύγας Κάτω από τη Θάλασσα';"},
    ]}
    
    for name, model in llm.items():
        name = name.capitalize()
        for messages in [messages_en, messages_gr]:
            print(f"\n{name} Assistant ({messages['lang']}):")
            response = model.create_chat_completion(messages=messages['prompts'])
            _print(name, messages['lang'], get_content(response))

def variety_llm_prompting(llm: dict[str, Llama]):
    print("\n--- Variety of Questions ---")
    # Chat histories for different topics
    messages_variety_en = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about programmers."},
        {"role": "user", "content": "What is the meaning of life?"},
        {"role": "user", "content": "Respond with a short poem about Artificial Intelligence."},
    ]
    messages_variety_gr = [
        {"role": "system", "content": "Είσαι ένας χρήσιμος βοηθός."},
        {"role": "user", "content": "Πες μου ένα ανέκδοτο για προγραμματιστές."},
        {"role": "user", "content": "Ποιο είναι το νόημα της ζωής;"},
        {"role": "user", "content": "Γράψε ένα μικρό ποίημα για την Τεχνητή Νοημοσύνη."},
    ]
    
    messages_variety_en = {"lang": "English","prompts":[[messages_variety_en[0], messages_variety_en[i]] for i in range(1, 4)]}
    messages_variety_gr = {"lang": "Greek","prompts":[[messages_variety_gr[0], messages_variety_gr[i]] for i in range(1, 4)]}

    for name, model in llm.items():
        name = name.capitalize()
        for messages in [messages_variety_en, messages_variety_gr]:
            lang = messages['lang']
            for message in messages['prompts']:
                print(f"\n{name} Assistant ({lang}):")
                response = model.create_chat_completion(messages=message)
                _print(name, lang, get_content(response))

def alternative_responses(llm: Llama):
    print("\n--- Alternative Responses ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Respond with a short poem about Artificial Intelligence."},
    ]
    for temp in [0.1, 0.7, 1.0]:
        for top_p in [0.8, 0.9, 1.0]:
            print(f"\n---Temperature: {temp}, Top-p: {top_p}---")
            response = llm.create_chat_completion(messages=messages, temperature=temp, top_p=top_p)
            _print("Llama", "English", get_content(response))

def embedding_similarity(llm: Llama):
    sentences = [
        "Artificial Intelligence is transforming the job market.",
        "A car crash can be a traumatic experience.",
        "Alchohol is bad for your health.",
        "Deep learning is used in image recognition.",
        "Greece is known for it's delicious food.",
    ]

    embeds = [llm.create_embedding(input=s)["data"][0] for s in sentences]
    embeddings = [np.array(embed["embedding"], dtype=np.float32).flatten() for embed in embeds]
    
    # Pad embeddings with mean to the same length
    max_len = max([len(embed) for embed in embeddings])
    embeddings = [np.pad(embed, (0, max_len - len(embed)), mode='mean') for embed in embeddings]

    # print([len(embed) for embed in embeddings])

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(np.array(embeddings))

    print("\n--- Cosine Similarity Matrix ---")
    size = len(sentences)
    df = pd.DataFrame(similarity_matrix, 
                      columns=[f"Sentence {i+1}" for i in range(size)], 
                      index=[f"Sentence {i+1}" for i in range(size)])
    print(df.to_string())


if __name__ == "__main__":
    # Load models
    llama = get_llm("Llama")
    aya = get_llm("Aya")
    llms={"llama": llama, "aya": aya}
    if args["mode"] == 1:
        simple_llm_prompting(llms)
    elif args["mode"] == 2:
        variety_llm_prompting(llms)
    elif args["mode"] == 3:
        alternative_responses(llama)
    elif args["mode"] == 4:
        embedding_similarity(get_llm("Aya", embed=True))
    else:
        print("Invalid mode. Please choose 1, 2, 3, or 4.")
        exit(1)