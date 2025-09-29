import os
import uuid
import argparse
import numpy as np
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- 1. INITIAL SETUP ---


# --- 2. GLOBAL OBJECTS (Load models once) ---

print("Loading embedding model (this may take a moment)...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
print(f"Embedding model loaded successfully on {DEVICE}.")

print("Initializing Groq client...")
try:
    GROQ_CLIENT = Groq()
    print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    GROQ_CLIENT = None

# --- 3. VECTOR DATABASE SIMULATOR ---

class MockVectorDB:
    """A simple in-memory vector database simulator."""
    def __init__(self):
        self.vectors = {}
        self.metadata = {}

    def store(self, vector_id: str, vector: list[float], metadata: dict):
        """Stores a vector and its metadata."""
        self.vectors[vector_id] = np.array(vector)
        self.metadata[vector_id] = metadata
        print(f"-> DB: Stored prompt '{metadata['prompt_text'][:40]}...' in memory.")

    def query(self, query_vector: list[float], user_id: str, top_k: int = 3) -> list[dict]:
        """Queries the database for the most similar vectors for a specific user. Returns list of {'text', 'score'} dicts."""
        query_vector = np.array(query_vector)
        user_vectors = {
            vid: vec for vid, vec in self.vectors.items()
            if self.metadata[vid]['user_id'] == user_id
        }
        if not user_vectors:
            return []
            
        ids = list(user_vectors.keys())
        vecs = np.array(list(user_vectors.values()))
        
        similarities = cosine_similarity(query_vector.reshape(1, -1), vecs)[0]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        # In class MockVectorDB, inside the query function:
        ...
        return [
            {'text': self.metadata[ids[i]]['prompt_text'] , 'score':similarities[i]}
            for i in top_k_indices
        ]

# --- 4. CORE LOGIC FUNCTIONS ---

def get_embedding(text: str) -> list[float]:
    """Generates a vector embedding for a given text."""
    return EMBEDDING_MODEL.encode(text, convert_to_tensor=False).tolist()

def append_prompt_to_json(user_id: str, prompt_text: str, filename: str = "prompts.json"):
    """Reads, updates, and writes back to the JSON file to persist the new prompt."""
    print(f"   - Persisting new prompt to '{filename}'...")
    try:
        # Read the existing data
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, start with an empty structure
        data = {}

    # Get the list of prompts for the user, or create it if it doesn't exist
    user_prompts = data.get(user_id, [])
    
    # Add the new prompt if it's not already there
    if prompt_text not in user_prompts:
        user_prompts.append(prompt_text)
        data[user_id] = user_prompts
        
        # Write the updated data back to the file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   - Successfully saved prompt for user '{user_id}'.")
    else:
        print("   - Prompt already exists for this user. Not adding duplicate.")


def enhance_prompt(user_id: str, original_prompt: str, db: MockVectorDB) -> str:
    """Finds context and calls the Groq API to enhance the prompt."""
    print(f"\n1. Enhancing prompt for user '{user_id}'...")
    query_vector = get_embedding(original_prompt)
    context = db.query(query_vector, user_id=user_id, top_k=3)

    if not context:
        print("   - No similar prompts found in user history.")
        meta_prompt = f"Please rewrite this prompt to be more effective: '{original_prompt}'. Define a clear goal, target audience, and desired output format."
    else:
        print(f"   - Found {len(context)} similar prompts to use as context.")
        texts = []
        for item in context:
            score = item.get('score')
            text = item.get('text', '')
            print(f" Score {score:.2f} - {text}")
            texts.append(text)
        formatted_context = "\n\n---\n\n".join(texts)
        meta_prompt = f"""
You are an AI assistant that enhances user prompts. Your goal is to rewrite a user's new prompt by learning from their past prompts to understand their intentions, typical audience, and desired output format.

**CONTEXT FROM USER'S HISTORY:**
{formatted_context}

**USER'S NEW PROMPT TO ENHANCE:**
"{original_prompt}"

**YOUR TASK:**
Analyze the patterns in the user's history and rewrite the "NEW PROMPT" into a detailed, structured, and highly effective prompt that reflects these patterns. Provide only the final, enhanced prompt as your output.
"""
    if not GROQ_CLIENT:
        return "Error: Groq client not initialized."

    try:
        print("   - Sending request to Groq API...")
        chat_completion = GROQ_CLIENT.chat.completions.create(
            messages=[{"role": "user", "content": meta_prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"

def process_and_store_prompt(user_id: str, original_prompt: str, db: MockVectorDB):
    """Main workflow: enhances a prompt and then stores the original prompt."""
    enhanced_prompt = enhance_prompt(user_id, original_prompt, db)

    print("\n2. Storing original prompt...")
    # Store in the temporary in-memory DB for the current session
    prompt_vector = get_embedding(original_prompt)
    prompt_id = f"prompt_{uuid.uuid4()}"
    metadata = {'user_id': user_id, 'prompt_text': original_prompt}
    db.store(prompt_id, prompt_vector, metadata)
    
    # *** NEW: Persist the prompt by writing to the JSON file ***
    append_prompt_to_json(user_id, original_prompt)
    
    print("\n--- ✅ ENHANCEMENT COMPLETE ---")
    print(f"**Original Prompt:**\n{original_prompt}\n")
    print(f"**Enhanced Prompt:**\n{enhanced_prompt}")
    print("---------------------------------")

# --- 5. INITIAL DATABASE SEEDING ---

def seed_database(db: MockVectorDB, filename: str = "prompts.json"):
    """Populates the database by reading from a prompts JSON file."""
    print(f"\nSeeding database from '{filename}'...")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        for user_id, prompts in data.items():
            for i, p in enumerate(prompts):
                prompt_id = f"{user_id}_init_{i}"
                embedding = get_embedding(p)
                metadata = {'user_id': user_id, 'prompt_text': p}
                # Only store in the in-memory DB, don't write back
                db.vectors[prompt_id] = np.array(embedding)
                db.metadata[prompt_id] = metadata

        print("Database seeding from local file complete.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"   - '{filename}' not found or is empty. Will create it on first run.")
        print("   - Running with an empty database.")

# --- 6. SCRIPT EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a user prompt based on their history.")
    parser.add_argument("--user-id", type=str, required=True, help="The ID of the user (e.g., 'priya' or 'arjun').")
    parser.add_argument("--prompt", type=str, required=True, help="The vague prompt you want to enhance.")
    
    args = parser.parse_args()

    db_instance = MockVectorDB()
    seed_database(db_instance)
    
    process_and_store_prompt(args.user_id, args.prompt, db_instance)