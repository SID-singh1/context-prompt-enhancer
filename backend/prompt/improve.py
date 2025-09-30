import os
import uuid
import numpy as np
import torch
import json
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CONFIGURATION ---
PROMPTS_FILENAME = "prompts.json" # No longer need the hardcoded USER_ID

# --- 2. GLOBAL OBJECTS (Loaded once at startup) ---
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on application startup.
    It loads the models, initializes the client, and seeds the database.
    """
    print("--- 🚀 Application Starting Up... ---")

    print("Loading embedding model (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app_state["embedding_model"] = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    print(f"Embedding model loaded successfully on {device}.")

    print("Initializing Groq client...")
    try:
        app_state["groq_client"] = Groq()
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        app_state["groq_client"] = None

    db_instance = MockVectorDB()
    seed_database(db_instance, filename=PROMPTS_FILENAME)
    app_state["db_instance"] = db_instance

    print("--- ✅ Application Startup Complete ---")
    yield
    print("---  shutting down ---")

app = FastAPI(lifespan=lifespan)

# --- CORS MIDDLEWARE ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. VECTOR DATABASE SIMULATOR ---
class MockVectorDB:
    """A simple in-memory vector database simulator."""
    def __init__(self):
        self.vectors = {}
        self.metadata = {}

    def store(self, vector_id: str, vector: list[float], metadata: dict):
        self.vectors[vector_id] = np.array(vector)
        self.metadata[vector_id] = metadata
        print(f"-> DB: Stored prompt '{metadata['prompt_text'][:40]}...' in memory.")

    def query(self, query_vector: list[float], user_id: str, top_k: int = 3) -> list[dict]:
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

        return [
            {'text': self.metadata[ids[i]]['prompt_text'], 'score': similarities[i]}
            for i in top_k_indices
        ]

# --- 4. CORE LOGIC FUNCTIONS ---
def get_embedding(text: str) -> list[float]:
    model = app_state.get("embedding_model")
    if model is None:
        raise RuntimeError("Embedding model not loaded.")
    return model.encode(text, convert_to_tensor=False).tolist()

def append_prompt_to_json(user_id: str, prompt_text: str, filename: str):
    print(f"   - Persisting new prompt to '{filename}'...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    user_prompts = data.get(user_id, [])
    if prompt_text not in user_prompts:
        user_prompts.append(prompt_text)
        data[user_id] = user_prompts
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   - Successfully saved prompt for user '{user_id}'.")
    else:
        print("   - Prompt already exists for this user. Not adding duplicate.")

def enhance_prompt(user_id: str, original_prompt: str, db: MockVectorDB) -> str:
    print(f"\n1. Enhancing prompt for user '{user_id}'...")
    query_vector = get_embedding(original_prompt)
    context = db.query(query_vector, user_id=user_id, top_k=3)

    if not context:
        print("   - No similar prompts found in user history.")
        meta_prompt = f"Please rewrite this prompt to be more effective: '{original_prompt}'. Define a clear goal, target audience, and desired output format."
    else:
        print(f"   - Found {len(context)} similar prompts to use as context.")
        print(f"{context}")
        texts = [item['text'] for item in context]
        formatted_context = "\n\n---\n\n".join(texts)
        meta_prompt = f"""
You are an AI assistant that enhances user prompts. Your goal is to rewrite a user's new prompt by learning from their past prompts to understand their intentions, typical audience, and desired output format.

**CONTEXT FROM USER'S HISTORY:**
{formatted_context}

**USER'S NEW PROMPT TO ENHANCE:**
"{original_prompt}"

**YOUR TASK:**
Analyze the patterns in the user's history and rewrite the "NEW PROMPT" into a detailed, structured, and highly effective prompt that reflects these patterns. Provide only the final, enhanced prompt as your output.

Give me the output in the json format:
"
{
    "prompt: enchanced prompt over here "
}
"
"""
    client = app_state.get("groq_client")
    if not client:
        return "Error: Groq client not initialized."

    try:
        print("   - Sending request to Groq API...")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": meta_prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with the Groq API: {e}")

# --- 5. INITIAL DATABASE SEEDING ---
def seed_database(db: MockVectorDB, filename: str):
    print(f"\nSeeding database from '{filename}'...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        for user_id, prompts in data.items():
            for i, p in enumerate(prompts):
                prompt_id = f"{user_id}_init_{i}"
                embedding = get_embedding(p)
                metadata = {'user_id': user_id, 'prompt_text': p}
                db.vectors[prompt_id] = np.array(embedding)
                db.metadata[prompt_id] = metadata
        print("Database seeding from local file complete.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"   - '{filename}' not found or is empty. Will create it on first run.")

# --- 6. API ENDPOINT DEFINITION ---

# Pydantic model for the request body - NOW INCLUDES user_id
class PromptRequest(BaseModel):
    prompt: str
    user_id: str # <-- ADDED THIS FIELD

@app.post("/enhance-prompt")
async def process_and_store_prompt(request: PromptRequest):
    """
    Main API endpoint.
    Receives a user_id and prompt, enhances it, stores the original,
    and returns the result.
    """
    original_prompt = request.prompt
    user_id = request.user_id # <-- GET user_id from the request
    db = app_state.get("db_instance")
    
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    # Main workflow: enhance and then store
    enhanced_prompt = enhance_prompt(user_id, original_prompt, db)

    print("\n2. Storing original prompt...")
    prompt_vector = get_embedding(original_prompt)
    prompt_id = f"prompt_{uuid.uuid4()}"
    metadata = {'user_id': user_id, 'prompt_text': original_prompt}
    db.store(prompt_id, prompt_vector, metadata)

    append_prompt_to_json(user_id, original_prompt, filename=PROMPTS_FILENAME)

    print("\n--- ✅ API REQUEST COMPLETE ---")
    return {
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "user_id": user_id
    }

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Prompt Enhancer API is running."}

