from ctransformers import AutoModelForCausalLM
import time
from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from tqdm import tqdm
import hashlib

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf?download=true"
MODEL_DIR = "models"
MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q4_K_S.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_file(url: str, destination: str):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def ensure_model_exists():
    """Ensure the model file exists, download if necessary."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Downloading model from Hugging Face...")
        download_file(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    else:
        print(f"Model found at {MODEL_PATH}")

class Chatbot:
    def __init__(self):
        self.model = None
        self.conversation_history: List[Tuple[str, str]] = []
        self.max_history = 5  # Keep last 5 exchanges for context
        
    def initialize_model(self):
        if self.model is None:
            print("Initializing Mistral chatbot...")
            ensure_model_exists()  # Ensure model is downloaded
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                model_file=MODEL_FILENAME,
                model_type="mistral",
                max_new_tokens=256,  # Reduced for faster responses
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                context_length=1024,  # Reduced for faster processing
                threads=4  # Utilize multiple CPU threads
            )
            print("Chatbot ready! Type 'quit' to exit.")
            print("-" * 50)

    def format_prompt(self, user_input: str) -> str:
        # Format conversation history
        history_text = ""
        for user_msg, assistant_msg in self.conversation_history[-self.max_history:]:
            history_text += f"[INST] {user_msg} [/INST] {assistant_msg}\n"
        
        # Add current user input
        return f"{history_text}[INST] {user_input} [/INST]"

    def update_history(self, user_input: str, response: str):
        self.conversation_history.append((user_input, response))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_response(self, user_input: str) -> dict:
        start_time = time.time()
        prompt = self.format_prompt(user_input)
        response = self.model(prompt)
        end_time = time.time()
        
        response = response.strip()
        self.update_history(user_input, response)
        
        return {
            "response": response,
            "response_time": round(end_time - start_time, 2)
        }

# Initialize FastAPI app
app = FastAPI(title="Mistral Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize chatbot
chatbot = Chatbot()
chatbot.initialize_model()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    response_time: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = chatbot.get_response(request.message)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": chatbot.model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
