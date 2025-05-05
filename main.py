from ctransformers import AutoModelForCausalLM
import time
from typing import List, Tuple, Optional, Dict
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
        self.sessions: Dict[str, List[Tuple[str, str]]] = {}
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

    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get or create session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def format_prompt(self, user_input: str, session_id: str) -> str:
        history = self.get_session_history(session_id)
        history_text = ""
        for user_msg, assistant_msg in history[-self.max_history:]:
            history_text += f"[INST] {user_msg} [/INST] {assistant_msg}\n"
        return f"{history_text}[INST] {user_input} [/INST]"

    def update_history(self, session_id: str, user_input: str, response: str):
        history = self.get_session_history(session_id)
        history.append((user_input, response))
        if len(history) > self.max_history:
            history.pop(0)

    def get_response(self, user_input: str, session_id: str = "default") -> dict:
        start_time = time.time()
        prompt = self.format_prompt(user_input, session_id)
        response = self.model(prompt)
        end_time = time.time()
        
        response = response.strip()
        self.update_history(session_id, user_input, response)
        
        return {
            "response": response,
            "response_time": round(end_time - start_time, 2),
            "session_id": session_id
        }

# Initialize FastAPI app
app = FastAPI(
    title="Mistral Chatbot API",
    description="A FastAPI-based chatbot using Mistral-7B model",
    version="1.0.0"
)

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = Chatbot()
chatbot.initialize_model()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    response_time: float
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = chatbot.get_response(request.message, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": chatbot.model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "active_sessions": len(chatbot.sessions)
    }

if __name__ == "__main__":
    # Run on all network interfaces (0.0.0.0) to allow external access
    uvicorn.run(
        app, 
        host="0.0.0.0",  # This allows external access
        port=8000,
        workers=1  # Single worker for the model
    )
