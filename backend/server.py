from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import PyPDF2
import io
import httpx

DB_AVAILABLE = False
notes_memory_store: dict[str, dict] = {}

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (optional; app falls back to in-memory storage)
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(
    mongo_url,
    serverSelectionTimeoutMS=int(os.environ.get('MONGO_SERVER_SELECTION_TIMEOUT_MS', '1500')),
)
db = client[os.environ.get('DB_NAME', 'test_database')]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


async def _init_db_connection() -> None:
    """Initialize DB connectivity. If Mongo isn't reachable, keep running with memory storage."""
    global DB_AVAILABLE
    try:
        await client.admin.command("ping")
        DB_AVAILABLE = True
        logging.info("MongoDB connected")
    except Exception as e:
        DB_AVAILABLE = False
        logging.warning(f"MongoDB not reachable; using in-memory notes store. Details: {e}")

# Models
class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str  # "pdf" or "text"
    original_filename: Optional[str] = None
    extracted_text: str
    notes_content: str
    notes_length: str  # "short", "medium", "detailed"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GenerateNotesRequest(BaseModel):
    text: str
    notes_length: str = "medium"
    source_type: str = "text"
    original_filename: Optional[str] = None

class GenerateNotesResponse(BaseModel):
    note_id: str
    notes_content: str

# Helper function to extract text from PDF
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")

# Helper function to generate notes using AI
async def generate_notes_with_ai(text: str, length: str) -> str:
    try:
        # Try multiple free APIs in order: Groq (free), Gemini, OpenAI
        groq_key = os.environ.get('GROQ_API_KEY')
        gemini_key = os.environ.get('GEMINI_API_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        # Configure prompt based on length
        length_instructions = {
            "short": "Create concise notes (2-3 key sections with 3-5 bullet points each)",
            "medium": "Create comprehensive notes (4-6 sections with detailed bullet points)",
            "detailed": "Create extensive notes (6-10 sections with thorough explanations and sub-points)"
        }
        
        instruction = length_instructions.get(length, length_instructions["medium"])
        
        prompt = f"""You are an expert note-taking assistant. Convert the provided content into well-structured study notes.

Formatting rules:
- Use clear markdown headings (## for main topics, ### for subtopics)
- Use bullet points (-) for lists
- Bold important terms using **term**
- Create clear sections
- Include key concepts and takeaways
- {instruction}
- Make notes easy to scan and study from

Content to convert into notes:

{text[:8000]}"""
        
        async with httpx.AsyncClient() as client:
            # Try Groq API first (FREE and FAST!)
            if groq_key and groq_key != 'your-groq-api-key-here':
                try:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {groq_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": [
                                {"role": "system", "content": "You are an expert note-taking assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 2048
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        logging.warning(f"Groq API failed: {response.text}")
                except Exception as e:
                    logging.warning(f"Groq API error: {str(e)}")
            
            # Try Gemini API second if available
            if gemini_key and gemini_key != 'your-gemini-api-key-here':
                try:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                        headers={
                            "Content-Type": "application/json"
                        },
                        json={
                            "contents": [{
                                "parts": [{
                                    "text": prompt
                                }]
                            }],
                            "generationConfig": {
                                "temperature": 0.7,
                                "maxOutputTokens": 2048,
                            }
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        logging.warning(f"Gemini API failed: {response.text}")
                except Exception as e:
                    logging.warning(f"Gemini API error: {str(e)}")
            
            # Fallback to OpenAI if available
            if openai_key and openai_key != 'your-openai-api-key-here':
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are an expert note-taking assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2048
                    },
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.text}")
                
                result = response.json()
                return result['choices'][0]['message']['content']
            
            raise HTTPException(status_code=500, detail="No API key configured. Please set GROQ_API_KEY (free), GEMINI_API_KEY, or OPENAI_API_KEY in .env file")
        
    except httpx.HTTPError as e:
        logging.error(f"HTTP error during AI generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error connecting to AI service: {str(e)}")
    except Exception as e:
        logging.error(f"AI generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating notes: {str(e)}")

# Routes
@api_router.get("/")
async def root():
    return {"message": "NoteGenius AI API"}

@api_router.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from uploaded PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    contents = await file.read()
    extracted_text = extract_text_from_pdf(contents)
    
    if not extracted_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "word_count": len(extracted_text.split())
    }

@api_router.post("/generate-notes", response_model=GenerateNotesResponse)
async def generate_notes(request: GenerateNotesRequest):
    """Generate AI-powered notes from text"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty")
    
    # Generate notes using AI
    notes_content = await generate_notes_with_ai(request.text, request.notes_length)
    
    # Create note document
    note = Note(
        source_type=request.source_type,
        original_filename=request.original_filename,
        extracted_text=request.text,
        notes_content=notes_content,
        notes_length=request.notes_length
    )
    
    # Store note (MongoDB if available, otherwise in-memory)
    doc = note.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()

    if DB_AVAILABLE:
        try:
            await db.notes.insert_one(doc)
        except Exception as e:
            logging.warning(f"Mongo insert failed; falling back to in-memory store. Details: {e}")
            notes_memory_store[note.id] = doc
    else:
        notes_memory_store[note.id] = doc
    
    return GenerateNotesResponse(
        note_id=note.id,
        notes_content=notes_content
    )

@api_router.get("/notes-history", response_model=List[Note])
async def get_notes_history():
    """Get all notes history"""
    if DB_AVAILABLE:
        notes = await db.notes.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    else:
        notes = list(notes_memory_store.values())
        notes.sort(key=lambda n: n.get("created_at", ""), reverse=True)
        notes = notes[:100]

    for note in notes:
        if isinstance(note.get('created_at'), str):
            note['created_at'] = datetime.fromisoformat(note['created_at'])

    return notes

@api_router.get("/notes/{note_id}", response_model=Note)
async def get_note(note_id: str):
    """Get a specific note by ID"""
    if DB_AVAILABLE:
        note = await db.notes.find_one({"id": note_id}, {"_id": 0})
    else:
        note = notes_memory_store.get(note_id)
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    if isinstance(note['created_at'], str):
        note['created_at'] = datetime.fromisoformat(note['created_at'])
    
    return note

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_db_client():
    await _init_db_connection()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()