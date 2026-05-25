# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile

# Find the folder named 'public'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
public_path = os.path.join(root_dir, "public")

# Import our custom AI Engines (using absolute imports)
from backend.pdf_processor import extract_text_from_pdf
from backend.summary_engine import SummaryEngine
from backend.contextualizer import Contextualizer
from backend.vector_store import MedicalVectorStore
from backend.chat_engine import ChatEngine

# 1. Initialize the Web App
app = FastAPI(title="MediSimple AI API")

app.mount("/public", StaticFiles(directory=public_path), name="public")

# 2. Allow the frontend to talk to this backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize AI Components
summary_engine = SummaryEngine()
contextualizer = Contextualizer()
vector_store = MedicalVectorStore()
chat_engine = ChatEngine(vector_store)


# --- UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # A. Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            temp_pdf_path = tmp_file.name

        # B. Run Phase 1: Document Understanding
        print(f"Processing uploaded file: {file.filename}...")
        raw_text = extract_text_from_pdf(temp_pdf_path)

        # Get the structured JSON for the frontend dashboard
        summary_data = summary_engine.get_analysis(raw_text)

        # C. Run Phase 2 & 3: Contextual Retrieval & Vector Store
        print("Building AI Memory (Vector Store)...")
        # Use the summary_text to contextualize the chunks
        enriched_chunks = contextualizer.process_document(raw_text, summary_data.summary_text)
        vector_store.build_and_save(enriched_chunks)

        # Clean up the temporary file
        os.remove(temp_pdf_path)

        # D. Return the Dashboard Data to the Frontend
        print("Upload complete! Sending data to frontend.")
        return {
            "status": "success",
            "message": "Document processed and AI memory built.",
            "dashboard_data": summary_data.model_dump()  # Converts the Pydantic object to JSON
        }

    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- CHAT REQUEST SCHEMA ---
class ChatRequest(BaseModel):
    session_id: str
    message: str
    global_summary: str


# --- CHAT ENDPOINT ---
@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        print(f"Received message for session {request.session_id}: {request.message}")

        # Ask our Chat Engine
        answer = chat_engine.ask_question(
            session_id=request.session_id,
            question=request.message,
            global_summary=request.global_summary
        )

        # Send the answer back to the frontend
        return {
            "status": "success",
            "answer": answer
        }

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    # This sends your index.html file to the user's browser
    return FileResponse(os.path.join(public_path, "index.html"))