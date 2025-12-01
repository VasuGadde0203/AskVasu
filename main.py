import os
import uuid
import json
import logging
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import FileResponse

# -----------------------------------------
# LOGGING SETUP
# -----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------
# ENV + OPENAI CLIENT
# -----------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------
# FASTAPI app
# -----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# AUDIO DIRECTORY
# -----------------------------------------
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# -----------------------------------------
# LOAD RESUME
# -----------------------------------------
# def load_resume_text(file_path="resume.txt") -> str:
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             return f.read().strip()
#     except Exception as e:
#         logger.error(f"Resume load error: {e}")
#         raise HTTPException(status_code=500, detail="Failed to load resume file")

# RESUME_TEXT = load_resume_text()

# -----------------------------------------
# WHISPER STT (NO FFMPEG NEEDED)
# -----------------------------------------
def speech_to_text(audio_path: str) -> str:
    logger.info(f"Transcribing with Whisper: {audio_path}")
    try:
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"  
            )
        return transcript.text
    except Exception as e:
        logger.error(f"Whisper STT error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech recognition error: {str(e)}")

# -----------------------------------------
# OPENAI TTS
# -----------------------------------------
def text_to_speech(text: str, save_path: str) -> str:
    logger.info("Generating TTS using OpenAI")
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        with open(save_path, "wb") as f:
            f.write(response.read())
        return save_path
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

# -----------------------------------------
# CHAT COMPLETION
# -----------------------------------------
def get_openai_response(messages: List[Dict[str, str]]) -> str:
#     system_prompt = f"""
# You are a voice bot representing me, answering based on my resume:

# {RESUME_TEXT}

# Instructions:
# - Answer like me, in simple conversational English.
# - Only answer about my background, skills, personality, experience.
# - If unrelated: say "I'd prefer to focus on questions about my background and experiences."
# - Keep it short, natural, personal.
# - Never invent details not in my resume.

# Special Case: If asked "Tell me about yourself" or similar:
# - Give a 4–6 line clean intro:
#   - Who I am
#   - My skills
#   - Relevant projects
#   - What I'm looking for next
# """
    system_prompt = f"""
        You are a voice bot representing Vasu Gadde. Your job is to answer recruiter-style questions exactly how Vasu would answer in a real interview, based strictly on the information below.

    ------------------------------------------------------------------------
    Vasu’s Professional Profile (Complete Verified Information)
    ------------------------------------------------------------------------

    Vasu Ramesh Gadde  
    Data Scientist | AI/ML Engineer | Data Analyst  
    BSc Information Technology, University of Mumbai (2021–2024)  

    Strong in: Python, ML, GenAI, REST APIs, backend engineering, data pipelines, vector databases, RAG, LLM apps, and real-time data systems.

    ------------------------------------------------------------------------
    Current Role – AI/ML Engineer (JHS Associates & LLP)  
    (Last 6 months – present)
    ------------------------------------------------------------------------

    1. **AI-Powered Recruitment System (End-to-End)**
    - HR uploads resumes → processed → converted to embeddings (Pinecone)
    - Job role + description → semantic matching + OpenAI filtering
    - Sends WhatsApp messages to shortlisted candidates using Infobip
    - Webhook handles candidate replies + interview scheduling
    - HR panels for tracking shortlisted/selected/rejected/joined
    - Gmail-style analytics for pipeline health
    **Tech:** FastAPI, Python, MongoDB, Pinecone, OpenAI, Infobip, HTML/CSS/JS

    2. **Timesheet Application**
    - Employees submit daily work logs
    - Managers approve/reject
    - Admin dashboard with analytics
    - Future: chatbot-assisted timesheet entry

    3. **Stockeye (Audit App)**
    - Auditors fill daily audit tasks
    - Auto captures selfie + geolocation
    - Sends checklist report to managers
    - Recon module compares expected vs filled data
    - Updated admin panel delivered to client

    4. **Company-wide PowerBI Dashboards**
    - Built dashboards for HR, IT, Accounts, Timesheet analytics
    - Integrated workflows & KPI views

    5. **Workflow Automation**
    - Automated PDF → Excel → reconciliation pipelines
    - Automated data cleaning & report generation

    6. **Chatbot Development (RAG)**
    - HR policy chatbot
    - Audit manual chatbot
    - Company data chatbot (Work-in-progress)

    7. **Email Automation Tool**
    - Streamlit app used by founders for mass campaign emails

    8. **Regulatory Intelligence System (RIS)**
    - Fetch news from ET, TOI, etc., using SerpAPI
    - Daily scheduled at 10 AM via GitHub Actions
    - Extracts relevant updates + emails to founder

    9. **Upcoming Projects**
    - “JHS ChatGPT” – internal knowledge bot using embeddings
    - Website chatbot
    - Audit/HR manual automation
    - Data reconciliation automation

    ------------------------------------------------------------------------
    Past Experience – Machine Learning Intern (Supermoney Advisors)
    ------------------------------------------------------------------------

    1. Built **NLP and RAG systems** using OpenAI + Pinecone  
    2. Designed **end-to-end pipelines** using Python and MongoDB  
    3. Time-series forecasting for price trends  
    4. Created policy-based “Insurance Agent Chatbot” with level-wise answering  
    5. Extracted structured info from PDFs/DOCs  
    6. Built backend APIs for finance-related Q&A

    ------------------------------------------------------------------------
    Projects (Personal + Academic)
    ------------------------------------------------------------------------

    1. **WhatsApp Chat Analyzer (Streamlit)**
    - Chat statistics, timelines, heatmaps
    - Wordclouds, emoji analysis
    - Sentiment analysis, LDA topic modeling
    - Response time analytics
    - User-wise + Group-wise analysis

    2. **GitaChatBot**
    - FastAPI backend + Gemini
    - RAG using FAISS + MongoDB + S3
    - Login/registration system
    - Answers based on spiritual text

    3. **CryptoLab**
    - FastAPI backend for crypto forecasting
    - Binance API real-time data fetch
    - LSTM model for BTC prediction
    - Market sentiment classification  
    - JWT auth + SMTP mailer

    4. **Personal Research Assistant**
    - LangChain + FAISS + Azure OpenAI
    - Semantic search over research papers
    - JWT auth + conversation storage

    5. **Finance Quiz App**
    - Adaptive finance quizzes using GPT
    - Micro-learning modules

    6. **AI-Powered Document Analyzer**
    - PDF summarization (short/long)
    - Document Q&A using Gemini

    ------------------------------------------------------------------------
    Soft Answers (Personality-Based)
    ------------------------------------------------------------------------

    Life Story:
    - Curious about systems, data, and problem-solving.
    - Passionate about backend + AI + automation.
    - Grew through practical projects, APIs, and real business problems.

    Superpower:
    - Fast learner + adaptable.

    Growth Areas:
    - Advanced ML/AI depth  
    - Communication/storytelling  
    - Leadership/mentorship  

    Misconception:
    - Appears quiet initially but contributes strongly once comfortable.

    How I push boundaries:
    - Take on work slightly outside comfort zone; break down problems; experiment fast.

    ------------------------------------------------------------------------
    RESPONSE RULES
    ------------------------------------------------------------------------

    1. **Never hallucinate.**  
    If the resume doesn’t contain info, give a general answer.

    2. **Tone:**  
    - conversational  
    - confident  
    - simple English  
    - sounds like an actual candidate speaking  
    - no robotic phrasing  

    3. **If question is unrelated (personal, irrelevant, inappropriate):**  
    Respond with:  
    “I’d prefer to focus on questions about my background and experiences.”

    4. **For recruiter-style questions, follow interview best practices.**

    5. **Special Case → “Tell me about yourself”, “Introduce yourself”**  
    Provide a clean 4–6 line introduction:
    - who you are  
    - what you do  
    - key skills  
    - notable projects  
    - recent experience  
    - what you're looking for next

    6. **For “Explain your current role”, “Walk me through your experience”,  
    “What are you doing at JHS?” → Use the JHS section above.**

    7. **For “Explain your projects”, give structured, outcome-driven answers.**

    8. **For “Why should we hire you?” → Combine skills + proof + impact.**

    9. **ALWAYS stay within Vasu’s actual experience—do not invent.**

    10. **Use clean Markdown formatting (short paragraphs, bullet points, and numbered lists) whenever it improves clarity.**

    ------------------------------------------------------------------------
    END OF SYSTEM PROMPT
    ------------------------------------------------------------------------
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"ChatCompletion error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI chat error")

# -----------------------------------------
# /process_audio
# -----------------------------------------
# @app.post("/process_audio")
# async def process_audio(file: UploadFile = File(...), messages: str = Form(default="[]")):
#     logger.info("Received /process_audio")

#     try:
#         # Parse message history
#         try:
#             messages_list = json.loads(messages)
#             if not isinstance(messages_list, list):
#                 raise ValueError
#         except:
#             raise HTTPException(status_code=422, detail="Invalid messages JSON format")

#         # Save uploaded audio (WebM/WAV/anything Whisper accepts)
#         audio_id = str(uuid.uuid4())
#         audio_path = f"temp_{audio_id}.{file.filename.split('.')[-1]}"

#         with open(audio_path, "wb") as f:
#             data = await file.read()
#             if not data:
#                 raise HTTPException(status_code=422, detail="Empty audio file")
#             f.write(data)

#         # Whisper STT
#         transcript = speech_to_text(audio_path)

#         # Append transcript to conversation
#         messages_list.append({"role": "user", "content": transcript})

#         # Chat response
#         response_text = get_openai_response(messages_list)

#         # Generate TTS
#         response_audio_path = os.path.join(AUDIO_DIR, f"response_{audio_id}.mp3")
#         text_to_speech(response_text, response_audio_path)
#         audio_url = "/" + response_audio_path.replace("\\", "/")

#         # Cleanup temp file
#         os.remove(audio_path)

#         return {
#             "transcript": transcript,
#             "response": response_text,
#             "audio_url": audio_url,
#             "messages": messages_list + [{"role": "assistant", "content": response_text}]
#         }

#     except Exception as e:
#         logger.error(f"process_audio error: {e}")
#         raise

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...), messages: str = Form(default="[]")):
    logger.info("Received /process_audio (AUDIO MODE)")

    try:
        # Load conversation history
        messages_list = json.loads(messages)
        if not isinstance(messages_list, list):
            raise HTTPException(status_code=422, detail="Invalid messages format")

        # Save uploaded audio
        audio_id = str(uuid.uuid4())
        ext = file.filename.split('.')[-1]
        audio_path = f"temp_{audio_id}.{ext}"

        with open(audio_path, "wb") as f:
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=422, detail="Empty audio file")
            f.write(audio_bytes)

        # Whisper → text
        transcript = speech_to_text(audio_path)
        messages_list.append({"role": "user", "content": transcript})

        # LLM → text answer
        response_text = get_openai_response(messages_list)

        # TTS → audio answer
        response_audio_path = os.path.join(AUDIO_DIR, f"response_{audio_id}.mp3")
        text_to_speech(response_text, response_audio_path)
        audio_url = "/" + response_audio_path.replace("\\", "/")

        # Remove temp input
        os.remove(audio_path)

        return {
            "transcript": transcript,
            "response": response_text,
            "audio_url": audio_url,
            "messages": messages_list + [{"role": "assistant", "content": response_text}]
        }

    except Exception as e:
        logger.error(f"/process_audio error: {e}")
        raise


# -----------------------------------------
# /ask_question (text only)
# -----------------------------------------
class AskQuestionRequest(BaseModel):
    question: str
    messages: List[Dict[str, str]]

# @app.post("/ask_question")
# async def ask_question(req: AskQuestionRequest):
#     logger.info("Received /ask_question")

#     messages = req.messages
#     messages.append({"role": "user", "content": req.question})

#     response_text = get_openai_response(messages)

#     # TTS
#     audio_id = str(uuid.uuid4())
#     response_audio_path = os.path.join(AUDIO_DIR, f"response_{audio_id}.mp3")
#     text_to_speech(response_text, response_audio_path)
#     audio_url = "/" + response_audio_path.replace("\\", "/")

#     return {
#         "transcript": req.question,
#         "response": response_text,
#         "audio_url": audio_url,
#         "messages": messages + [{"role": "assistant", "content": response_text}]
#     }

@app.post("/ask_question")
async def ask_question(req: AskQuestionRequest):
    logger.info("Received /ask_question (TEXT MODE)")

    # Append user text to history
    messages = req.messages
    messages.append({"role": "user", "content": req.question})

    # LLM response
    response_text = get_openai_response(messages)

    # Return text-only response
    return {
        "transcript": req.question,
        "response": response_text,
        "audio_url": None,  # <-- IMPORTANT
        "messages": messages + [{"role": "assistant", "content": response_text}]
    }


# -----------------------------------------
# STATIC + FRONTEND
# -----------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse('static/index.html')
