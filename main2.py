# import os
# from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# import speech_recognition as sr
# from gtts import gTTS
# from openai import AzureOpenAI
# from dotenv import load_dotenv
# import uuid
# import json
# from typing import List, Dict
# import logging
# from pydub import AudioSegment

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize Azure OpenAI client
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version=AZURE_OPENAI_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT
# )

# # Directory to store audio responses
# AUDIO_DIR = "static/audio"
# os.makedirs(AUDIO_DIR, exist_ok=True)

# def convert_to_wav(input_path: str, output_path: str) -> str:
#     logger.info(f"Converting audio from {input_path} to {output_path}")
#     try:
#         audio = AudioSegment.from_file(input_path)
#         audio = audio.set_channels(1).set_frame_rate(16000)  # Mono, 16kHz for speech_recognition
#         audio.export(output_path, format="wav")
#         logger.info(f"Converted audio to WAV: {output_path}")
#         return output_path
#     except Exception as e:
#         logger.error(f"Audio conversion error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Audio conversion error: {str(e)}")

# def speech_to_text(audio_path: str) -> str:
#     logger.info(f"Processing audio file: {audio_path}")
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(audio_path) as source:
#             audio_data = recognizer.record(source)
#             transcript = recognizer.recognize_google(audio_data)
#             logger.info(f"Transcript: {transcript}")
#             return transcript
#     except sr.UnknownValueError:
#         logger.error("Speech recognition could not understand audio")
#         raise HTTPException(status_code=400, detail="Could not understand audio")
#     except sr.RequestError as e:
#         logger.error(f"Speech recognition error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Speech recognition error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Unexpected error in speech_to_text: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# def text_to_speech(text: str, audio_file_path: str) -> str:
#     logger.info(f"Generating audio for text: {text[:50]}...")
#     try:
#         tts = gTTS(text=text, lang='en')
#         tts.save(audio_file_path)
#         logger.info(f"Audio saved to: {audio_file_path}")
#         return audio_file_path
#     except Exception as e:
#         logger.error(f"Text-to-speech error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Text-to-speech error: {str(e)}")

# # def get_openai_response(messages: List[Dict[str, str]]) -> str:
# #     logger.info(f"Sending messages to OpenAI: {messages}")
# #     system_message = [{"role": "system", "content": "You are a helpful AI chatbot that answers questions asked by the user."}]
# #     try:
# #         response = client.chat.completions.create(
# #             model="gpt-35-turbo",
# #             messages=system_message + messages
# #         )
# #         response_text = response.choices[0].message.content
# #         logger.info(f"OpenAI response: {response_text[:50]}...")
# #         return response_text
# #     except Exception as e:
# #         logger.error(f"OpenAI API error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# # Read resume text from file
# def load_resume_text(file_path: str = "resume.txt") -> str:
#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             resume_text = file.read().strip()
#         logger.info("Successfully loaded resume text")
#         return resume_text
#     except FileNotFoundError:
#         logger.error(f"Resume file not found: {file_path}")
#         raise HTTPException(status_code=500, detail="Resume file not found")
#     except Exception as e:
#         logger.error(f"Error reading resume file: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error reading resume file: {str(e)}")

# # Load resume text once at module level
# RESUME_TEXT = load_resume_text()

# def get_openai_response(messages: List[Dict[str, str]]) -> str:
#     logger.info(f"Sending messages to OpenAI: {messages}")
#     system_prompt = f"""
# You are a voice bot representing me, answering questions as I would based on my resume. Use the following resume information to provide accurate, personal responses:

# {RESUME_TEXT}

# Instructions:
# - Answer questions as if you are me, using a conversational, intuitive tone in simple English.
# - Reflect my personality and experiences from the resume.
# - Only respond to questions about my life, skills, experiences, or personal traits (e.g., life story, superpower, growth areas, misconceptions, boundaries).
# - If the question is unrelated or inappropriate, politely say: "I'd prefer to focus on questions about my background and experiences."
# - Keep responses concise, natural, and engaging, as if I'm speaking in an interview.
# - Avoid fabricating details not in the resume; if unsure, give a general response based on the resume context.
# """
#     system_message = [{"role": "system", "content": system_prompt}]
#     try:
#         response = client.chat.completions.create(
#             model="gpt-35-turbo",
#             messages=system_message + messages,
#             temperature=0.7,  # Balanced creativity for natural responses
#             max_tokens=500    # Limit response length for voice output
#         )
#         response_text = response.choices[0].message.content.strip()
#         logger.info(f"OpenAI response: {response_text[:50]}...")
#         return response_text
#     except Exception as e:
#         logger.error(f"OpenAI API error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# @app.post("/process_audio")
# async def process_audio(file: UploadFile = File(...), messages: str = Form(default="[]")):
#     logger.info("Received request to /process_audio")
#     try:
#         # Parse messages from JSON string
#         try:
#             messages_list = json.loads(messages)
#             if not isinstance(messages_list, list):
#                 raise ValueError("Messages must be a list")
#             logger.info(f"Parsed messages: {messages_list}")
#         except json.JSONDecodeError as e:
#             logger.error(f"Invalid messages JSON: {str(e)}")
#             raise HTTPException(status_code=422, detail=f"Invalid messages format: {str(e)}")

#         # Save uploaded audio file
#         audio_id = str(uuid.uuid4())
#         temp_audio_path = f"temp_audio_{audio_id}.webm"  # Assume WebM initially
#         wav_audio_path = f"temp_audio_{audio_id}.wav"
#         logger.info(f"Saving audio to: {temp_audio_path}")
#         try:
#             with open(temp_audio_path, "wb") as f:
#                 content = await file.read()
#                 if not content:
#                     logger.error("Received empty audio file")
#                     raise HTTPException(status_code=422, detail="Empty audio file")
#                 f.write(content)
#         except Exception as e:
#             logger.error(f"Error saving audio file: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error saving audio file: {str(e)}")

#         # Convert to WAV
#         convert_to_wav(temp_audio_path, wav_audio_path)

#         # Speech to text
#         transcript = speech_to_text(wav_audio_path)

#         # Update messages with user input
#         messages_list.append({"role": "user", "content": transcript})

#         # Get OpenAI response
#         response_text = get_openai_response(messages_list)

#         # Text to speech
#         response_audio_path = os.path.join(AUDIO_DIR, f"response_{audio_id}.mp3")
#         audio_url = response_audio_path.replace("\\", '/')
#         text_to_speech(response_text, response_audio_path)
        
        
#         # audio_url = response_audio_path.replace("\\", '/')
#         # audio_url = f"/backend/{audio_url}"
#         print(audio_url)

#         # Return response
#         return {
#             "transcript": transcript,
#             "response": response_text,
#             "audio_url": '/'+audio_url,
#             "messages": messages_list + [{"role": "assistant", "content": response_text}]
#         }
#     except HTTPException as e:
#         logger.error(f"Error processing request: {str(e)}")
#         raise
#     finally:
#         # Clean up temporary audio files
#         for path in [temp_audio_path, wav_audio_path]:
#             if os.path.exists(path):
#                 try:
#                     os.remove(path)
#                     logger.info(f"Cleaned up temporary file: {path}")
#                 except Exception as e:
#                     logger.error(f"Error cleaning up temporary file: {str(e)}")

# # Serve static audio files
# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="static"), name="static")

# from fastapi.responses import HTMLResponse

# @app.get("/", response_class=HTMLResponse)
# async def serve_index():
#     with open("static/index.html", "r") as f:
#         return f.read()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




