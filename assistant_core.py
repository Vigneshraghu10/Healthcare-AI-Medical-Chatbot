import os
import logging
import platform
import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq
from gtts import gTTS
import json
import time
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HealthcareAIAssistant:
    def __init__(self):
        # Load API keys from environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Validate API keys
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY. Please set it as an environment variable.")
        
        if not self.elevenlabs_api_key:
            logging.warning("Missing ELEVENLABS_API_KEY. Premium voice features will be unavailable.")

        # Initialize Groq client with error handling
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            raise

        # Model configurations
        self.llm_model = "llama-3.3-70b-versatile"
        self.stt_model = "whisper-large-v3"

        # Cache for API responses
        self.cache = {}

        # Supported languages
        self.languages = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "hi": "Hindi",
            "zh": "Chinese",
            "ja": "Japanese",
            "ru": "Russian",
            "ar": "Arabic",
            "pt": "Portuguese",
            "bn": "Bengali",
            "ko": "Korean"
        }

    def transcribe_audio(self, audio_filepath, language="en"):
        """
        Transcribe audio file to text using Groq's Whisper model.
        
        Args:
            audio_filepath (str): Path to the audio file
            language (str): Language code for transcription
            
        Returns:
            str: Transcribed text or error message
        """
        try:
            # Validate input
            if not audio_filepath or not os.path.exists(audio_filepath):
                return "Error: Audio file not found"
            
            # Check file size
            file_size = os.path.getsize(audio_filepath)
            if file_size == 0:
                return "Error: Audio file is empty"
            
            logging.info(f"Transcribing audio file: {audio_filepath} ({file_size} bytes)")
            
            # Create cache key
            cache_key = f"transcribe_{Path(audio_filepath).stem}_{language}_{file_size}"
            
            if cache_key in self.cache:
                logging.info("Returning cached transcription")
                return self.cache[cache_key]
            
            # Compress if file is too large (>20MB)
            if file_size > 20 * 1024 * 1024:
                logging.info("Audio file too large, compressing...")
                compressed_path = self._compress_audio(audio_filepath)
                if compressed_path:
                    audio_filepath = compressed_path
            
            # Transcribe with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(audio_filepath, "rb") as audio_file:
                        transcription = self.groq_client.audio.transcriptions.create(
                            model=self.stt_model,
                            file=audio_file,
                            language=language
                        )
                    
                    result = transcription.text.strip()
                    if result:
                        self.cache[cache_key] = result
                        return result
                    else:
                        return "Error: No speech detected in audio"
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
                        
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return f"Error transcribing audio: {str(e)}"

    def _compress_audio(self, audio_filepath):
        """Compress audio file to reduce size"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            audio = AudioSegment.from_file(audio_filepath)
            audio.export(temp_path, format="mp3", bitrate="64k")
            
            logging.info(f"Compressed audio from {os.path.getsize(audio_filepath)} to {os.path.getsize(temp_path)} bytes")
            return temp_path
            
        except Exception as e:
            logging.error(f"Audio compression error: {e}")
            return None

    def analyze_text(self, text_query, language="en"):
        """
        Analyze text and generate medical notes.
        
        Args:
            text_query (str): Text to analyze
            language (str): Language code for response
            
        Returns:
            str: Generated medical notes
        """
        try:
            if not text_query or not text_query.strip():
                return "Error: No text provided for analysis"
            
            # Create cache key
            cache_key = f"analyze_{hash(text_query)}_{language}"
            
            if cache_key in self.cache:
                logging.info("Returning cached analysis")
                return self.cache[cache_key]
            
            language_instruction = ""
            if language != "en":
                language_instruction = f" Respond in {self.languages.get(language, 'English')}."
            
            system_prompt = (
                "You are an experienced medical professional creating structured clinical notes. "
                "Format your response with clear sections:\n"
                "1. CHIEF COMPLAINT: Brief primary concern\n"
                "2. HISTORY OF PRESENT ILLNESS: Detailed chronology with onset, duration, severity\n"
                "3. PAST MEDICAL HISTORY: Previous conditions (if mentioned)\n"
                "4. ASSESSMENT: Diagnostic impressions and differential diagnoses\n"
                "5. PLAN: Specific treatment recommendations and follow-up\n\n"
                "Be concise but comprehensive. Use appropriate medical terminology. "
                "If information is missing, note what additional information would be helpful."
                + language_instruction
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_query}
            ]
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.llm_model,
                temperature=0.2
            )
            
            result = chat_completion.choices[0].message.content
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return f"Error analyzing text: {str(e)}"

    def generate_emr_content(self, transcription, language="en"):
        """
        Generate structured EMR content in JSON format.
        
        Args:
            transcription (str): Patient consultation text
            language (str): Language code
            
        Returns:
            str: EMR content in JSON format
        """
        try:
            if not transcription or not transcription.strip():
                return json.dumps({"error": "No transcription provided"})
            
            cache_key = f"emr_{hash(transcription)}_{language}"
            
            if cache_key in self.cache:
                logging.info("Returning cached EMR content")
                return self.cache[cache_key]
            
            language_instruction = ""
            if language != "en":
                language_instruction = f" Field names in English, values in {self.languages.get(language, 'English')}."
            
            system_prompt = (
                "Extract structured medical information and return a valid JSON object with these fields:\n"
                "- patientName: Patient's name (if mentioned)\n"
                "- age: Patient's age in years\n"
                "- gender: Patient's gender\n"
                "- chiefComplaint: Primary reason for visit\n"
                "- symptoms: Array of symptoms\n"
                "- duration: Duration of symptoms\n"
                "- pastMedicalHistory: Array of past conditions\n"
                "- currentMedications: Array of current medications\n"
                "- allergies: Array of allergies\n"
                "- vitalSigns: Object with vital signs\n"
                "- assessment: Diagnostic impression\n"
                "- plan: Array of treatment plan elements\n"
                "- recommendedTests: Array of suggested tests\n"
                "- followUp: Follow-up instructions\n"
                "- prescriptions: Array of medications with dosage\n\n"
                "Include only explicitly mentioned information. Use null for missing data. "
                "Return ONLY valid JSON, no additional text."
                + language_instruction
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription}
            ]
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.llm_model,
                temperature=0.1
            )
            
            response_content = chat_completion.choices[0].message.content
            
            # Extract and validate JSON
            json_content = self._extract_json(response_content)
            
            try:
                json_obj = json.loads(json_content)
                result = json.dumps(json_obj, indent=2)
                self.cache[cache_key] = result
                return result
            except json.JSONDecodeError:
                # Fallback response
                fallback = json.dumps({
                    "content": response_content,
                    "note": "Response could not be parsed as JSON",
                    "timestamp": int(time.time())
                }, indent=2)
                self.cache[cache_key] = fallback
                return fallback
                
        except Exception as e:
            logging.error(f"EMR content generation error: {e}")
            return json.dumps({"error": f"Error generating EMR content: {str(e)}"})

    def _extract_json(self, response_content):
        """Extract JSON from response content"""
        if '```json' in response_content:
            return response_content.split('```json')[1].split('```')[0].strip()
        elif '```' in response_content:
            return response_content.split('```')[1].split('```')[0].strip()
        return response_content.strip()

    def generate_prescription(self, medical_notes, language="en"):
        """
        Generate prescription based on medical notes.
        
        Args:
            medical_notes (str): Medical notes text
            language (str): Language code
            
        Returns:
            tuple: (prescription_text, explanation_text)
        """
        try:
            if not medical_notes or not medical_notes.strip():
                return "Error: No medical notes provided", "Cannot generate prescription without medical notes"
            
            cache_key = f"rx_{hash(medical_notes)}_{language}"
            
            if cache_key in self.cache:
                logging.info("Returning cached prescription")
                return self.cache[cache_key]
            
            language_instruction = ""
            if language != "en":
                language_instruction = f" Respond in {self.languages.get(language, 'English')}."
            
            prescription_prompt = (
                "Based on these medical notes, generate a prescription and patient explanation:\n\n"
                f"Medical Notes:\n{medical_notes}\n\n"
                "Provide:\n"
                "1. PRESCRIPTION (formatted clearly):\n"
                "   - Medication name\n"
                "   - Dosage with units\n"
                "   - Route of administration\n"
                "   - Frequency and timing\n"
                "   - Duration of treatment\n"
                "   - Special instructions\n\n"
                "2. PATIENT EXPLANATION (after '---' separator):\n"
                "   - Why this medication was chosen\n"
                "   - Expected effects and timeline\n"
                "   - Important side effects to watch for\n"
                "   - When to seek medical attention\n\n"
                "Be medically accurate and use appropriate terminology for prescriptions, "
                "but plain language for patient explanations."
                + language_instruction
            )
            
            messages = [
                {"role": "system", "content": prescription_prompt},
                {"role": "user", "content": "Generate the prescription and explanation."}
            ]
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.llm_model,
                temperature=0.3
            )
            
            full_response = chat_completion.choices[0].message.content
            
            # Parse prescription and explanation
            if "---" in full_response:
                parts = full_response.split("---", 1)
                prescription_text = parts[0].strip()
                explanation_text = parts[1].strip()
            else:
                # Try alternative separators
                parts = full_response.split('\n\n', 1)
                prescription_text = parts[0].strip()
                explanation_text = parts[1].strip() if len(parts) > 1 else "No explanation generated."
            
            result = (prescription_text, explanation_text)
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logging.error(f"Prescription generation error: {e}")
            return f"Error generating prescription: {str(e)}", "Error generating explanation."

    def text_to_speech_with_gtts(self, input_text, output_filepath, language="en"):
        """
        Convert text to speech using Google Text-to-Speech.
        
        Args:
            input_text (str): Text to convert
            output_filepath (str): Output file path
            language (str): Language code
            
        Returns:
            bool: Success status
        """
        try:
            if not input_text or not input_text.strip():
                logging.error("No text provided for TTS")
                return False
            
            # Create temporary file first
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Truncate text if too long
                    if len(input_text) > 5000:
                        input_text = input_text[:5000] + "..."
                    
                    audioobj = gTTS(
                        text=input_text,
                        lang=language,
                        slow=False
                    )
                    audioobj.save(temp_path)
                    
                    # Verify file was created and has content
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Copy to final destination
                        shutil.copy2(temp_path, output_filepath)
                        os.unlink(temp_path)
                        logging.info(f"TTS audio saved to {output_filepath}")
                        return True
                    else:
                        raise Exception("Generated audio file is empty")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"TTS attempt {attempt + 1} failed: {e}")
                        time.sleep(1)
                    else:
                        raise e
                        
        except Exception as e:
            logging.error(f"TTS error: {e}")
            return False
        finally:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        logging.info("Cache cleared")