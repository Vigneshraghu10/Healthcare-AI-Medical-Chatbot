import os
import logging
import platform
import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq
from gtts import gTTS
from elevenlabs import ElevenLabs
import elevenlabs
from dotenv import load_dotenv
import json
import time
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HealthcareAIAssistant:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Load API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Validate API keys
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY. Please check your .env file.")
        if not self.elevenlabs_api_key:
            logging.warning("Missing ELEVENLABS_API_KEY. Premium voice features will be unavailable.")

        # Initialize Groq client with error handling
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
        except TypeError as e:
            if 'proxies' in str(e):
                # Import the required modules to create the client manually
                from groq._client import Groq as GroqOriginal
                from groq._base_client import BaseClient

                # Create a custom initialization that avoids the proxies parameter
                class CustomGroq(GroqOriginal):
                    def __init__(self, api_key=None):
                        BaseClient.__init__(
                            self,
                            api_key=api_key,
                            base_url=None,
                            timeout=120,  # Increased timeout for better reliability
                        )

                self.groq_client = CustomGroq(api_key=self.groq_api_key)
            else:
                # If it's a different error, raise it
                raise

        # Improved model configurations for better accuracy
        self.llm_model = "llama-3.3-70b-versatile"  # Using the most capable Llama model
        self.stt_model = "whisper-large-v3"  # Using the latest Whisper model

        # Cache for API responses to improve performance
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
        Transcribe audio file to text using Groq's Whisper model with enhanced error handling.

        Args:
            audio_filepath (str): Path to the audio file to transcribe.
            language (str): Language code for transcription.

        Returns:
            str: Transcribed text.
        """
        try:
            # Create cache key
            cache_key = f"transcribe_{audio_filepath}_{language}"

            # Check cache first
            if cache_key in self.cache:
                logging.info("Returning cached transcription")
                return self.cache[cache_key]

            # Check if file exists and has content
            if not os.path.exists(audio_filepath):
                return "Error transcribing audio: File does not exist"

            # Check file size
            file_size = os.path.getsize(audio_filepath)
            if file_size == 0:
                return "Error transcribing audio: File is empty"

            logging.info(f"Audio file size: {file_size} bytes")

            # Check if file is too large, compress if needed
            if file_size > 20 * 1024 * 1024:  # 20MB limit
                logging.info("Audio file too large, compressing...")
                compressed_path = self._compress_audio(audio_filepath)
                if compressed_path:
                    audio_filepath = compressed_path

            # Try with retries
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    # Open the file in binary mode
                    with open(audio_filepath, "rb") as audio_file:
                        # Send to Groq API
                        transcription = self.groq_client.audio.transcriptions.create(
                            model=self.stt_model,
                            file=audio_file,
                            language=language
                        )

                    # Cache the successful result
                    result = transcription.text
                    self.cache[cache_key] = result
                    return result

                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Transcription attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return f"Error transcribing audio: {str(e)}"

    def _compress_audio(self, audio_filepath):
        """Compress audio to reduce file size"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()

            audio = AudioSegment.from_file(audio_filepath)
            audio.export(temp_path, format="mp3", bitrate="64k")

            logging.info(
                f"Compressed audio from {os.path.getsize(audio_filepath)} to {os.path.getsize(temp_path)} bytes")
            return temp_path
        except Exception as e:
            logging.error(f"Audio compression error: {e}")
            return None

    def analyze_text(self, text_query, language="en"):
        """
        Process text query with LLM to generate medical notes or analysis with improved prompting.

        Args:
            text_query (str): The text to analyze.
            language (str): Language code for the response.

        Returns:
            str: Generated analysis or medical notes.
        """
        try:
            # Create cache key
            cache_key = f"analyze_{text_query[:50]}_{language}"

            # Check cache first
            if cache_key in self.cache:
                logging.info("Returning cached analysis")
                return self.cache[cache_key]

            language_instruction = ""
            if language != "en":
                language_instruction = f" Respond in {self.languages.get(language, 'English')}."

            # Improved prompt for better medical notes
            system_prompt = (
                    "You are an experienced medical professional creating structured clinical notes from patient descriptions. "
                    "Format your response as a proper medical note with sections for:\n"
                    "1. Chief Complaint: Brief statement of primary concern\n"
                    "2. History of Present Illness: Detailed chronology including onset, duration, severity, aggravating/alleviating factors\n"
                    "3. Past Medical History: Only include if mentioned in the input\n"
                    "4. Assessment: Clear diagnostic impressions with differential diagnoses when appropriate\n"
                    "5. Plan: Specific recommendations for treatment, testing, and follow-up\n\n"
                    "Be concise but comprehensive. Use appropriate medical terminology balanced with clarity. "
                    "If information is missing, do not invent details - note what information would be needed."
                    + language_instruction
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text_query
                }
            ]

            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.llm_model,
                temperature=0.2  # Lower temperature for more consistent, factual responses
            )

            result = chat_completion.choices[0].message.content
            self.cache[cache_key] = result
            return result

        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return f"Error analyzing text: {str(e)}"

    def text_to_speech_with_gtts(self, input_text, output_filepath, language="en"):
        """
        Convert text to speech using Google Text-to-Speech API with improved error handling.

        Args:
            input_text (str): Text to convert to speech.
            output_filepath (str): Path to save the audio file.
            language (str): Language code for speech.

        Returns:
            bool: Success status
        """
        try:
            # Create temp file first to avoid file system issues
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()

            # Create the audio with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    audioobj = gTTS(
                        text=input_text,
                        lang=language,
                        slow=False
                    )
                    audioobj.save(temp_path)

                    # Copy to the final destination
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        import shutil
                        shutil.copy2(temp_path, output_filepath)
                        os.unlink(temp_path)
                        return True
                    else:
                        raise Exception("Generated audio file is empty")

                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"TTS attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(1)
                    else:
                        raise

        except Exception as e:
            logging.error(f"gTTS error: {e}")
            return False

    def text_to_speech_with_elevenlabs(self, input_text, output_filepath):
        """
        Convert text to speech using ElevenLabs API with improved error handling.

        Args:
            input_text (str): Text to convert to speech.
            output_filepath (str): Path to save the audio file.

        Returns:
            bool: Success status
        """
        try:
            if not self.elevenlabs_api_key:
                return False

            client = ElevenLabs(api_key=self.elevenlabs_api_key)

            # Fallback to gTTS if text is too long for ElevenLabs
            if len(input_text) > 5000:
                logging.warning("Text too long for ElevenLabs, using gTTS instead")
                return self.text_to_speech_with_gtts(input_text, output_filepath, "en")

            # Try with retries
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    audio = client.generate(
                        text=input_text,
                        voice="Aria",
                        output_format="mp3_44100_128",  # Higher quality audio
                        model="eleven_turbo_v2"
                    )
                    elevenlabs.save(audio, output_filepath)
                    return True

                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"ElevenLabs attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(2)
                    else:
                        # Fallback to gTTS
                        logging.warning(f"ElevenLabs failed after all retries: {e}, falling back to gTTS")
                        return self.text_to_speech_with_gtts(input_text, output_filepath, "en")

        except Exception as e:
            logging.error(f"ElevenLabs error: {e}")
            # Fallback to gTTS
            return self.text_to_speech_with_gtts(input_text, output_filepath, "en")

    def generate_emr_content(self, transcription, language="en"):
        """
        Process transcription to generate structured content for EMR systems with improved JSON handling.

        Args:
            transcription (str): The transcribed text from patient audio.
            language (str): Language code for the response.

        Returns:
            str: EMR content in JSON format.
        """
        try:
            # Create cache key
            cache_key = f"emr_{transcription[:50]}_{language}"

            # Check cache first
            if cache_key in self.cache:
                logging.info("Returning cached EMR content")
                return self.cache[cache_key]

            language_instruction = ""
            if language != "en":
                language_instruction = f" Return field names in English but values in {self.languages.get(language, 'English')}."

            # Improved prompt for better JSON structure
            messages = [
                {
                    "role": "system",
                    "content": (
                            "You will extract structured medical information from patient consultation text. "
                            "Return a valid, properly formatted JSON object with these fields:\n"
                            "- patientName (string): Patient's full name if mentioned\n"
                            "- age (number): Patient's age in years if mentioned\n"
                            "- gender (string): Patient's gender if mentioned\n"
                            "- chiefComplaint (string): Primary reason for visit\n"
                            "- symptoms (array): List of all symptoms mentioned\n"
                            "- duration (string): How long symptoms have been present\n"
                            "- pastMedicalHistory (array): List of past medical conditions\n"
                            "- currentMedications (array): List of current medications\n"
                            "- allergies (array): List of allergies\n"
                            "- vitalSigns (object): Any mentioned vital signs (BP, HR, temp, etc.)\n"
                            "- assessment (string): Diagnostic impression\n"
                            "- plan (array): Treatment plan elements\n"
                            "- recommendedTests (array): Suggested diagnostic tests\n"
                            "- followUp (string): When patient should return\n"
                            "- prescriptions (array): Recommended medications with dosage\n\n"
                            "ONLY include fields where information is explicitly provided. "
                            "For absent information, include the field with null value or empty array. "
                            "Return ONLY the JSON object, no additional text."
                            + language_instruction
                    )
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ]

            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.llm_model,
                temperature=0.1  # Very low temperature for consistent JSON structure
            )

            response_content = chat_completion.choices[0].message.content

            # Enhanced JSON extraction and validation
            json_content = response_content

            # Extract JSON if wrapped in code blocks
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0].strip()
            elif '```' in response_content:
                json_content = response_content.split('```')[1].split('```')[0].strip()

            # Try to parse the JSON
            try:
                json_obj = json.loads(json_content)
                result = json.dumps(json_obj, indent=2)
                self.cache[cache_key] = result
                return result
            except json.JSONDecodeError:
                # Try to clean up common JSON formatting issues
                import re

                # Replace single quotes with double quotes
                cleaned = json_content.replace("'", "\"")

                # Fix trailing commas in arrays and objects
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*\]', ']', cleaned)

                try:
                    json_obj = json.loads(cleaned)
                    result = json.dumps(json_obj, indent=2)
                    self.cache[cache_key] = result
                    return result
                except:
                    # Last resort - return as plain text in JSON wrapper
                    fallback = json.dumps({
                        "content": response_content,
                        "note": "Response could not be parsed as JSON",
                        "timestamp": time.time()
                    }, indent=2)
                    self.cache[cache_key] = fallback
                    return fallback

        except Exception as e:
            logging.error(f"EMR content generation error: {e}")
            return json.dumps({"error": f"Error generating EMR content: {str(e)}"})

    def generate_prescription(self, medical_notes, language="en"):
        """
        Generate a prescription based on medical notes with improved prompting for better results.

        Args:
            medical_notes (str): Medical notes text.
            language (str): Language code for the response.

        Returns:
            tuple: A tuple containing the prescription text and the doctor's explanation.
        """
        try:
            # Create cache key - shorter for less memory usage
            cache_key = f"rx_{medical_notes[:30]}_{language}"

            # Check cache first
            if cache_key in self.cache:
                logging.info("Returning cached prescription")
                return self.cache[cache_key]

            language_instruction = ""
            if language != "en":
                language_instruction = f" Respond in {self.languages.get(language, 'English')}."

            # Enhanced prescription prompt
            prescription_prompt = (
                    "You are an experienced medical professional generating a prescription based on these medical notes:\n\n"
                    f"'{medical_notes}'\n\n"
                    "First, provide a clear prescription with:\n"
                    "1. Medication name (use appropriate generic or brand names)\n"
                    "2. Precise dosage (with units)\n"
                    "3. Administration route (oral, topical, etc.)\n"
                    "4. Frequency (exact schedule)\n"
                    "5. Duration (specific timeframe)\n"
                    "6. Special instructions (with meals, tapering schedule, etc.)\n\n"
                    "After a clear separator (---), provide a patient-friendly explanation of:\n"
                    "- Why this medication was chosen\n"
                    "- What to expect (onset of action, common side effects)\n"
                    "- Important warnings or precautions\n"
                    "- When to seek immediate medical attention\n\n"
                    "Be concise, accurate, and use appropriate medical terminology in the prescription section, "
                    "but plain language in the explanation section."
                    + language_instruction
            )

            messages = [
                {
                    "role": "system",
                    "content": prescription_prompt
                },
                {
                    "role": "user",
                    "content": "Generate the prescription and explanation based on the notes."
                }
            ]

            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.llm_model,
                temperature=0.3  # Moderate temperature for medical precision with some flexibility
            )

            full_response = chat_completion.choices[0].message.content

            # Improved parsing of prescription and explanation
            if "---" in full_response:
                parts = full_response.split("---", 1)
                prescription_text = parts[0].strip()
                explanation_text = parts[1].strip()
            else:
                # Try to split by double newline if no separator
                parts = full_response.split('\n\n', 1)
                prescription_text = parts[0].strip()
                explanation_text = parts[1].strip() if len(parts) > 1 else "No explanation generated."

            result = (prescription_text, explanation_text)
            self.cache[cache_key] = result
            return result

        except Exception as e:
            logging.error(f"Prescription generation error: {e}")
            return f"Error generating prescription: {str(e)}", "Error generating explanation."