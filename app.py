from assistant_core import HealthcareAIAssistant
import os
import time
import tempfile
import streamlit as st
import logging
import json
import threading
import sounddevice as sd
import numpy as np
import wave
from pathlib import Path
import asyncio
from io import BytesIO
import queue
import traceback

# Suppress pydub warnings about ffmpeg
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Compatibility function for different Streamlit versions
def rerun():
    """Compatible rerun function for different Streamlit versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually")

# Check system requirements
def check_system_requirements():
    """Check if required system components are available"""
    requirements = {
        'ffmpeg': False,
        'microphone': False,
        'audio_devices': []
    }
    
    # Check ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            requirements['ffmpeg'] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check microphone and audio devices
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                requirements['microphone'] = True
                input_devices.append(f"{i}: {device['name']} (channels: {device['max_input_channels']})")
        requirements['audio_devices'] = input_devices
    except Exception as e:
        logger.error(f"Error checking audio devices: {e}")
    
    return requirements

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .status-recording {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
        animation: pulse 2s infinite;
    }
    
    .status-ready {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
    
    .status-processing {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 2px solid #ff9800;
    }
    
    .status-error {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #f44336;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .tab-content {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .recording-controls {
        display: flex;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .audio-level {
        width: 100%;
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .audio-level-fill {
        height: 100%;
        background: linear-gradient(90deg, #4caf50 0%, #ff9800 70%, #f44336 100%);
        transition: width 0.1s ease;
    }
</style>
""", unsafe_allow_html=True)

class ImprovedAudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.audio_queue = queue.Queue()
        self.level_callback = None
        self.error_callback = None
        self.lock = threading.Lock()
        self.current_level = 0.0
        self.last_error = None
        
    def set_callbacks(self, level_callback=None, errorCallback=None):
        """Set callbacks for audio level and error handling"""
        self.level_callback = levelCallback
        self.error_callback = errorCallback
        
    def get_available_devices(self):
        """Get list of available input devices"""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            return input_devices
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return []
    
    def test_device(self, device_index=None):
        """Test if audio device is working"""
        try:
            # Try to create a temporary stream
            test_stream = sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=1024,
                dtype=np.float32
            )
            test_stream.start()
            time.sleep(0.1)  # Brief test
            test_stream.stop()
            test_stream.close()
            return True
        except Exception as e:
            logger.error(f"Device test failed: {e}")
            return False
    
    def get_current_level(self):
        """Get current audio level (thread-safe)"""
        with self.lock:
            return self.current_level
    
    def get_last_error(self):
        """Get last error message (thread-safe)"""
        with self.lock:
            return self.last_error
    
    def start_recording(self):
        """Start recording audio with improved error handling"""
        with self.lock:
            if self.recording:
                return False, "Already recording"
            
            self.recording = True
            self.audio_data = []
            self.last_error = None
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                    # Store error in instance variable instead of calling callback directly
                    with self.lock:
                        self.last_error = f"Audio status: {status}"
                
                if self.recording:
                    # Store audio data
                    self.audio_data.append(indata.copy())
                    
                    # Calculate and store audio level
                    level = float(np.sqrt(np.mean(indata**2)))
                    with self.lock:
                        self.current_level = level
            
            try:
                # Test device first
                if not self.test_device(self.device):
                    return False, "Audio device test failed"
                
                self.stream = sd.InputStream(
                    device=self.device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=audio_callback,
                    dtype=np.float32,
                    blocksize=1024
                )
                self.stream.start()
                logger.info(f"Recording started with device: {self.device}")
                return True, "Recording started successfully"
                
            except Exception as e:
                self.recording = False
                error_msg = f"Error starting recording: {str(e)}"
                logger.error(error_msg)
                with self.lock:
                    self.last_error = error_msg
                return False, error_msg
    
    def stop_recording(self):
        """Stop recording and return audio file path"""
        with self.lock:
            if not self.recording:
                return None, "Not recording"
            
            self.recording = False
            
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                
                if not self.audio_data:
                    return None, "No audio data recorded"
                
                # Convert to numpy array
                audio_array = np.concatenate(self.audio_data, axis=0)
                
                # Check if audio is too short
                duration = len(audio_array) / self.sample_rate
                if duration < 0.5:  # Less than 0.5 seconds
                    return None, f"Recording too short: {duration:.2f}s"
                
                # Convert to 16-bit PCM
                audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_path = temp_file.name
                temp_file.close()
                
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                
                logger.info(f"Audio saved to {temp_path}, duration: {duration:.2f}s")
                return temp_path, f"Recording saved successfully ({duration:.2f}s)"
                
            except Exception as e:
                error_msg = f"Error stopping recording: {str(e)}"
                logger.error(error_msg)
                return None, error_msg
    
    def is_recording(self):
        """Check if currently recording"""
        return self.recording
    
    def cleanup(self):
        """Clean up resources"""
        with self.lock:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            self.recording = False

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'assistant' not in st.session_state:
        try:
            st.session_state.assistant = HealthcareAIAssistant()
            st.session_state.assistant_loaded = True
        except Exception as e:
            st.session_state.assistant_loaded = False
            st.session_state.assistant_error = str(e)
    
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = ImprovedAudioRecorder()
    
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = False
        
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = False
    
    if 'audio_level' not in st.session_state:
        st.session_state.audio_level = 0.0
    
    if 'recording_error' not in st.session_state:
        st.session_state.recording_error = None
    
    if 'processed_count' not in st.session_state:
        st.session_state.processed_count = 0

def display_status(message, status_type="ready"):
    """Display status message with styling"""
    status_classes = {
        "ready": "status-ready",
        "recording": "status-recording",
        "processing": "status-processing",
        "error": "status-error"
    }
    
    st.markdown(f"""
    <div class="{status_classes.get(status_type, 'status-ready')} status-box">
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_audio_level(level):
    """Display audio level visualization"""
    level_percent = min(level * 100, 100)
    st.markdown(f"""
    <div class="audio-level">
        <div class="audio-level-fill" style="width: {level_percent}%"></div>
    </div>
    <small>Audio Level: {level_percent:.1f}%</small>
    """, unsafe_allow_html=True)

def process_audio_file(audio_file, language):
    """Process audio file and return all outputs"""
    try:
        if not audio_file:
            return None, "No audio file provided", "", "{}", "", "", None
        
        # Save uploaded file to temporary location
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        
        # Handle different input types
        if hasattr(audio_file, 'getbuffer'):
            # Streamlit UploadedFile
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_file.getbuffer())
        elif hasattr(audio_file, 'read'):
            # BytesIO or file-like object
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_file.read())
        else:
            # String path
            import shutil
            shutil.copy(audio_file, temp_audio_path)
        
        assistant = st.session_state.assistant
        
        # Step 1: Transcribe
        with st.spinner("Transcribing audio..."):
            transcription = assistant.transcribe_audio(temp_audio_path, language)
            if transcription.startswith("Error"):
                return temp_audio_path, transcription, "", "{}", "", "", None
        
        # Step 2: Generate medical notes
        with st.spinner("Generating medical notes..."):
            medical_notes = assistant.analyze_text(transcription, language)
        
        # Step 3: Generate EMR content
        with st.spinner("Generating EMR data..."):
            emr_content = assistant.generate_emr_content(transcription, language)
        
        # Step 4: Generate prescription
        with st.spinner("Generating prescription..."):
            prescription_text, prescription_explanation = assistant.generate_prescription(medical_notes, language)
        
        # Step 5: Generate TTS
        with st.spinner("Generating audio summary..."):
            tts_output_path = os.path.join(tempfile.gettempdir(), f"medical_audio_{int(time.time())}.mp3")
            notes_and_explanation = medical_notes + "\n\n" + prescription_explanation
            
            tts_success = assistant.text_to_speech_with_gtts(notes_and_explanation, tts_output_path, language)
            final_audio = tts_output_path if tts_success else None
        
        # Clean up temporary file
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        return temp_audio_path, transcription, medical_notes, emr_content, prescription_text, prescription_explanation, final_audio
        
    except Exception as e:
        logger.error(f"Process audio error: {e}")
        logger.error(traceback.format_exc())
        return None, f"Error processing audio: {str(e)}", "", "{}", "", "", None

def process_text_input(text_input, language):
    """Process text input and return all outputs"""
    try:
        if not text_input or not text_input.strip():
            return "", "", "{}", "", "", None
        
        assistant = st.session_state.assistant
        
        # Step 1: Generate medical notes
        with st.spinner("Generating medical notes..."):
            medical_notes = assistant.analyze_text(text_input, language)
        
        # Step 2: Generate EMR content
        with st.spinner("Generating EMR data..."):
            emr_content = assistant.generate_emr_content(text_input, language)
        
        # Step 3: Generate prescription
        with st.spinner("Generating prescription..."):
            prescription_text, prescription_explanation = assistant.generate_prescription(medical_notes, language)
        
        # Step 4: Generate TTS
        with st.spinner("Generating audio summary..."):
            tts_output_path = os.path.join(tempfile.gettempdir(), f"text_audio_{int(time.time())}.mp3")
            notes_and_explanation = medical_notes + "\n\n" + prescription_explanation
            
            tts_success = assistant.text_to_speech_with_gtts(notes_and_explanation, tts_output_path, language)
            final_audio = tts_output_path if tts_success else None
        
        return medical_notes, emr_content, prescription_text, prescription_explanation, final_audio
        
    except Exception as e:
        logger.error(f"Process text error: {e}")
        logger.error(traceback.format_exc())
        return f"Error processing text: {str(e)}", "{}", "", "", None

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare AI Assistant</h1>
        <p>Transform patient conversations into structured medical documentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check system requirements
    requirements = check_system_requirements()
    
    # Show system requirements status
    if not requirements['ffmpeg']:
        st.warning("""
        ⚠️ **FFmpeg not found**: Audio processing may be limited. 
        
        **To install FFmpeg:**
        - **Windows**: Download from https://ffmpeg.org/download.html or use `winget install ffmpeg`
        - **macOS**: `brew install ffmpeg`
        - **Linux**: `sudo apt install ffmpeg` or `sudo yum install ffmpeg`
        """)
    
    if not requirements['microphone']:
        st.error("🎙️ **No microphone detected**: Please check your audio devices and permissions.")
        st.info("You can still upload audio files for processing.")
    
    # Check if assistant is loaded
    if not st.session_state.assistant_loaded:
        st.error(f"Failed to initialize assistant: {st.session_state.assistant_error}")
        st.markdown("Please check your API keys and try again.")
        return
    
    # Get language choices
    language_choices = list(st.session_state.assistant.languages.items())
    language_options = [f"{name} ({code})" for code, name in language_choices]
    language_mapping = {f"{name} ({code})": code for code, name in language_choices}
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        selected_language_display = st.selectbox(
            "Select Language",
            options=language_options,
            index=0
        )
        selected_language = language_mapping[selected_language_display]
        
        # Audio device selection
        if requirements['microphone']:
            st.subheader("🎙️ Audio Device")
            devices = st.session_state.audio_recorder.get_available_devices()
            if devices:
                device_options = ["Default"] + [f"{d['name']} ({d['index']})" for d in devices]
                selected_device = st.selectbox("Select Audio Device", device_options)
                
                if selected_device != "Default":
                    device_index = int(selected_device.split('(')[-1].split(')')[0])
                    st.session_state.audio_recorder.device = device_index
                else:
                    st.session_state.audio_recorder.device = None
        
        st.header("📊 Statistics")
        st.metric("Processed Files", st.session_state.processed_count)
        
        # Display audio devices info
        if requirements['audio_devices']:
            st.subheader("🔊 Available Audio Devices")
            for device in requirements['audio_devices']:
                st.text(device)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["🎙️ Voice Input", "📝 Text Input"])
    
    # Add columns for Voice Input tab (fix for col2 error)
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Audio Recording")
            
            # Recording status
            if st.session_state.audio_recorder.is_recording():
                display_status("Recording in progress...", "recording")
            else:
                display_status("Ready to record", "ready")
            
            # Audio level indicator
            audio_level = st.session_state.audio_level
            display_audio_level(audio_level)
            
            # Record button
            if st.button("🎙️ Start Recording", key="start_recording", use_container_width=True):
                st.session_state.recording_state = True
                st.session_state.processing_state = False
                
                success, message = st.session_state.audio_recorder.start_recording()
                if success:
                    st.success(message)
                else:
                    st.error(message)
            
            # Stop button
            if st.button("⏹️ Stop Recording", key="stop_recording", use_container_width=True):
                st.session_state.recording_state = False
                
                audio_path, message = st.session_state.audio_recorder.stop_recording()
                if audio_path:
                    st.success(f"Recording saved: {audio_path}")
                    
                    # Set current audio source for processing
                    st.session_state.current_audio_source = audio_path
                    
                    # Auto-process after recording
                    st.session_state.processing_state = True
                    rerun()
                else:
                    st.error(message)
        
        with col2:
            st.subheader("Results")
            
            # Process audio if requested
            if st.session_state.processing_state:
                audio_source = st.session_state.current_audio_source
                
                if audio_source:
                    display_status("🔄 Processing audio...", "processing")
                    
                    try:
                        _, transcription, medical_notes, emr_content, prescription_text, prescription_explanation, final_audio = process_audio_file(
                            audio_source, selected_language
                        )
                        
                        # Store results in session state
                        st.session_state.transcription = transcription
                        st.session_state.medical_notes = medical_notes
                        st.session_state.emr_content = emr_content
                        st.session_state.prescription_text = prescription_text
                        st.session_state.prescription_explanation = prescription_explanation
                        st.session_state.final_audio = final_audio
                        
                        # Update processed count
                        st.session_state.processed_count += 1
                        
                        st.success("Audio processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
                        logger.error(f"Audio processing error: {e}")
                        logger.error(traceback.format_exc())
                    
                    st.session_state.processing_state = False
                    rerun()
            
            # Display results if available
            if hasattr(st.session_state, 'transcription') and st.session_state.transcription:
                st.text_area("Transcription", st.session_state.transcription, height=100)
                
                # Create tabs for different outputs
                result_tabs = st.tabs(["📋 Medical Notes", "💊 Prescription", "📊 EMR Data"])
                
                with result_tabs[0]:
                    st.text_area("Clinical Notes", st.session_state.medical_notes, height=300)
                
                with result_tabs[1]:
                    st.text_area("Prescription", st.session_state.prescription_text, height=150)
                    st.text_area("Patient Explanation", st.session_state.prescription_explanation, height=200)
                
                with result_tabs[2]:
                    try:
                        emr_data = json.loads(st.session_state.emr_content)
                        st.json(emr_data)
                    except:
                        st.text_area("EMR Data", st.session_state.emr_content, height=200)
                
                # Audio output
                if st.session_state.final_audio and os.path.exists(st.session_state.final_audio):
                    st.audio(st.session_state.final_audio, format='audio/mp3')
                    
                    # Download button for audio
                    with open(st.session_state.final_audio, 'rb') as f:
                        st.download_button(
                            label="📥 Download Audio Summary",
                            data=f.read(),
                            file_name=f"medical_summary_{int(time.time())}.mp3",
                            mime="audio/mp3"
                        )
    
    with tab2:
        st.header("Text Input Processing")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Patient Description")
            
            text_input = st.text_area(
                "Enter patient symptoms, medical history, and current condition",
                height=300,
                key="text_input"
            )
            
            if st.button("🔄 Generate Medical Notes", key="process_text", use_container_width=True):
                if text_input.strip():
                    st.session_state.text_processing_state = True
                    st.session_state.current_text_input = text_input
                    rerun()
                else:
                    st.warning("Please enter some text first")
            
            if st.button("🗑️ Clear All", key="clear_text", use_container_width=True):
                # Clear text input
                st.session_state.text_input = ""
                # Clear all text-related session state
                for key in ['text_medical_notes', 'text_emr_content', 'text_prescription_text', 
                           'text_prescription_explanation', 'text_final_audio']:
                    if key in st.session_state:
                        del st.session_state[key]
                rerun()
        
        with col2:
            st.subheader("Results")
            
            # Process text if requested
            if hasattr(st.session_state, 'text_processing_state') and st.session_state.text_processing_state:
                display_status("🔄 Processing text...", "processing")
                
                try:
                    medical_notes, emr_content, prescription_text, prescription_explanation, final_audio = process_text_input(
                        st.session_state.current_text_input, selected_language
                    )
                    
                    # Store results in session state
                    st.session_state.text_medical_notes = medical_notes
                    st.session_state.text_emr_content = emr_content
                    st.session_state.text_prescription_text = prescription_text
                    st.session_state.text_prescription_explanation = prescription_explanation
                    st.session_state.text_final_audio = final_audio
                    
                    # Update processed count
                    st.session_state.processed_count += 1
                    
                    st.success("Text processed successfully!")
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    logger.error(f"Text processing error: {e}")
                    logger.error(traceback.format_exc())
                
                st.session_state.text_processing_state = False
                rerun()
            
            # Display results if available
            if hasattr(st.session_state, 'text_medical_notes') and st.session_state.text_medical_notes:
                # Create tabs for different outputs
                result_tabs = st.tabs(["📋 Medical Notes", "💊 Prescription", "📊 EMR Data"])
                
                with result_tabs[0]:
                    st.text_area("Clinical Notes", st.session_state.text_medical_notes, height=300, key="text_notes_display")
                
                with result_tabs[1]:
                    st.text_area("Prescription", st.session_state.text_prescription_text, height=150, key="text_prescription_display")
                    st.text_area("Patient Explanation", st.session_state.text_prescription_explanation, height=200, key="text_explanation_display")
                
                with result_tabs[2]:
                    try:
                        emr_data = json.loads(st.session_state.text_emr_content)
                        st.json(emr_data)
                    except:
                        st.text_area("EMR Data", st.session_state.text_emr_content, height=200, key="text_emr_display")
                
                # Audio output
                if st.session_state.text_final_audio and os.path.exists(st.session_state.text_final_audio):
                    st.audio(st.session_state.text_final_audio, format='audio/mp3')
                    
                    # Download button for audio
                    with open(st.session_state.text_final_audio, 'rb') as f:
                        st.download_button(
                            label="📥 Download Audio Summary",
                            data=f.read(),
                            file_name=f"text_medical_summary_{int(time.time())}.mp3",
                            mime="audio/mp3",
                            key="text_audio_download"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>🏥 Healthcare AI Assistant - Professional Medical Documentation</p>
        <p><small>Always consult with qualified healthcare professionals for medical decisions</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-cleanup old files
    cleanup_old_files()

def cleanup_old_files():
    """Clean up old temporary files to prevent disk space issues"""
    try:
        temp_dir = tempfile.gettempdir()
        current_time = time.time()
        
        # Find and delete files older than 1 hour
        for filename in os.listdir(temp_dir):
            if filename.startswith(('medical_audio_', 'text_audio_')) and filename.endswith('.mp3'):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.getctime(file_path) < current_time - 3600:  # 1 hour
                        os.unlink(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                except:
                    pass  # Ignore errors during cleanup
                    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# Error handling for the main app
def safe_main():
    """Main function with error handling"""
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {e}")
        logger.error(traceback.format_exc())
        
        # Show error details in expandable section
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        
        # Recovery options
        st.markdown("### Recovery Options:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart Application"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                rerun()
        
        with col2:
            if st.button("🧹 Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared")
        
        with col3:
            if st.button("📋 Copy Error Log"):
                error_log = traceback.format_exc()
                st.code(error_log)
                st.info("Error log displayed above - you can copy it manually")

# Application entry point
if __name__ == "__main__":
    # Set up error handling
    try:
        safe_main()
    except KeyboardInterrupt:
        st.info("Application interrupted by user")
    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        logger.critical(f"Fatal application error: {e}")
        
        # Last resort error display
        st.markdown("### Fatal Error Occurred")
        st.markdown("The application encountered a critical error and needs to be restarted.")
        st.code(str(e))
        
        if st.button("🔄 Force Restart"):
            # Clear everything and restart
            st.session_state.clear()
            st.cache_data.clear()
            rerun()