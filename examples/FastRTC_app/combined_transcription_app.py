"""
Combined Transcription App

This application combines both real-time and file-based transcription using
distil-whisper-FastRTC and FastRTC.
"""

import gradio as gr
import numpy as np
from fastrtc import Stream, StreamHandler
from distil_whisper_fastrtc import get_stt_model, load_audio

# Initialize the STT model (shared between both interfaces)
print("Loading STT model (this may take a moment the first time)...")
model = get_stt_model(device="cpu")  # Use "cpu" since CUDA is not available

# File transcription function
def transcribe_file(audio_file):
    """
    Transcribe an audio file using the distil-whisper-FastRTC model.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Transcription text
    """
    if audio_file is None:
        return "No audio file provided"
    
    print(f"Transcribing file: {audio_file}")
    
    # Load the audio file
    audio = load_audio(audio_file)
    
    # Transcribe the audio
    transcription = model.stt(audio)
    
    print(f"Transcription: {transcription}")
    return transcription

# Create a custom stream handler for real-time transcription
class TranscriptionHandler(StreamHandler):
    def __init__(self, stt_model):
        super().__init__()
        self.stt_model = stt_model
        self.transcription_history = []
        self.full_transcription = ""
        self.buffer_size = 5  # Number of audio chunks to accumulate before transcribing
        self.audio_buffer = []
        
    def receive(self, audio):
        # Add the audio to the buffer
        sample_rate, audio_data = audio
        
        # Ensure audio is single channel
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono by averaging channels
        
        # Check if the audio exceeds the noise threshold
        noise_threshold = 0.1  # Further increased threshold to filter out more background noise
        audio_amplitude = np.abs(audio_data).mean()
        
        if audio_amplitude > noise_threshold:
            print(f"Audio amplitude: {audio_amplitude:.6f} - Above threshold")
            self.audio_buffer.append(audio_data)
        else:
            print(f"Audio amplitude: {audio_amplitude:.6f} - Below threshold (skipping)")
            # Clear the buffer if we're just getting noise
            if len(self.audio_buffer) > 0:
                self.audio_buffer = []
            return
        
        # If we have enough audio chunks, transcribe them
        if len(self.audio_buffer) >= self.buffer_size:
            # Concatenate the audio chunks
            print("Accumulated enough audio, transcribing...")
            combined_audio = np.concatenate(self.audio_buffer)
            
            # Ensure the combined audio is single channel and float32
            if combined_audio.ndim > 1:
                combined_audio = combined_audio.mean(axis=1)
            
            # Ensure float32 format
            if combined_audio.dtype != np.float32:
                combined_audio = combined_audio.astype(np.float32)
            
            # Double-check if the combined audio is still above the threshold
            combined_amplitude = np.abs(combined_audio).mean()
            if combined_amplitude <= noise_threshold:
                print(f"Combined audio amplitude too low: {combined_amplitude:.6f} - Skipping transcription")
                self.audio_buffer = []
                return
            
            # Transcribe the combined audio
            try:
                text = self.stt_model.stt((sample_rate, combined_audio))
                print(f"Transcribed: {text}")
            except Exception as e:
                print(f"Transcription error: {e}")
                text = ""
            
            # Filter out common noise transcriptions
            noise_words = ["you", "you know", "um", "uh", "ah", "hmm", "mm", "oh", "eh", "huh", "ha", "yeah", "yes", "no", "okay", "ok"]
            is_noise = text.strip().lower() in noise_words
            
            # Update the full transcription if it's not noise
            if text.strip() and not is_noise:  # Only add non-empty, non-noise transcriptions
                self.full_transcription += " " + text
                self.full_transcription = self.full_transcription.strip()
                
                # Add to history
                self.transcription_history.append(text)
                
                # Send the transcription back to the client
                print(f"Transcribed: {text}")
                print(f"Full transcription: {self.full_transcription}")
                self.send_message_sync({
                    "transcription": text,
                    "full_transcription": self.full_transcription,
                    "history": self.transcription_history
                })
            elif is_noise:
                print(f"Filtered out noise transcription: '{text}'")
            
            # Clear the buffer
            self.audio_buffer = []
    
    def copy(self):
        # Create a copy of this handler
        return TranscriptionHandler(self.stt_model)
    
    def emit(self):
        # This method is required but not used in our case
        pass

# Create a stream with our handler for real-time transcription
stream = Stream(
    TranscriptionHandler(model),
    modality="audio",  # Audio modality for speech input
    mode="send-receive",  # Allow bidirectional communication
)

# Configure the UI to allow device selection
stream.ui.allow_device_selection = True
stream.ui.sample_rate = 16000  # 16kHz is standard for speech recognition

# Add a title and description to the stream UI
stream.ui.title = "Real-time Speech Transcription"
stream.ui.description = """
# Real-time Speech Transcription with distil-whisper-FastRTC

This app uses the distil-whisper-FastRTC model to transcribe your speech in real-time.

## Instructions:
1. Select your microphone from the dropdown menu
2. Click the microphone button to start recording
3. Speak clearly into your microphone
4. Your speech will be transcribed in real-time
5. Click the stop button to end recording

## Troubleshooting:
- If you're getting "you know" transcriptions when not speaking, try increasing your speaking volume
- Make sure you're in a quiet environment with minimal background noise
- Try selecting a different microphone if available
"""

# Add a custom UI component to display the full transcription
with stream.ui:
    with gr.Blocks():
        with gr.Row():
            # Add a text area to display the full transcription
            full_transcription = gr.Textbox(
                label="Full Transcription", 
                placeholder="Transcription will appear here...",
                lines=5,
                interactive=False
            )
            
            # Add a button to clear the transcription
            clear_button = gr.Button("Clear Transcription")
            
            # Define a function to clear the transcription
            def clear_transcription():
                return ""
            
            # Connect the button to the function
            clear_button.click(fn=clear_transcription, outputs=full_transcription)

# Create a Gradio interface for file transcription
file_interface = gr.Interface(
    fn=transcribe_file,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Textbox(label="Transcription"),
    title="Audio File Transcription",
    description="""
    # Audio File Transcription with distil-whisper-FastRTC
    
    Upload an audio file to transcribe it using the distil-whisper-FastRTC model.
    
    Supported formats: WAV, MP3, OGG, FLAC, etc.
    """
)

# Create a combined app with tabs
demo = gr.TabbedInterface(
    [stream.ui, file_interface],
    ["Real-time Transcription", "File Transcription"],
    title="Speech-to-Text Transcription App",
    theme=gr.themes.Soft()
)

# Launch the app
if __name__ == "__main__":
    print("Starting Combined Transcription App")
    print("The app will open in your browser.")
    demo.launch(server_port=7860)
