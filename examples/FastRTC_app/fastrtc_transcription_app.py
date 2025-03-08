"""
FastRTC Transcription App

This application uses FastRTC and distil-whisper-FastRTC to create a real-time
speech-to-text transcription app with a Gradio UI.
"""

from fastrtc import Stream, StreamHandler
from distil_whisper_fastrtc import get_stt_model
import numpy as np

# Initialize the STT model
print("Loading STT model (this may take a moment the first time)...")
model = get_stt_model(device="cpu")  # Use "cuda" if GPU is available

# Create a custom stream handler for transcription
class TranscriptionHandler(StreamHandler):
    def __init__(self, stt_model):
        super().__init__()
        self.stt_model = stt_model
        self.transcription_history = []
        
    def receive(self, audio):
        # Transcribe the audio
        print("Received audio, transcribing...")
        text = self.stt_model.stt(audio)
        
        # Add to history
        self.transcription_history.append(text)
        
        # Send the transcription back to the client
        print(f"Transcribed: {text}")
        self.send_message_sync({
            "transcription": text,
            "history": self.transcription_history
        })
    
    def copy(self):
        # Create a copy of this handler
        return TranscriptionHandler(self.stt_model)
    
    def emit(self):
        # This method is required but not used in our case
        pass

# Create a stream with our handler
stream = Stream(
    TranscriptionHandler(model),
    modality="audio",  # Audio modality for speech input
    mode="send-receive",  # Allow bidirectional communication
)

# Add a title and description to the UI
stream.ui.title = "Real-time Speech Transcription"
stream.ui.description = """
# Real-time Speech Transcription with distil-whisper-FastRTC

This app uses the distil-whisper-FastRTC model to transcribe your speech in real-time.

## Instructions:
1. Click the microphone button to start recording
2. Speak into your microphone
3. Your speech will be transcribed in real-time
4. Click the stop button to end recording
"""

# Launch the UI
if __name__ == "__main__":
    print("Starting FastRTC Transcription App")
    print("The app will open in your browser. Speak into your microphone to see live transcription.")
    stream.ui.launch(
        server_port=7860,
        share=False,  # Set to True if you want to create a public link
        server_name="0.0.0.0"  # Bind to all interfaces
    )
