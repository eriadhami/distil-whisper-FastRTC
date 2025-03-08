"""
File Transcription App

This application uses distil-whisper-FastRTC to transcribe audio from a file.
It demonstrates how to use the model with pre-recorded audio files.
"""

from distil_whisper_fastrtc import get_stt_model, load_audio
import gradio as gr

# Initialize the STT model
print("Loading STT model (this may take a moment the first time)...")
model = get_stt_model(device="cpu")  # Use "cuda" if GPU is available

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

# Create a Gradio interface
demo = gr.Interface(
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

# Launch the app
if __name__ == "__main__":
    print("Starting File Transcription App")
    print("The app will open in your browser. Upload an audio file to transcribe it.")
    demo.launch(server_port=7861)  # Using a different port than the real-time app
