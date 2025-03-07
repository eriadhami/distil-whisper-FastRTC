"""
Example showing how to integrate distil-whisper-fastrtc with FastRTC.

This is a conceptual example and may need adjustments based on the
actual FastRTC implementation details.
"""

import numpy as np
from distil_whisper_fastrtc import DistilWhisperSTT

def simulate_fastrtc_usage():
    """
    Simulate how FastRTC would use our STT model.
    
    This is a conceptual example - in a real scenario, FastRTC would
    instantiate and use our model according to its own architecture.
    """
    # Create our STT model
    whisper_model = DistilWhisperSTT(
        model="distil-whisper/distil-small.en",
        device="cpu"
    )
    
    # Simulate audio data that would come from FastRTC
    # In reality, this would be captured from a microphone or received over WebRTC
    sample_rate = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a simple audio signal (in real usage, this would be actual speech)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # FastRTC would pass the audio data to our model's stt method
    transcription = whisper_model.stt((sample_rate, audio_data))
    
    print(f"Transcription: {transcription}")
    
    # In FastRTC, this text would then be used for further processing,
    # such as displaying in a UI or sending to other components

if __name__ == "__main__":
    print("Simulating FastRTC integration with distil-whisper-fastrtc")
    simulate_fastrtc_usage()
    print("Note: This is a simulation. In a real FastRTC application, the audio")
    print("would come from a microphone or WebRTC connection.")
