import numpy as np
import pytest
from distil_whisper_fastrtc import DistilWhisperSTT, get_stt_model

def test_model_initialization():
    """Test that the model can be initialized."""
    # This test doesn't actually load the model to save time
    # It just checks that the class can be instantiated
    model = DistilWhisperSTT(device="cpu")
    assert model.device == "cpu"
    assert model.model_id == "distil-whisper/distil-small.en"

def test_helper_function():
    """Test the helper function."""
    model = get_stt_model(device="cpu")
    assert isinstance(model, DistilWhisperSTT)

@pytest.mark.skip(reason="Requires downloading model weights")
def test_transcription():
    """Test transcription with a simple audio sample."""
    # Create a simple sine wave as test audio
    sample_rate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Initialize model
    model = get_stt_model(device="cpu")
    
    # Run transcription
    result = model.stt((sample_rate, audio))
    
    # We don't expect meaningful transcription from a sine wave,
    # but the function should run without errors
    assert isinstance(result, str)
