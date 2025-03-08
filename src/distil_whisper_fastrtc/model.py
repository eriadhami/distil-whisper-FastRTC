from typing import Literal, Optional, Protocol, Tuple, Union
import os
import torch
import numpy as np
from numpy.typing import NDArray
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class DistilWhisperSTT:
    """
    A Speech-to-Text model using Hugging Face's distil-whisper model.
    Implements the FastRTC STTModel protocol.

    Attributes:
        model_id: The Hugging Face model ID
        device: The device to run inference on ('cpu', 'cuda', 'mps')
        dtype: Data type for model weights (float16, float32)
    """

    MODEL_OPTIONS = Literal[
        "distil-whisper/distil-small.en",
        "distil-whisper/distil-medium.en",
        "distil-whisper/distil-large-v2",
        "distil-whisper/distil-large-v3",
    ]

    def __init__(
        self,
        model: MODEL_OPTIONS = "distil-whisper/distil-small.en",
        device: Optional[str] = None,
        dtype: Literal["float16", "float32"] = "float16",
    ):
        """
        Initialize the Distil-Whisper STT model.

        Args:
            model: Model size/variant to use
            device: Device to use for inference (auto-detected if None)
            dtype: Model precision (float16 recommended for faster inference)
        """
        self.model_id = model

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if dtype == "float16" else torch.float32

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the model and processor from Hugging Face."""
        torch_dtype = self.dtype

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model = self.model.to(self.device)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            device=self.device,
        )

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """
        Transcribe audio to text using distil-whisper.

        Args:
            audio: Tuple of (sample_rate, audio_data)
                  where audio_data is a numpy array of int16 or float32

        Returns:
            Transcribed text as string
        """
        sample_rate, audio_np = audio
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Handle different audio formats
        if audio_np.dtype == np.int16:
            # Convert int16 to float32 and normalize to [-1, 1]
            audio_np = audio_np.astype(np.float32) / 32768.0

        # Run transcription
        result = self.pipe(
            {"sampling_rate": sample_rate, "array": audio_np},
        )

        return result["text"].strip()


# For simpler imports
def get_stt_model(
    model_name: str = "distil-whisper/distil-small.en", **kwargs
) -> STTModel:
    """
    Helper function to easily get an STT model instance.
    Warms up the model with a small array of zeros to improve first inference time.

    Args:
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        A warmed-up STTModel instance
    """
    model = DistilWhisperSTT(model=model_name, **kwargs)
    
    # Warm up the model with a small array of zeros
    sample_rate = 16000  # Standard sample rate
    duration = 0.1  # Very short duration for warm-up
    zeros = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Run inference on zeros array to warm up the model
    _ = model.stt((sample_rate, zeros))
    
    return model
