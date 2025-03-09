from typing import Literal, Optional, Protocol, Tuple, Union
from functools import lru_cache
import os
import sys
from pathlib import Path
import click
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
@lru_cache
def get_stt_model(
    model_name: str = "distil-whisper/distil-small.en", 
    verbose: bool = True,
    **kwargs
) -> STTModel:
    """
<<<<<<< HEAD
    Helper function to easily get an STT model instance with warm-up.
=======
    Helper function to easily get an STT model instance.
    Warms up the model with a small array of zeros to improve first inference time.
>>>>>>> e108ba4e8fb28cfe0802e9a9b918416aa87bc247

    Args:
        model_name: Name of the model to use
        verbose: Whether to print status messages
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        A warmed-up STTModel instance
    """
<<<<<<< HEAD
    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create the model - remove verbose from kwargs to avoid TypeError
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
    m = DistilWhisperSTT(model=model_name, **filtered_kwargs)
    
    # Warm up the model
    curr_dir = Path(__file__).parent
    
    # Load test audio file for warm-up
    import numpy as np
    sample_rate = 16000
    
    # Try to load test_file.wav if it exists, otherwise use silence
    test_file_path = curr_dir / "test_file.wav"
    if test_file_path.exists():
        # Simple load of the test file
        # This is a placeholder - in a real implementation, you'd use a proper audio loading function
        audio = np.zeros(sample_rate, dtype=np.float32)  # Placeholder
    else:
        # Create a simple silence array for warm-up
        audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
    
    # Print only to stderr with green styling
    if verbose:
        msg = click.style("INFO", fg="green") + ":\t  Warming up STT model.\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
    
    # Warm up the model
    m.stt((sample_rate, audio))
    
    if verbose:
        msg = click.style("INFO", fg="green") + ":\t  STT model warmed up.\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
    
    return m
=======
    model = DistilWhisperSTT(model=model_name, **kwargs)
    
    # Warm up the model with a small array of zeros
    sample_rate = 16000  # Standard sample rate
    duration = 0.1  # Very short duration for warm-up
    zeros = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Run inference on zeros array to warm up the model
    _ = model.stt((sample_rate, zeros))
    
    return model
>>>>>>> e108ba4e8fb28cfe0802e9a9b918416aa87bc247
