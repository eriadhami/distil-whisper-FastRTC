# Transcription Apps with distil-whisper-FastRTC

This repository contains two applications that demonstrate how to use the distil-whisper-FastRTC model for speech-to-text transcription:

1. **Real-time Transcription App**: Uses FastRTC to create a real-time speech-to-text transcription app with a Gradio UI.
2. **File Transcription App**: Transcribes audio from uploaded files.

## Prerequisites

Before running the applications, make sure you have the required packages installed:

```bash
pip install distil-whisper-fastrtc[audio] fastrtc gradio
```

## Real-time Transcription App

The real-time transcription app uses FastRTC to create a WebRTC-enabled interface for transcribing speech in real-time.

### Features

- Real-time speech-to-text transcription
- Transcription history
- Built-in Gradio UI

### Usage

Run the application with:

```bash
python fastrtc_transcription_app.py
```

This will start a Gradio server on port 7860 and open a browser window. Follow the instructions in the UI to start recording and see the transcriptions.

## File Transcription App

The file transcription app allows you to upload audio files and transcribe them using the distil-whisper-FastRTC model.

### Features

- Upload audio files (WAV, MP3, OGG, FLAC, etc.)
- Transcribe pre-recorded audio
- Simple Gradio UI

### Usage

Run the application with:

```bash
python file_transcription_app.py
```

This will start a Gradio server on port 7861 and open a browser window. Upload an audio file to transcribe it.

## Test File

The repository includes a test audio file (`test_file.wav`) that you can use to test the file transcription app.

## Advanced Configuration

Both applications use the default `distil-whisper/distil-small.en` model, which is optimized for English transcription. You can modify the code to use a different model by changing the `get_stt_model` call:

```python
# Use a larger model for better quality
model = get_stt_model(model_name="distil-whisper/distil-large-v3", device="cuda")
```

Available models:
- `distil-whisper/distil-small.en` (default, English only, fastest)
- `distil-whisper/distil-medium.en` (English only, better quality)
- `distil-whisper/distil-large-v2` (Multilingual, highest quality)
- `distil-whisper/distil-large-v3` (Latest version, best quality)

## Troubleshooting

- If you encounter issues with audio input, make sure your microphone is properly connected and has the necessary permissions.
- For GPU acceleration, make sure you have the appropriate CUDA drivers installed and specify `device="cuda"` when creating the model.
- If you get an error about missing librosa, install the audio extras: `pip install distil-whisper-fastrtc[audio]`
