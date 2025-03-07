"""
Simple script to test importing the distil-whisper-fastrtc package.
"""

try:
    import distil_whisper_fastrtc
    print(f"Successfully imported distil_whisper_fastrtc version {distil_whisper_fastrtc.__version__}")
    print(f"Available modules: {distil_whisper_fastrtc.__all__}")
except ImportError as e:
    print(f"Failed to import distil_whisper_fastrtc: {e}")
