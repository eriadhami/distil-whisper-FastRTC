"""
Test script to verify that the distil-whisper-fastrtc package works correctly
in a virtual environment.
"""

import sys
import numpy as np

def main():
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("\nTesting distil-whisper-fastrtc package...")
    
    try:
        import distil_whisper_fastrtc
        print(f"Successfully imported distil_whisper_fastrtc version {distil_whisper_fastrtc.__version__}")
        print(f"Available modules: {distil_whisper_fastrtc.__all__}")
        
        # Create a simple model instance (without loading weights)
        from distil_whisper_fastrtc import DistilWhisperSTT
        model = DistilWhisperSTT(device="cpu")
        print(f"Created model instance with model_id: {model.model_id}")
        
        print("\nPackage test successful!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
