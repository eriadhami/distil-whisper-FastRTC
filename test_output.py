"""
Script to test output visibility by writing to a file.
"""

import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'output.txt')

print(f"Writing output to: {output_path}")

# Redirect stdout to a file
with open(output_path, 'w') as f:
    # Print some test output
    print("Testing output visibility", file=f)
    
    # Try to import our package
    try:
        import distil_whisper_fastrtc
        print(f"Successfully imported distil_whisper_fastrtc version {distil_whisper_fastrtc.__version__}", file=f)
        print(f"Available modules: {distil_whisper_fastrtc.__all__}", file=f)
    except ImportError as e:
        print(f"Failed to import distil_whisper_fastrtc: {e}", file=f)
    
    # Print Python path
    print("\nPython path:", file=f)
    for path in sys.path:
        print(f"  {path}", file=f)

print(f"Finished writing to: {output_path}")
