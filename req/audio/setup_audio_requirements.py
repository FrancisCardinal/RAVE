import os
import sys

os.system("pip install pipwin")
os.system("pipwin install pyaudio")
os.system("pipwin install shapely")

if len(sys.argv) > 1:
    if sys.argv[1] == '-r':
        os.system(f"pip install -r audio_requirements.txt")
