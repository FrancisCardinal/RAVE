import os
import sys
import argparse


def install_req(jetson_req, audio_req):

    # Check coherence
    if jetson_req and audio_req:
        print("Cannot install both dependencies, choose jetson or audio.")
        exit()

    # Detect system os
    if not jetson_req and not audio_req:
        if os.name != 'nt':
            os.system(f"pip install -r req/requirements_linux.txt")
        else:
            os.system("pip install pipwin")
            os.system("pipwin install pyaudio")
            os.system(f"pip install -r req/requirements_win.txt")

    # Install audio requirements
    if audio_req:
        if os.name != 'nt':
            os.system("pip install pyaudio")
            os.system("pip install shapely")
        else:
            os.system("pip install pipwin")
            os.system("pipwin install pyaudio")
            os.system("pipwin install shapely")
        os.system(f"pip install -r req/requirements_audio.txt")

    # Install jetson requirements
    if jetson_req:
        os.system(f"pip install -r req/requirements_jetson.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio", action="store_true", help="Install audio requirements"
    )

    parser.add_argument(
        "-j", "--jetson", action="store_true", help="Install jetson requirements"
    )

    args = parser.parse_args()

    install_req(
        args.jetson,
        args.audio
    )


