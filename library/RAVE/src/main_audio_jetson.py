import numpy as np
import argparse

from pyodas.utils import get_delays_based_on_mic_array

from RAVE.audio.AudioManager import AudioManager

TARGET = [0, 1, 0.5]


def main(DEBUG, MASK, TIMER):

    # Get all files in a subdirectory
    audio_man = AudioManager(debug=DEBUG, mask=MASK, use_timers=TIMER)
    audio_man.init_app(save_input=True, save_output=True, passthrough_mode=True, output_path="./audio_files", gain=2)

    if MASK:
        audio_man.set_target(np.array(TARGET))
    else:
        target = np.array([TARGET])
        delay = get_delays_based_on_mic_array(target, frame_size=audio_man.frame_size, mic_array=audio_man.mic_array)[0]
        audio_man.set_target(delay)

    audio_man.start_app()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("-m", "--mask", action="store_true", help="Run with KISS mask instead of model prediction")
    parser.add_argument("-t", "--timer", action="store_true", help="Calculate time with timers")

    args = parser.parse_args()

    main(args.debug, args.mask, args.timer)
