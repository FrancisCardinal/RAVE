import os
import glob
import argparse

from multiprocessing import Process, Queue

from tkinter import filedialog

from RAVE.audio.AudioManager import AudioManager


def run_loop(file_queue, worker_num, DEBUG, MASK, TIMER):

    # TODO: ADD POSSIBILITY TO RESET AUDIO_MANAGER INSTEAD OF RECREATING
    while not file_queue.empty():

        # Get file in queue
        audio_file = file_queue.get()
        print(f'Worker {worker_num}: Starting speech enhancement on {audio_file}')

        # Run audio manager
        audio_manager = AudioManager(debug=DEBUG, mask=MASK, use_timers=TIMER)
        audio_dict = {
            'name': 'loop_sim_source',
            'type': 'sim',
            'file': audio_file
        }
        audio_manager.initialise_audio(source=audio_dict)
        audio_manager.main_loop()
        # audio_manager.reset_manager()


def main2(DEBUG, MASK, TIMER, WORKERS, SOURCE_DIR):

    # Get all files in a subdirectory
    if SOURCE_DIR == 'default':
        audio_manager = AudioManager(debug=DEBUG, mask=MASK, use_timers=TIMER)
        audio_manager.initialise_audio()
        audio_manager.main_loop()
    else:
        # Check if given a dataset folder or a subfolder
        wav_file = glob.glob(os.path.join(SOURCE_DIR, 'audio.wav'))
        if wav_file:
            # Directly given a subfolder
            audio_dict = {
                'name': 'loop_sim_source',
                'type': 'sim',
                'file': wav_file[0]
            }
            audio_manager = AudioManager(debug=DEBUG, mask=MASK, use_timers=TIMER)
            audio_manager.initialise_audio(source=audio_dict, path=None)
            audio_manager.main_loop()
        else:
            # Given a whole dataset folder
            worker_count = WORKERS
            worker_list = []
            file_queue = Queue()
            # Get audio files
            # TODO: ADD OPTION TO GIVE .WAV DIRECTLY
            input_files = glob.glob(os.path.join(SOURCE_DIR, '**', 'audio.wav'), recursive=True)
            for audio_file in input_files:
                file_queue.put(audio_file)

            # Start workers
            for w in range(1, worker_count+1):
                p = Process(target=run_loop, args=(file_queue, w, DEBUG, MASK, TIMER, ))
                worker_list.append(p)
                p.start()
                print(f'Worker {w}: started.')

            # Join workers when done
            for w_num, p in enumerate(worker_list):
                p.join()
                print(f'Worker {w_num+1}: finished.')


if __name__ == "__main__":
    # main2()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run in debug mode"
    )
    parser.add_argument(
        "-m", "--mask", action="store_true", help="Run with KISS mask instead of model prediction"
    )
    parser.add_argument(
        "-t", "--timer", action="store_true", help="Calculate time with timers"
    )

    parser.add_argument(
        "-s",
        "--sources",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to audio sources to enhance",
    )

    parser.add_argument(
        "-w",
        "--workers",
        action="store",
        type=int,
        default=1,
        help="Number of workers to use to run generator.",
    )

    args = parser.parse_args()

    # parse sources
    source_subfolder = args.sources
    if source_subfolder == 'tkinter':
        source_subfolder = filedialog.askdirectory(title="Audio extracts dataset folder")

    main2(args.debug, args.mask, args.timer, args.workers, source_subfolder)
