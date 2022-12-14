import os
import glob
import argparse
import yaml

from multiprocessing import Process, Queue, Manager

from tkinter import filedialog

from RAVE.audio.AudioManager import AudioManager


def run_loop(file_queue, snr_sdr_dict, worker_num, DEBUG, MASK, TIMER, MODEL):

    # TODO: ADD POSSIBILITY TO RESET AUDIO_MANAGER INSTEAD OF RECREATING
    while not file_queue.empty():

        # Get file in queue
        audio_file = file_queue.get()
        print(f'Worker {worker_num}: Starting speech enhancement on {audio_file}')

        # Run audio manager
        audio_manager = AudioManager(debug=DEBUG, mask=MASK, use_timers=TIMER, model_path=MODEL)
        audio_dict = {
            'name': 'loop_sim_source',
            'type': 'sim',
            'file': audio_file
        }
        audio_manager.initialise_audio(source=audio_dict)
        audio_manager.main_loop()

        # Calculate snr and sdr
        if audio_manager.print_sdr:
            snr_sdr_dict['print_results'] = True
            snr_sdr_dict['count'] += 1
            n = snr_sdr_dict['count']
            snr_sdr_dict['old_snr'] = snr_sdr_dict['old_snr'] * (n-1)/n + audio_manager.old_snr/n
            snr_sdr_dict['new_snr'] = snr_sdr_dict['new_snr'] * (n-1)/n + audio_manager.new_snr/n
            snr_sdr_dict['old_sdr'] = snr_sdr_dict['old_sdr'] * (n-1)/n + audio_manager.old_sdr/n
            snr_sdr_dict['new_sdr'] = snr_sdr_dict['new_sdr'] * (n-1)/n + audio_manager.new_sdr/n

        # audio_manager.reset_manager()

    snr_sdr_dict['gain_snr'] = snr_sdr_dict['new_snr'] - snr_sdr_dict['old_snr']
    snr_sdr_dict['gain_sdr'] = snr_sdr_dict['new_sdr'] - snr_sdr_dict['old_sdr']

    if snr_sdr_dict['print_results']:
        print(f"SDR_SNR: Measured over {snr_sdr_dict['count']} samples :")
        print(f"\t Old SNR: {snr_sdr_dict['old_snr']} \t New SNR: {snr_sdr_dict['new_snr']} "
              f"\t Gain SNR: {snr_sdr_dict['gain_snr']}")
        print(f"\t Old SDR: {snr_sdr_dict['old_sdr']} \t New SDR: {snr_sdr_dict['new_sdr']}"
              f"\t Gain SDR: {snr_sdr_dict['gain_sdr']}")


def main(DEBUG, MASK, TIMER, WORKERS, SOURCE_DIR, MODEL):

    # Check if given a dataset folder or a subfolder
    wav_file = glob.glob(os.path.join(SOURCE_DIR, 'audio.wav'))
    if wav_file:
        # Directly given a subfolder
        audio_dict = {
            'name': 'loop_sim_source',
            'type': 'sim',
            'file': wav_file[0]
        }
        audio_manager = AudioManager(debug=DEBUG, mask=MASK, use_timers=TIMER, model_path=MODEL)
        audio_manager.initialise_audio(source=audio_dict, save_path=None)
        audio_manager.main_loop()
    else:
        with Manager() as manager:
            # Given a whole dataset folder
            worker_count = WORKERS
            worker_list = []
            file_queue = Queue()
            snr_sdr_dict = manager.dict()
            snr_sdr_dict['print_results'] = False
            snr_sdr_dict['count'] = 0
            snr_sdr_dict['old_snr'] = 0
            snr_sdr_dict['new_snr'] = 0
            snr_sdr_dict['gain_snr'] = 0
            snr_sdr_dict['old_sdr'] = 0
            snr_sdr_dict['new_sdr'] = 0
            snr_sdr_dict['gain_sdr'] = 0

            # Get audio files
            input_files = glob.glob(os.path.join(SOURCE_DIR, '**', 'audio.wav'), recursive=True)
            if not input_files:
                print("No audio files found in given directory. Exiting.")
                exit()
            else:
                for audio_file in input_files:
                    file_queue.put(audio_file)

            # Start workers
            for w in range(1, worker_count+1):
                p = Process(target=run_loop, args=(file_queue, snr_sdr_dict, w, DEBUG, MASK, TIMER, MODEL, ))
                worker_list.append(p)
                p.start()
                print(f'Worker {w}: started.')

            # Join workers when done
            for w_num, p in enumerate(worker_list):
                p.join()
                print(f'Worker {w_num+1}: finished.')

            snr_sdr_dict['gain_snr'] = snr_sdr_dict['new_snr'] - snr_sdr_dict['old_snr']
            snr_sdr_dict['gain_sdr'] = snr_sdr_dict['new_sdr'] - snr_sdr_dict['old_sdr']

            results_yaml_path = os.path.join(SOURCE_DIR, 'sdr_results.yaml')
            with open(results_yaml_path, "w") as outfile:
                yaml.dump(snr_sdr_dict, outfile, default_flow_style=None)

            if snr_sdr_dict['print_results']:
                print(f"SDR_SNR: Measured over {snr_sdr_dict['count']} samples :")
                print(f"\t Old SNR: {snr_sdr_dict['old_snr']} \t New SNR: {snr_sdr_dict['new_snr']} "
                      f"\t Gain SNR: {snr_sdr_dict['gain_snr']}")
                print(f"\t Old SDR: {snr_sdr_dict['old_sdr']} \t New SDR: {snr_sdr_dict['new_sdr']}"
                      f"\t Gain SDR: {snr_sdr_dict['gain_sdr']}")


if __name__ == "__main__":

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
        "--model",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to trained model",
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

    # Parse paths
    source_subfolder = args.sources
    if source_subfolder == 'tkinter':
        source_subfolder = filedialog.askdirectory(title="Audio extracts dataset folder")
    model_path = args.model
    if not args.mask and model_path == 'tkinter':
        model_path = filedialog.askopenfilename(title="Trained model file")

    main(args.debug, args.mask, args.timer, args.workers, source_subfolder, model_path)
