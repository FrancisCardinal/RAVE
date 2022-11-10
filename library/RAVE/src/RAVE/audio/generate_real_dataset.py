import argparse
from tkinter import filedialog

import time
import yaml
import os
import sys
from glob import glob

from multiprocessing import Process, Queue, Value, Array


sys.path.insert(1, './Dataset')
from Dataset.AudioDatasetBuilder_Real import AudioDatasetBuilderReal

CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset', 'dataset_config.yaml')


def run_generator_loop(source_queue, worker_num, run_params, configs, file_cnt):
    """
    Function to be called by individual workers. Creates generator object, gets an audio file from the queue and
    generated the audio files required with the configs.

    Args:
        source_queue (Queue): Queue containing audio files from source folder.
        worker_num (int): Number index of current worker process.
        run_params (dict): Dictionary containing info from run parameters.
        configs (dict): Configurations loaded from config file.
        file_cnt (int): Shared int containing current file count.
    """
    # TODO: CHECK TO RUN 1 GENERATOR AND ALL WORKERS CALL ON IT
    dataset_builder = AudioDatasetBuilderReal(run_params['OUTPUT'],
                                              run_params['DEBUG'],
                                              configs)
    while not source_queue.empty():
        # Get source file
        audio_paths = source_queue.get()

        # Run generator
        file_increment = dataset_builder.generate_dataset(source_paths=audio_paths, save_run=True)

        # Add results
        with file_cnt.get_lock():
            file_cnt.value += file_increment
        print(f'Worker {worker_num}: {file_cnt.value} ({audio_paths["speech"]})')


# Script used to generate the audio dataset
def main(SOURCE, OUTPUT, DEBUG, WORKERS):
    """
    Main running loop to generate dataset. Calls forth worker functions with multiprocessing.

    Args:
        SOURCE (str): Path to sources folder.
        OUTPUT (str): Path to output folder.
        DEBUG (bool): Whether to run in debug mode or not.
        WORKERS (int): Number of worker processes to call.
    """

    start_time = time.time()

    # Save run parameters
    # TODO: CHECK IF WE CAN PUT SOME PARAMS IN CONFIG FILE
    run_params = {
        'SOURCE': SOURCE,
        'OUTPUT': OUTPUT,
        'DEBUG': DEBUG
    }

    # Load multiprocess
    worker_list = []
    audio_queue = Queue()

    # Load configs
    configs = AudioDatasetBuilderReal.load_configs(CONFIGS_PATH)

    # Load sources per room
    user_pos_paths = [os.path.normpath(i) for i in glob(os.path.join(SOURCE, '*', '*'))]
    for user_pos_path in user_pos_paths:

        speech_paths = glob(os.path.join(user_pos_path, 'speech', '**', 'audio.wav'))
        noise_paths = glob(os.path.join(user_pos_path, 'noise', '**', 'audio.wav'))

        for idx, speech_path in enumerate(speech_paths):
            other_speech = speech_paths[:idx] + speech_paths[idx+1:]

            # Speech configs
            config_path = os.path.join(os.path.split(speech_path)[0], 'configs.yaml')
            with open(config_path, "r") as stream:
                try:
                    sample_configs = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # Job dict
            real_audio_dict = {'speech': speech_path,
                               'noise': noise_paths,
                               'other_speech': other_speech,
                               'configs': sample_configs}
            audio_queue.put(real_audio_dict)

    print(f"Starting to generate dataset with {configs}.")

    # Shared global variables
    file_cnt = Value('i', 0)

    # Start workers
    for w in range(1, WORKERS+1):
        p = Process(target=run_generator_loop, args=(audio_queue, w, run_params, configs, file_cnt))
        worker_list.append(p)
        p.start()
        print(f'Worker {w}: started.')

    # Join workers when done
    for w_num, p in enumerate(worker_list):
        p.join()
        print(f'Worker {w_num + 1}: finished.')

    end_time = time.time()
    print(f"Finished generating dataset.")
    print(f"{end_time-start_time} seconds user time to generate {file_cnt.value} files into {OUTPUT}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run in debug mode"
    )

    # Path variables
    parser.add_argument(
        "-s",
        "--source",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to recorded samples (dir/room/location/{speech|noise}/name).",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to output dataset folder",
    )

    # Multiprocess
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
    source_subfolder = args.source
    if source_subfolder == 'tkinter':
        source_subfolder = filedialog.askdirectory(title="Sources folder")

    # parse output
    output_subfolder = args.output
    if output_subfolder == 'tkinter':
        output_subfolder = filedialog.askdirectory(title="Output folder")

    main(
        source_subfolder,
        output_subfolder,
        args.debug,
        args.workers
    )
