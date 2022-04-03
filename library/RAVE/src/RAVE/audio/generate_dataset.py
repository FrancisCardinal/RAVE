import argparse
from tkinter import filedialog

import time
import os
from glob import glob

from multiprocessing import Process, Queue, Value, Array

import sys
sys.path.insert(1, './Dataset')
from Dataset.AudioDatasetBuilder import AudioDatasetBuilder

# CONFIGS_PATH = 'C:\\GitProjet\\RAVE\\library\\RAVE\\src\\RAVE\\audio\\Dataset\\' + 'dataset_config.yaml'
CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset', 'dataset_config.yaml')


def run_generator_loop(source_queue, worker_num, run_params, configs, file_cnt, total_cnt):
    # TODO: CHECK TO RUN 1 GENERATOR AND ALL WORKERS CALL ON IT
    dataset_builder = AudioDatasetBuilder(run_params['SOURCES'],
                                          run_params['NOISES'],
                                          run_params['OUTPUT'],
                                          run_params['DEBUG'],
                                          run_params['REVERB'],
                                          configs)
    while source_queue.qsize() > 0:
        # Get source file
        audio_file = source_queue.get()

        # Run generator
        file_increment, dataset_list = dataset_builder.generate_dataset(source_path=audio_file, save_run=True)

        # Add results
        with file_cnt.get_lock():
            file_cnt.value += file_increment
        print(f'Worker {worker_num}: {file_cnt.value}/{total_cnt.value} ({audio_file})')


# Script used to generate the audio dataset
def main(SOURCES, NOISES, OUTPUT, DEBUG, WORKERS, REVERB):

    start_time = time.time()

    # Save run parameters
    # TODO: CHECK IF WE CAN PUT SOME PARAMS IN CONFIG FILE
    run_params = {
        'SOURCES': SOURCES,
        'NOISES': NOISES,
        'OUTPUT': OUTPUT,
        'DEBUG': DEBUG,
        'REVERB': REVERB
    }

    # Load multiprocess
    worker_list = []
    file_queue = Queue()

    # Load sources and configs
    source_paths_list = glob(os.path.join(SOURCES, '*.wav'))
    for source_path in source_paths_list:
        file_queue.put(source_path)
    configs = AudioDatasetBuilder.load_configs(CONFIGS_PATH)
    print(f"Starting to generate dataset with {configs}.")
    total_file_cnt = len(source_paths_list) * configs['sample_per_speech']
    print(f'Generating {total_file_cnt} dataset elements into {OUTPUT}')

    # Shared global variables
    file_cnt = Value('i', 0)
    total_cnt = Value('i', total_file_cnt)

    # Start workers
    for w in range(1, WORKERS+1):
        p = Process(target=run_generator_loop, args=(file_queue, w, run_params, configs, file_cnt, total_cnt))
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
        "--sources",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to audio sources",
    )
    parser.add_argument(
        "-n",
        "--noises",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to noise sources",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to output dataset folder",
    )

    # RIR params
    parser.add_argument(
        "-r", "--reverb", action="store_true", help="Use reverberation to generate RIRs"
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
    source_subfolder = args.sources
    if source_subfolder == 'tkinter':
        source_subfolder = filedialog.askdirectory(title="Sources folder")

    # parse noises
    noise_subfolder = args.noises
    if noise_subfolder == 'tkinter':
        noise_subfolder = filedialog.askdirectory(title="Noises folder")

    # parse output
    output_subfolder = args.output
    if output_subfolder == 'tkinter':
        output_subfolder = filedialog.askdirectory(title="Output folder")

    main(
        source_subfolder,
        noise_subfolder,
        output_subfolder,
        args.debug,
        args.workers,
        args.reverb
    )
