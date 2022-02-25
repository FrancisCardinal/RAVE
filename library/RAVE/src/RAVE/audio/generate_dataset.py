import argparse
from tkinter import filedialog

import time

import sys
sys.path.insert(1, './Dataset')
from Dataset.AudioDatasetBuilder import AudioDatasetBuilder


# Script used to generate the audio dataset
def main(SOURCES, NOISES, OUTPUT, NOISE_COUNT, SPEECH_AS_NOISE, SAMPLE_COUNT, DEBUG):
    start_time = time.time()
    dataset_builder = AudioDatasetBuilder(SOURCES, NOISES, OUTPUT, NOISE_COUNT, SPEECH_AS_NOISE, SAMPLE_COUNT, DEBUG)
    file_count, dataset_list = dataset_builder.generate_dataset(save_run=True)
    end_time = time.time()
    print(f"Finished generating dataset. Generated {file_count} files into {OUTPUT}.")
    print(f"In total, took {end_time-start_time} seconds to generate.")


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

    # Speech variables
    parser.add_argument(
        "-a",
        "--sample_count",
        action="store",
        type=int,
        default=25,
        help="Number of samples to use per speech source.",
    )

    # Noise variables
    parser.add_argument(
        "-x", "--xtra_speech", action="store_true", help="Add speech as possible noise sources"
    )
    parser.add_argument(
        "-c",
        "--noise_count",
        action="store",
        nargs='+',
        default=[5, 10],
        help="Range of noise count to add to audio (ex. '-c 5 10' to have 5 to 10 noise sources)",
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

    # noise count to int
    for i in range(len(args.noise_count)):
        args.noise_count[i] = int(args.noise_count[i])

    main(
        source_subfolder,
        noise_subfolder,
        output_subfolder,
        args.noise_count,
        args.xtra_speech,
        args.sample_count,
        args.debug
    )
