import argparse
from tkinter import filedialog

from ..audio import AudioDatasetBuilder

# Script used to generate the audio dataset
def main(SOURCES, NOISES, OUTPUT, MAX_SOURCES):
    dataset_builder = AudioDatasetBuilder(MAX_SOURCES, SOURCES, OUTPUT, NOISES)

    positions = dataset_builder.positions

    # For each room
    for room in dataset_builder.rooms:
        # TODO: Check if noise and source audios are too big for memory
        # TODO: Check if saving audios with RIRs are worth in long run

        # Generate noises once for each room
        noise_with_rirs_pos = []
        for noise_pos in dataset_builder.positions:
            noise_with_rirs = []
            for noise_src in dataset_builder.noise_paths:
                noise_name = noise_src.split('.')[-2]       # Get filename for noise
                noise_with_rir = dataset_builder.generate_and_apply_rirs(noise_src, noise_pos, room)
                noise_with_rirs.append((noise_name, noise_with_rir))
            noise_with_rirs_pos.append(noise_with_rirs)

        # Generate source files with rirs
        source_with_rirs_pos = []
        for source_pos in dataset_builder.positions:
            source_with_rirs = []
            for source_src in dataset_builder.source_paths:
                source_name = source_src.split('.')[-2]       # Get filename for source
                source_with_rir = dataset_builder.generate_and_apply_rirs(source_src, source_pos, room)
                source_with_rirs.append((source_name, source_with_rir))
            source_with_rirs_pos.append(source_with_rirs)

        # Combine sources and noises
        # TODO: Add option for multiple noise sources
        for source_i, source_audio_pos_list in enumerate(source_with_rirs_pos):
            for noise_i, noise_audio_pos_list in enumerate(noise_with_rirs_pos):
                # Skip noises where position is on top of source
                if noise_i != source_i:
                    for source_audio in source_audio_pos_list:
                        source_name = source_audio[0]
                        source_signal = source_audio[1]
                        source_gt = dataset_builder.generate_ground_truth(source_signal)
                        for noise_audio in noise_audio_pos_list:
                            # TODO: Add optimisation to stop calculating SCM on same combination of sounds
                            noise_name = noise_audio[0]
                            noise_name_list = [noise_name]
                            noise_signal = noise_audio[1]
                            noise_signal_list = [noise_signal]
                            noise_gt = dataset_builder.generate_ground_truth(noise_signal)
                            noise_gt_list = [noise_gt]
                            combined_noise_gt = noise_gt
                            combined_audio = dataset_builder.combine_sources(source_signal, noise_signal_list)
                            combined_scm = dataset_builder.generate_ground_truth(combined_audio)

                            dataset_builder.save_files(combined_audio, combined_scm,
                                                       source_name, source_gt,
                                                       noise_name_list, noise_gt_list, combined_noise_gt)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-t", "--train", action="store_true", help="Train the neural network"
    # )
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

    parser.add_argument(
        "-m",
        "--max_sources",
        action="store",
        type=int,
        default=3,
        help="Maximum number of interfering sources (and noises) to add",
    )

    args = parser.parse_args()

    # parse sources
    source_subfolder = args.sources
    if source_subfolder == 'tkinter':
        source_subfolder = filedialog.askdirectory()

    # parse noises
    noise_subfolder = args.noises
    if noise_subfolder == 'tkinter':
        noise_subfolder = filedialog.askdirectory()

    # parse output
    output_subfolder = args.output
    if output_subfolder == 'tkinter':
        output_subfolder = filedialog.askdirectory()

    main(
        source_subfolder,
        noise_subfolder,
        output_subfolder,
        args.max_sources
    )
