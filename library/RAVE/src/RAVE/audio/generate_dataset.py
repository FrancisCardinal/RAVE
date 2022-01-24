import argparse
from tkinter import filedialog

from audio import AudioDatasetBuilder

SAMPLES_PER_SPEECH = 1


# Script used to generate the audio dataset
def main(SOURCES, NOISES, OUTPUT, MAX_SOURCES, SPEECH_AS_NOISE, DEBUG):
    dataset_builder = AudioDatasetBuilder(SOURCES, NOISES, OUTPUT, MAX_SOURCES, SPEECH_AS_NOISE, DEBUG)

    # For each room
    for room in dataset_builder.rooms:
        # TODO: Add random position for user (receivers)
        # TODO: Check if noise and source audios are too big for memory
        # TODO: Check if saving audios with RIRs are worth in long run

        # Generate receiver positions from room dimensions
        dataset_builder.generate_abs_receivers(room)

        # Run through every source
        for source_path in dataset_builder.source_paths:
            source_name = source_path.split('\\')[-1].split('.')[0]  # Get filename for source (before extension '.')
            source_audio = dataset_builder.read_audio_file(source_path)

            # Run SAMPLES_PER_SPEECH samples per speech clip
            for _ in range(SAMPLES_PER_SPEECH):
                # Generate source audio and RIR
                source_pos = dataset_builder.generate_random_position(room)
                source_with_rir = dataset_builder.generate_and_apply_rirs(source_audio, source_pos, room)
                source_gt = dataset_builder.generate_ground_truth(source_with_rir)

                # Add varying number of noise sources
                noise_source_paths = dataset_builder.get_random_noise()
                noise_pos_list = dataset_builder.generate_random_position(room, source_pos, True)

                # For each noise get name, RIR and ground truth
                noise_name_list = []
                noise_RIR_list = []
                noise_gt_list = []
                for noise_source_path, noise_pos in zip (noise_source_paths, noise_pos_list):
                    noise_name_list.append(noise_source_path.split('\\')[-1].split('.')[0])
                    noise_audio = dataset_builder.read_audio_file(noise_source_path)
                    noise_with_rir = dataset_builder.generate_and_apply_rirs(noise_audio, noise_pos, room)
                    noise_RIR_list.append(noise_with_rir)
                    noise_gt_list.append(dataset_builder.generate_ground_truth(noise_with_rir))

                # Combine noises and get gt
                combined_noise_rir = dataset_builder.combine_sources(noise_RIR_list)
                combined_noise_gt = dataset_builder.generate_ground_truth(combined_noise_rir)

                # Combine source with noises
                audio = [source_with_rir, combined_noise_rir]
                combined_audio = dataset_builder.combine_sources(audio)
                combined_gt = dataset_builder.generate_ground_truth(combined_audio)

                # Save element to dataset
                subfolder_path = dataset_builder.save_files(combined_audio, combined_gt,
                                                            source_name, source_gt, source_pos,
                                                            noise_name_list, noise_pos_list, noise_gt_list,
                                                            combined_noise_gt)

                print("Created: " + subfolder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run in debug mode"
    )

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
        "-x", "--xtra_speech", action="store_true", help="Add speech as possible noise sources"
    )
    parser.add_argument(
        "-m",
        "--max_sources",
        action="store",
        type=int,
        default=1,
        help="Maximum number of interfering sources (and noises) to add",
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
        args.max_sources,
        args.xtra_speech,
        args.debug
    )
