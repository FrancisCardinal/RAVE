import glob
import os
import random

import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point
import pyroomacoustics as pra
from pydub import AudioSegment

from .AudioDatasetBuilder import AudioDatasetBuilder


SIDE_ID = 0  # X
DEPTH_ID = 1  # Y
HEIGHT_ID = 2  # Z

SOUND_MARGIN = 1  # Assume every sound source is margins away from receiver and each other
SOURCE_USER_DISTANCE = 5  # Radius of circle in which source can be
USER_MIN_DISTANCE = 2  # Minimum distance needed between user and walls

MAX_POS_TRIES = 50
TILT_RANGE = 0.25

SHOW_GRAPHS = False
SAVE_RIR = True


class AudioDatasetBuilderSim(AudioDatasetBuilder):
    """
    Class which handles the generation of the audio dataset through domain randomization of parameters passed through
    dataset_config.yaml and constructor params.

    Args:
        output_path (str): Path to output directory.
        debug (bool): Run in debug mode.
        configs (dict): Dict containing configurations loaded from dataset_config.yaml
    """

    user_dir = []
    xy_angle = 0

    current_room = []
    current_room_shape = []

    receiver_height = 1.5

    def __init__(self, output_path, debug, configs):

        self.is_sim = True
        super().__init__(output_path, debug, configs)

        # TODO: Split configs between real and sim
        # Room configs
        self.room_shapes = self.configs["room_shapes"]
        self.reverb_room_shapes = self.configs["reverb_room_shapes"]
        self.room_sizes = self.configs["room_sizes"]
        # Noise configs
        self.sample_per_speech = self.configs["sample_per_speech"]
        # RIR configs
        self.wall_absorption_limits = self.configs["wall_absorption_limits"]
        self.rir_reflexion_order = self.configs["rir_reflexion_order"]
        self.air_absorption = self.configs["air_absorption"]
        self.air_humidity = self.configs["air_humidity"]

        # Init
        self.receiver_abs = None
        self.dir_noise_count = self.dir_noise_count_range[0]
        self.speech_noise_start = 0
        self.is_reverb = None
        self.noise_paths = None

    @staticmethod
    def apply_rir(rirs, source):
        """
        Method to apply (possibly multichannel) RIRs to a mono-signal.

        Args:
            rirs (ndarray): RIRs generated previously to reflect room, source position and mic positions
            source (ndarray): Mono-channel input signal to be reflected to multiple microphones
        """
        channels = rirs.shape[0]
        frames = len(source["signal"])
        output = np.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = signal.convolve(source["signal"], rirs[channel_index])[:frames]

        source["signal_w_rir"] = output

    @staticmethod
    def rotate_coords(coords, angle, inverse=False):
        """
        Method to apply rotation matrix to coordinates (only works for x,y).
        Used for user with rotation on x,y axis.

        Args:
            coords (list[float]): Original coordinates to rotate ([x, y] or [x, y, z]).
            angle (float): Angle around which to rotate (rads).
            inverse (bool): Whether to proceed to inverse rotation.

        Returns:
            List with new coordinates ([x, y] or [x, y, z]).
        """
        # TODO: implement 3d rotation matrix when adding head tilt
        old_coords = np.array([coords[SIDE_ID], coords[DEPTH_ID]]).T

        # Get correct rotation matrix
        x_rot = [math.cos(angle), -math.sin(angle)]
        y_rot = [math.sin(angle), math.cos(angle)]
        if inverse:
            x_rot = [math.cos(angle), math.sin(angle)]
            y_rot = [-math.sin(angle), math.cos(angle)]
        rot_matrix = np.array([x_rot, y_rot])

        # Apply rotation matrix
        new_coords = rot_matrix @ old_coords
        if len(coords) == HEIGHT_ID + 1:
            new_coords = new_coords.tolist()
            new_coords.append(coords[HEIGHT_ID])

        return new_coords

    def plot_scene(self, audio_dict, save_path=None):
        """
        Visualize the virtual room by plotting.

        Args:
            audio_dict (dict): All audio sources.
            save_path (str): Save path for 2D plot.

        """
        # 2D
        # Room
        fig2d, ax = plt.subplots()
        ax.set_xlabel("Side (x)")
        plt.xlim([-5, max(self.current_room_size[0], self.current_room_size[1]) + 5])
        plt.ylim([-5, max(self.current_room_size[0], self.current_room_size[1]) + 5])
        ax.set_ylabel("Depth (y)")
        room_corners = self.current_room_shape
        for corner_idx in range(len(room_corners)):
            corner_1 = room_corners[corner_idx]
            corner_2 = room_corners[(corner_idx + 1) % len(room_corners)]
            ax.plot([corner_1[0], corner_2[0]], [corner_1[1], corner_2[1]])

        # User
        for mic_pos in self.receiver_abs:
            ax.scatter(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], marker="x", c="b")
        # ax.text(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], 'User')
        user_dir_point = [self.user_pos[0] + self.user_dir[0], self.user_pos[1] + self.user_dir[1]]
        ax.plot([self.user_pos[0], user_dir_point[0]], [self.user_pos[1], user_dir_point[1]])

        # Source
        speech_pos = audio_dict["speech"][0]["position"]
        speech_name = audio_dict["speech"][0]["name"]
        ax.scatter(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], c="g")
        ax.text(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], speech_name)
        source_circle = plt.Circle(
            (self.user_pos[0] + self.user_dir[0], self.user_pos[1] + self.user_dir[1]),
            SOURCE_USER_DISTANCE,
            color="g",
            fill=False,
        )
        ax.add_patch(source_circle)

        # Noise
        for dir_noise in audio_dict["dir_noise"]:
            dir_noise_pos = dir_noise["position"]
            dir_noise_name = dir_noise["name"]
            ax.scatter(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], c="m")
            ax.text(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], dir_noise_name)
        for dif_noise in audio_dict["dif_noise"]:
            dif_noise_pos = dif_noise["position"]
            dif_noise_name = dif_noise["name"]
            ax.scatter(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], c="r")
            ax.text(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], dif_noise_name)

        plt.savefig(os.path.join(save_path, "scene2d.jpg"))

        if SHOW_GRAPHS:
            fig2d.show()

        plt.close()

        # 3D
        # Room
        if self.is_debug:
            fig3d, ax = self.current_room.plot(img_order=0)
            ax.set_xlabel("Side (x)")
            ax.set_ylabel("Depth (y)")
            ax.set_zlabel("Height (z)")

            # User
            for mic_pos in self.receiver_abs:
                ax.scatter3D(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], mic_pos[HEIGHT_ID], c="b")
            ax.text(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], mic_pos[HEIGHT_ID], "User")
            user_dir_point = [
                self.user_pos[0] + self.user_dir[0],
                self.user_pos[1] + self.user_dir[1],
                self.user_pos[2] + self.user_dir[2],
            ]
            ax.plot(
                [self.user_pos[0], user_dir_point[0]],
                [self.user_pos[1], user_dir_point[1]],
                [self.user_pos[2], user_dir_point[2]],
            )

            # Source
            ax.scatter3D(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], speech_pos[HEIGHT_ID], c="g")
            ax.text(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], speech_pos[HEIGHT_ID], speech_name)

            # Noise
            for dir_noise in audio_dict["dir_noise"]:
                dir_noise_pos = dir_noise["position"]
                dir_noise_name = dir_noise["name"]
                ax.scatter3D(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], dir_noise_pos[HEIGHT_ID], c="m")
                ax.text(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], dir_noise_pos[HEIGHT_ID], dir_noise_name)
            for dif_noise in audio_dict["dif_noise"]:
                dif_noise_pos = dif_noise["position"]
                dif_noise_name = dif_noise["name"]
                ax.scatter3D(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], dif_noise_pos[HEIGHT_ID], c="r")
                ax.text(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], dif_noise_pos[HEIGHT_ID], dif_noise_name)

            fig3d.show()
            plt.close()

    def generate_random_room(self):
        """
        Generate a random room from room shapes and sizes and creates a pra room with it.

        Returns: PyRoomAcoustics room object.

        """
        # Get random room from room shapes and sizes
        random_room_shape = self.room_shapes[np.random.randint(0, len(self.room_shapes))]
        size_factor = np.random.rand() / 2 + 0.75
        x_factor = np.random.randint(self.room_sizes[0][0], self.room_sizes[0][1])
        y_factor = x_factor * size_factor
        z_factor = np.random.randint(self.room_sizes[2][0], self.room_sizes[2][1])
        self.current_room_size = [x_factor, y_factor, z_factor]

        # Apply size to room shape
        self.current_room_shape = []
        for corner in random_room_shape:
            self.current_room_shape.append(
                [corner[0] * self.current_room_size[0], corner[1] * self.current_room_size[1]]
            )
        corners = np.array(self.current_room_shape).T

        # Generate room from corners and height
        if self.wall_absorption_limits[0] == self.wall_absorption_limits[1]:
            self.rir_wall_absorption = self.wall_absorption_limits[0]
        else:
            self.rir_wall_absorption = (
                float(np.random.randint(self.wall_absorption_limits[0] * 100, self.wall_absorption_limits[1] * 100))
                / 100.0
            )
        # TODO: CHECK WALL_ABSORPTION AND SCATTERING VALUES
        mat = pra.Material(float(self.rir_wall_absorption), 0.1)
        room = pra.Room.from_corners(
            corners,
            fs=self.sample_rate,
            air_absorption=self.air_absorption,
            humidity=self.air_humidity,
            max_order=self.rir_reflexion_order,
            materials=mat,
        )
        room.extrude(self.current_room_size[2], materials=mat)
        self.current_room = room

        return room

    def generate_user(self):
        """
        Generate absolute position for user (and for receivers).
        """
        # TODO: GENERATE A NOISE SPEECH SOURCE ON USER TO REPRESENT USER TALKING?

        # Get random position and assign x and y wth sound margins (user not stuck on wall)
        self.user_pos = self.get_random_position(user=True)
        if self.user_pos == -1:
            return
        while True:
            self.xy_angle = (np.random.rand() * 2 * math.pi) - math.pi
            # z_angle = (np.random.rand() - 0.5) * 2 * TILT_RANGE
            x_dir = (SOURCE_USER_DISTANCE + 1) * math.cos(self.xy_angle)
            y_dir = (SOURCE_USER_DISTANCE + 1) * math.sin(self.xy_angle)
            z_dir = 1e-5
            # z_dir = (np.random.rand() - 0.5) * 2 * TILT_RANGE

            if x_dir and y_dir and z_dir:
                break
        self.user_dir = [x_dir, y_dir, z_dir]

        # TODO: IMPLEMENT HEIGHT TILT
        # For every receiver, set x and y by room dimension and add human height as z
        self.receiver_abs = []
        for receiver in self.receiver_rel:
            receiver_center = self.user_pos.copy()
            new_x, new_y, new_z = self.rotate_coords(receiver, self.xy_angle, inverse=False)
            receiver_center[0] += float(new_x)
            receiver_center[1] += float(new_y)
            self.receiver_abs.append(receiver_center)

        # Generate user head wall corners
        # TODO: GENERATE HEAD CLOSER TO CIRCLE THAN SQUARE?
        # TODO: DONT GENERATE HEAD DIRECTLY BEHIND MICROPHONES
        rel_corners = self.receiver_rel
        abs_top_corners = []
        abs_bot_corners = []
        for rel_corner in rel_corners:
            # Rotated axis
            new_x, new_y, new_z = self.rotate_coords(rel_corner, self.xy_angle, inverse=False)
            # Top corners
            corner_top = self.user_pos.copy()
            corner_top[0] += new_x
            corner_top[1] += new_y
            corner_top[2] += 0.1
            abs_top_corners.append(corner_top)
            # Bottom corners
            corner_bot = self.user_pos.copy()
            corner_bot[0] += new_x
            corner_bot[1] += new_y
            corner_bot[2] -= 0.25
            abs_bot_corners.append(corner_bot)

        # Generate user head walls and add to room
        top_plane = np.array(abs_top_corners).T
        bot_plane = np.array(abs_bot_corners).T
        planes = [top_plane, bot_plane]
        for i in range(4):
            side_plane = np.array(
                [abs_top_corners[i], abs_top_corners[(i + 1) % 4], abs_bot_corners[(i + 1) % 4], abs_bot_corners[i]]
            ).T
            planes.append(side_plane)
        wall_absorption = np.array([0.95])
        scattering = np.array([0])
        for plane in planes:
            self.current_room.walls.append(pra.wall_factory(plane, wall_absorption, scattering, "Head"))

        # Generate microphone array in room
        x_mic_values = [mic[0] for mic in self.receiver_abs]
        y_mic_values = [mic[1] for mic in self.receiver_abs]
        z_mic_values = [mic[2] for mic in self.receiver_abs]
        mic_array = np.array([x_mic_values, y_mic_values, z_mic_values])
        self.current_room.add_microphone_array(pra.MicrophoneArray(mic_array, self.current_room.fs))

    def get_random_position(self, source_pos=None, user=False):
        """
        Get random position inside of polygon composed by PyRoomAcoustics room object walls.

        Args:
            source_pos (list): List of coordinates for source position (if exists).
            user (bool): Whether the random position to generate is for the user.

        Returns: Position composed of shapely.Point with coordinates [x, y, z]. -1 if no position could be made.

        """
        # Get room polygon
        room_poly = Polygon(self.current_room_shape)
        minx, miny, maxx, maxy = room_poly.bounds

        # If source, change min and max to generate in user front circle
        if not user and not source_pos:
            user_dir_point = [self.user_pos[0] + self.user_dir[0], self.user_pos[1] + self.user_dir[1]]
            minx = max(minx, user_dir_point[0] - SOURCE_USER_DISTANCE)
            miny = max(miny, user_dir_point[1] - SOURCE_USER_DISTANCE)
            maxx = min(maxx, user_dir_point[0] + SOURCE_USER_DISTANCE)
            maxy = min(maxy, user_dir_point[1] + SOURCE_USER_DISTANCE)

        for _ in range(MAX_POS_TRIES):
            # Create random point inside polygon bounds and check if contained
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            source_radius = USER_MIN_DISTANCE if user else SOUND_MARGIN
            circle = p.buffer(source_radius)
            if not room_poly.contains(circle):
                continue

            # Set height position for user and/or source
            z = self.receiver_height

            if not user:
                if source_pos is None:
                    # If source, check it is contained in circle in front of user
                    dx_user = p.x - user_dir_point[0]
                    dy_user = p.y - user_dir_point[1]
                    is_point_in_user_circle = dx_user**2 + dy_user**2 <= SOURCE_USER_DISTANCE**2
                    if not is_point_in_user_circle:
                        continue

                    # Calculate source direction based on user direction
                    # TODO: add tilt
                    coords = [
                        p.x - self.user_pos[SIDE_ID],
                        p.y - self.user_pos[DEPTH_ID],
                        z - self.user_pos[HEIGHT_ID],
                    ]
                    new_x, new_y, new_z = self.rotate_coords(coords, self.xy_angle, inverse=True)
                    front_facing_angle = 0.5 * math.pi
                    new_x, new_y, new_z = self.rotate_coords([new_x, new_y, new_z], front_facing_angle, inverse=False)
                    self.source_direction = [new_x, new_y, new_z]

                else:
                    # Check if noise is not on user
                    dx_user = p.x - self.user_pos[0]
                    dy_user = p.y - self.user_pos[1]
                    is_point_in_user_circle = dx_user * dx_user + dy_user * dy_user <= SOUND_MARGIN * SOUND_MARGIN
                    if is_point_in_user_circle:
                        continue

                    # Check it is not on top of source
                    dx_src = p.x - source_pos[0]
                    dy_src = p.y - source_pos[1]
                    is_point_in_src_circle = dx_src**2 + dy_src**2 <= SOUND_MARGIN**2
                    if is_point_in_src_circle:
                        continue

                    # Set height position for noise
                    z = np.random.rand() * self.current_room_size[2]

            position = [p.x, p.y, z]
            return position

        return -1

    def generate_and_apply_rirs(self, audio_dict):
        """
        Function that generates Room Impulse Responses (RIRs) to source signal positions and applies them to signals.
        See <https://pyroomacoustics.readthedocs.io/en/pypi-release/>.
        Will have to look later on at
        <https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory> for moving sources.

        Args:
            audio_dict (dict{}): Dictionary containing all audio sources and info.
        Returns:
            False if RIR was broken on one channel.
        """
        # TODO: MOVING RIR

        # Generate RIRs
        self.current_room.compute_rir()
        rir_list = self.current_room.rir
        rirs = np.array(rir_list, dtype=object)

        # Normalise RIR
        # TODO: CHECK FOR TRUE DIVIDE
        for channel_idx, channel_rirs in enumerate(rirs):
            for rir in channel_rirs:
                max_val = np.max(np.abs(rir))
                if max_val == 0:
                    return -1
                rir /= max_val

        # If diffuse, remove early RIR peaks (remove direct and non-diffuse peaks)
        for rir_channel in rirs:
            for diffuse_rir_idx in range(len(audio_dict["dif_noise"])):
                # Get peaks
                rir = rir_channel[len(audio_dict["speech"]) + len(audio_dict["dir_noise"]) + diffuse_rir_idx]
                peaks, _ = signal.find_peaks(rir, distance=100)
                if SHOW_GRAPHS:
                    plt.figure()
                    plt.plot(rir)
                    plt.plot(peaks, rir[peaks], "x")
                    plt.show()

                # Remove peaks from RIR
                min_peak_idx = peaks[len(peaks) // 2]
                rir[:min_peak_idx] = 0

                # Renormalise RIR
                max_val = np.max(np.abs(rir))
                rir /= max_val

                if SHOW_GRAPHS:
                    plt.figure()
                    plt.plot(rir)
                    plt.show()

        if SAVE_RIR:
            fig, axes = plt.subplots(self.n_channels // 2, 2)
            fig.suptitle("RIR graph values")
            for channel_i, channel_rir in enumerate(rirs):
                rir_plot = channel_rir[0].T
                axes[channel_i // 2, channel_i % 2].plot(rir_plot)
            plt.savefig(os.path.join(self.current_subfolder, "rir_channel_plots.jpg"))
            if SHOW_GRAPHS:
                plt.show()
            plt.close()

        # Apply RIR to signal
        rir_idx = 0
        for source_type, source_lists in audio_dict.items():
            for source in source_lists:
                rir = rirs[:, rir_idx]
                if self.is_debug:
                    source["rir"] = rir
                self.apply_rir(rir, source)

        if SHOW_GRAPHS:
            fig, axes = plt.subplots(2)
            fig.suptitle("Speech example before and after RIR")
            axes[0].plot(audio_dict["speech"][0]["signal"])
            axes[1].plot(audio_dict["speech"][0]["signal_w_rir"].T)
            plt.show()

    def init_sim(self, sources_path, noises_path, reverb):
        """
        Function used to initialize variables for simulated dataset generation.
        Args:
            sources_path (str): Path to sources directory. If None, gets folder from config file.
            noises_path (str): Path to noise directory. If None, gets folder from config file.
            reverb (bool): Whether to sue reverb of not (walls fully absorb and max reflexion order = 0).
        """
        self.is_reverb = reverb

        # If use reverb add rooms, if not fix rir max order to 0 and wall absorption to 1
        if self.is_reverb:
            self.room_shapes.extend(self.reverb_room_shapes)
        else:
            self.rir_reflexion_order = 0
            self.wall_absorption_limits = [1, 1]

        # Load input noise paths
        self.noise_paths = glob.glob(os.path.join(noises_path, "*.wav"))

        # Split noise paths (diffuse, directional) and remove banned noises
        for noise in self.noise_paths:
            if any(banned_noise in noise for banned_noise in self.banned_noises):
                continue
            if any(diffuse_noise in noise for diffuse_noise in self.diffuse_noises):
                self.dif_noise_paths.append(noise)
            else:
                self.dir_noise_paths.append(noise)

        # Add speech to directional if in arguments
        if self.speech_as_noise:
            self.speech_noise_start = len(self.dir_noise_paths)
            speeches = glob.glob(os.path.join(sources_path, "*.wav"))
            for i in range(len(self.dir_noise_paths)):
                self.dir_noise_paths.append(speeches[i])

        # # Add speech to directional if in arguments
        # if self.speech_as_noise:
        #     self.dir_noise_paths.extend(sources_path)

    def generate_dataset(self, source_path, save_run):
        """
        Main dataset generator function (for simulated data). Loops over rooms and sources and generates dataset.

        Args:
            source_path (str): String containing path to source audio file.
            save_run (bool): Save dataset to memory or not.

        Returns:
            file_count: Number of audio files (subfolders) generated.
        """

        # Run through every source
        # for source_path in tqdm(self.source_paths, desc="Source Paths used"):
        # Get source_audio for every source file
        source_name = os.path.split(source_path)[1].split(".")[0]  # Get filename for source (before extension)
        source_audio_base = self.read_audio_file(source_path, self.sample_rate)

        speech_noise_count = -1         # If run breaks, generates the same amount of speech noises as before

        # Run SAMPLES_PER_SPEECH samples per speech clip
        samples_created = 0
        total_snr = 0
        while samples_created < self.sample_per_speech:
            audio_source_dict = dict()
            # Get random room and generate user at a position in room
            self.generate_random_room()
            self.generate_user()
            if self.user_pos == -1:
                if self.is_debug: print(f"User position could not be made with {MAX_POS_TRIES}. Restarting new room.")
                continue

            # Generate source position and copy source_audio
            source_pos = self.get_random_position()
            if source_pos == -1:
                if self.is_debug: print(f"Source position could not be made with {MAX_POS_TRIES}. Restarting new room.")
                continue
            source_audio = source_audio_base.copy()
            audio_source_dict["speech"] = [{"name": source_name, "signal": source_audio, "position": source_pos}]

            # Add varying number of directional noise sources
            audio_source_dict["dir_noise"] = []
            dir_noise_source_paths, speech_noise_count = self.get_random_noise(source_path=source_path,
                                                                               sn_count=speech_noise_count)
            for noise_source_path in dir_noise_source_paths:
                dir_noise = dict()
                dir_noise["name"] = os.path.split(noise_source_path)[1].split(".")[0]
                dir_noise["signal"] = self.read_audio_file(noise_source_path, self.sample_rate)
                dir_noise["position"] = self.get_random_position(source_pos)
                audio_source_dict["dir_noise"].append(dir_noise)

            # Add varying number of diffuse noise sources
            audio_source_dict["dif_noise"] = []
            dif_noise_source_paths, _ = self.get_random_noise(diffuse_noise=True)
            for noise_source_path in dif_noise_source_paths:
                dif_noise = dict()
                dif_noise["name"] = os.path.split(noise_source_path)[1].split(".")[0]
                dif_noise["signal"] = self.read_audio_file(noise_source_path, self.sample_rate)
                dif_noise["position"] = self.get_random_position(source_pos)
                audio_source_dict["dif_noise"].append(dif_noise)

            # Create subfolder
            self.current_subfolder = self.create_subfolder(audio_source_dict, self.output_subfolder)

            # Truncate noise and source short to the shortest length
            self.truncate_sources(audio_source_dict)

            # Add sources to PyRoom and generate RIRs
            for source_list in audio_source_dict.values():
                for source in source_list:
                    self.current_room.add_source(source["position"], source["signal"])

            rir_success = self.generate_and_apply_rirs(audio_source_dict)
            if rir_success == -1:
                if self.is_debug: print("RIR returned false, problem with one channel. Restarting sample.")
                continue
            else:
                speech_noise_count = -1

            # Transform data to pydub AudioSegments
            for audio_type in audio_source_dict.values():
                for audio_sample in audio_type:
                    audio_data = audio_sample['signal']
                    pydub_audio_segment = AudioSegment(
                        audio_data.tobytes(),
                        frame_rate=self.sample_rate,
                        sample_width=audio_data.dtype.itemsize,
                        channels=self.n_channels
                    )
                    audio_sample['audio_segment'] = pydub_audio_segment

            # Combine noises
            self.combine_sources(audio_source_dict, ["dir_noise", "dif_noise"], "combined_noise", noise=True)

            # Combine source with noises at a random SNR between limits
            self.snr = np.random.rand() * (self.snr_limits[1] - self.snr_limits[0]) + self.snr_limits[0]
            if self.is_debug: total_snr += self.snr
            self.combine_sources(audio_source_dict, ["speech", "combined_noise"], "combined_audio", snr=self.snr)

            # Save elements
            if save_run:
                subfolder_path = self.save_files(audio_source_dict)
                if self.is_debug:
                    print("Created: " + subfolder_path)

            # Increase counters
            samples_created += 1

        if self.is_debug:
            print(f'Mean SNR: {total_snr/samples_created}')

        return samples_created
