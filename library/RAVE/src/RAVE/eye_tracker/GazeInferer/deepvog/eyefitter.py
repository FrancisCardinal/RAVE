import numpy as np
from .unprojection import (
    convert_ell_to_general,
    unprojectGazePositions,
    reproject,
    reverse_reproject,
)
from .intersection import (
    NoIntersectionError,
    intersect,
    fit_ransac,
    line_sphere_intersect,
)

"""
Unless specified, all units are in pixels.
All calculations are in camera frame (conversion would be commented)

"""


class SingleEyeFitter(object):
    def __init__(
        self,
        focal_length,
        pupil_radius,
        initial_eye_z,
        x_angle,
        image_shape,
        original_image_size_pre_crop,
        sensor_size,
    ):
        self.camera_rotation_matrix = np.eye(3)
        theta = np.deg2rad(x_angle)
        self.camera_rotation_matrix[1, 1] = np.cos(theta)
        self.camera_rotation_matrix[2, 2] = np.cos(theta)
        self.camera_rotation_matrix[2, 1] = np.sin(theta)
        self.camera_rotation_matrix[1, 2] = -np.sin(theta)
        self.camera_rotation_matrix = self.camera_rotation_matrix[0:2, 0:2]

        image_scaling_factor = np.linalg.norm(image_shape) / np.linalg.norm(
            original_image_size_pre_crop
        )
        x_angle_correction_factor = np.linalg.norm(
            original_image_size_pre_crop
        ) / np.linalg.norm(
            self.camera_rotation_matrix @ original_image_size_pre_crop
        )
        mm2px_scaling = (
            image_scaling_factor
            * x_angle_correction_factor
            * np.linalg.norm(original_image_size_pre_crop)
            / np.linalg.norm(sensor_size)
        )

        focal_length *= mm2px_scaling
        pupil_radius *= mm2px_scaling
        initial_eye_z *= mm2px_scaling

        self.focal_length = focal_length
        self.image_shape = image_shape

        self.pupil_radius = pupil_radius
        self.vertex = [0, 0, -focal_length]
        self.initial_eye_z = initial_eye_z

        # (p,n) of unprojected gaze vector and pupil 3D position in
        # SINGLE OBSERVATION
        # reserved for (3,1) np.array in camera frame
        self.current_gaze_pos = 0
        # reserved for (3,1) np.array in camera frame
        self.current_gaze_neg = 0
        # reserved for (3,1) np.array in camera frame
        self.current_pupil_3Dcentre_pos = 0
        # reserved for (3,1) np.array in camera frame
        self.current_pupil_3Dcentre_neg = 0
        # reserved for numpy array (2,1) in numpy indexing frame
        self.current_ellipse_centre = 0

        # List of parameters across a number (m) of observations
        # A list: ["gaze_positive"~np(m,3), "gaze_negative"~np(m,3)]
        self.unprojected_gaze_vectors = []
        # ["pupil_3Dcentre_positive"~np(m,3),"pupil_3Dcentre_negative"~np(m,3)]
        self.unprojected_3D_pupil_positions = []
        # reserved for numpy array (m,2) in numpy indexing frame,
        self.ellipse_centres = None
        # m = number of fitted ellipse centres corresponding to the projected
        # gaze lines
        # reserved for (m,3) np.array in camera frame
        self.selected_gazes = None
        # reserved for (m,3) np.array in camera frame
        self.selected_pupil_positions = None

        # Parameters of the eye model for consistent pupil estimate after
        # initialisation
        # reserved for numpy array (2,1). Centre coordinate in
        # numpy indexing frame.
        self.projected_eye_centre = None
        # reserved for (3,1) numpy array. 3D centre coordinate in camera frame
        self.eye_centre = None
        self.aver_eye_radius = None  # Scaler

        # Results of consistent pupil estimate
        self.pupil_new_position_max = None  # numpy array (3,1)
        self.pupil_new_position_min = None  # numpy array (3,1)
        self.pupil_new_radius_max = None  # scalar
        self.pupil_new_radius_min = None  # scalar
        self.pupil_new_gaze_max = None  # numpy array (3,1)
        self.pupil_new_gaze_min = None  # numpy array (3,1)

    def unproject_single_observation(self, observation):
        centre, w, h, radian = observation

        centre = self.camera_rotation_matrix @ centre

        wh = [w, h]
        wh = self.camera_rotation_matrix @ wh
        w, h = wh[0], wh[1]

        centre_cam = centre.copy()
        centre_cam[0] = centre_cam[0] - self.image_shape[1] / 2
        centre_cam[1] = centre_cam[1] - self.image_shape[0] / 2

        # Convert ellipse parameters to the coefficients of the general form
        # of ellipse equation
        A, B, C, D, E, F = convert_ell_to_general(
            centre_cam[0], centre_cam[1], w, h, radian
        )
        ell_co = (A, B, C, D, E, F)

        # Unproject the ellipse to obtain 2 ambiguous gaze vectors
        # with numpy shape (3,1),
        # and pupil_centre with numpy shape (3,1)
        (
            unprojected_gaze_pos,
            unprojected_gaze_neg,
            unprojected_pupil_3Dcentre_pos,
            unprojected_pupil_3Dcentre_neg,
        ) = unprojectGazePositions(self.vertex, ell_co, self.pupil_radius)

        # Normalize the gaze vectors and only take their real component
        unprojected_gaze_pos = unprojected_gaze_pos / np.linalg.norm(
            unprojected_gaze_pos
        )
        unprojected_gaze_neg = unprojected_gaze_neg / np.linalg.norm(
            unprojected_gaze_neg
        )

        (
            unprojected_gaze_pos,
            unprojected_gaze_neg,
            unprojected_pupil_3Dcentre_pos,
            unprojected_pupil_3Dcentre_neg,
        ) = (
            np.real(unprojected_gaze_pos),
            np.real(unprojected_gaze_neg),
            np.real(unprojected_pupil_3Dcentre_pos),
            np.real(unprojected_pupil_3Dcentre_neg),
        )
        (
            self.current_gaze_pos,
            self.current_gaze_neg,
            self.current_pupil_3Dcentre_pos,
            self.current_pupil_3Dcentre_neg,
        ) = (
            unprojected_gaze_pos,
            unprojected_gaze_neg,
            unprojected_pupil_3Dcentre_pos,
            unprojected_pupil_3Dcentre_neg,
        )
        self.current_ellipse_centre = np.array(centre).reshape(2, 1)

    def add_to_fitting(self):
        # Append parameterised gaze lines for fitting
        if (
            (self.current_gaze_pos is None)
            or (self.current_gaze_neg is None)
            or (self.current_pupil_3Dcentre_pos is None)
            or (self.current_pupil_3Dcentre_neg is None)
            or (self.current_ellipse_centre is None)
        ):
            raise TypeError(
                'No ellipse was caught in this observation, thus "None" is\
                    being added for fitting set, which is not allowed.\
                        Please manually skip this condition.'
            )

        # Store the gaze vectors and pupil 3D centres
        if (
            (len(self.unprojected_gaze_vectors) == 0)
            or (len(self.unprojected_3D_pupil_positions) == 0)
            or (self.ellipse_centres is None)
        ):
            self.unprojected_gaze_vectors.append(
                self.current_gaze_pos.reshape(1, 3)
            )
            self.unprojected_gaze_vectors.append(
                self.current_gaze_neg.reshape(1, 3)
            )
            self.unprojected_3D_pupil_positions.append(
                self.current_pupil_3Dcentre_pos.reshape(1, 3)
            )
            self.unprojected_3D_pupil_positions.append(
                self.current_pupil_3Dcentre_neg.reshape(1, 3)
            )
            self.ellipse_centres = self.current_ellipse_centre.reshape(1, 2)
        else:
            self.unprojected_gaze_vectors[0] = np.vstack(
                (
                    self.unprojected_gaze_vectors[0],
                    self.current_gaze_pos.reshape(1, 3),
                )
            )
            self.unprojected_gaze_vectors[1] = np.vstack(
                (
                    self.unprojected_gaze_vectors[1],
                    self.current_gaze_neg.reshape(1, 3),
                )
            )
            self.unprojected_3D_pupil_positions[0] = np.vstack(
                (
                    self.unprojected_3D_pupil_positions[0],
                    self.current_pupil_3Dcentre_pos.reshape(1, 3),
                )
            )
            self.unprojected_3D_pupil_positions[1] = np.vstack(
                (
                    self.unprojected_3D_pupil_positions[1],
                    self.current_pupil_3Dcentre_neg.reshape(1, 3),
                )
            )
            self.ellipse_centres = np.vstack(
                (
                    self.ellipse_centres,
                    self.current_ellipse_centre.reshape(1, 2),
                )
            )

    def fit_projected_eye_centre(
        self, ransac=False, max_iters=1000, min_distance=2000
    ):
        # You will need to determine when to fit outside of the class
        if (self.unprojected_gaze_vectors is None) or (
            self.ellipse_centres is None
        ):
            msg = "No unprojected gaze lines or ellipse centres were found\
                (not yet initalized). It is likely that the network fails\
                to segment the pupil from the video.Please ensure your input\
                video contains only a single eye but not other facial/body\
                features."
            raise TypeError(msg)

        # Combining positive and negative gaze vectors
        a = np.vstack((self.ellipse_centres, self.ellipse_centres))
        n = np.vstack(
            (
                self.unprojected_gaze_vectors[0][:, 0:2],
                self.unprojected_gaze_vectors[1][:, 0:2],
            )
        )  # [:, 0:2] takes only 2D projection

        # Normalisation of the 2D projection of gaze vectors is
        # done inside intersect()
        if ransac:
            # Assuming 0.5% of outliners
            samples_to_fit = max(np.ceil(a.shape[0] / 20).astype(np.int), 2)
            self.projected_eye_centre = fit_ransac(
                a,
                n,
                max_iters=max_iters,
                samples_to_fit=samples_to_fit,
                min_distance=min_distance,
            )
        else:
            self.projected_eye_centre = intersect(a, n)
        if self.projected_eye_centre is None:
            raise TypeError("Projected_eye_centre was not fitted.")
        return self.projected_eye_centre

    def estimate_eye_sphere(self):
        # This function is called once after fit_projected_eye_centre()
        # self.initial_eye_z is required (in pixel unit)
        # self.initial_eye_z shall be the z-distance between the point and
        # camera vertex (in camera frame)
        if self.projected_eye_centre is None:
            # pdb.set_trace()
            raise TypeError("Projected_eye_centre must be initialized first")

        # Unprojecting the 2D projected eye centre to 3D.
        # Converting the projected_eye_centre from numpy indexing frame to
        # camera frame
        projected_eye_centre_camera_frame = self.projected_eye_centre.copy()
        projected_eye_centre_camera_frame[0] = (
            projected_eye_centre_camera_frame[0] - self.image_shape[1] / 2
        )
        projected_eye_centre_camera_frame[1] = (
            projected_eye_centre_camera_frame[1] - self.image_shape[0] / 2
        )

        # Unprojection: Nearest intersection of two lines.
        # a = [eye_centre, pupil_3Dcentre], n =[gaze_vector, pupil_3D_centre]
        projected_eye_centre_camera_frame_scaled = reverse_reproject(
            projected_eye_centre_camera_frame,
            self.initial_eye_z,
            self.focal_length,
        )
        eye_centre_camera_frame = np.append(
            projected_eye_centre_camera_frame_scaled, self.initial_eye_z
        ).reshape(3, 1)

        # Reconstructed selected gaze vectors and pupil positions by rejecting
        # those pointing away from projected eyecentre
        m = self.unprojected_gaze_vectors[0].shape[0]
        for i in range(m):
            gazes = [
                self.unprojected_gaze_vectors[0][i, :].reshape(3, 1),
                self.unprojected_gaze_vectors[1][i, :].reshape(3, 1),
            ]
            positions = [
                self.unprojected_3D_pupil_positions[0][i, :].reshape(3, 1),
                self.unprojected_3D_pupil_positions[1][i, :].reshape(3, 1),
            ]
            (
                selected_gaze,
                selected_position,
            ) = self.select_pupil_from_single_observation(
                gazes, positions, eye_centre_camera_frame
            )

            (
                self.selected_gazes,
                self.selected_pupil_positions,
            ) = self.stacking_from_nx1_to_mxn(
                [self.selected_gazes, self.selected_pupil_positions],
                [selected_gaze, selected_position],
                [3, 3],
            )

        radius_counter = []
        for i in range(self.selected_gazes.shape[0]):
            gaze = self.selected_gazes[i, :].reshape(1, 3)
            position = self.selected_pupil_positions[i, :].reshape(1, 3)

            # Before stacking, you must reshape (3,1) to (1,3)
            a_3Dfitting = np.vstack(
                (eye_centre_camera_frame.reshape(1, 3), position)
            )
            n_3Dfitting = np.vstack(
                (gaze, (position / np.linalg.norm(position)))
            )

            intersected_pupil_3D_centre = intersect(a_3Dfitting, n_3Dfitting)
            radius = np.linalg.norm(
                intersected_pupil_3D_centre - eye_centre_camera_frame
            )
            radius_counter.append(radius)
        aver_radius = np.mean(radius_counter)

        self.aver_eye_radius = aver_radius
        self.eye_centre = eye_centre_camera_frame
        return aver_radius, radius_counter

    def gen_consistent_pupil(self):
        # This function must be called after using
        # unproject_single_observation() to update surrent observation
        if (self.eye_centre is None) or (self.aver_eye_radius is None):
            raise TypeError(
                "Call estimate_eye_sphere() to initialize eye_centre\
                    and eye_radius first."
            )
        else:
            (
                selected_gaze,
                selected_position,
            ) = self.select_pupil_from_single_observation(
                [self.current_gaze_pos, self.current_gaze_neg],
                [
                    self.current_pupil_3Dcentre_pos,
                    self.current_pupil_3Dcentre_neg,
                ],
                self.eye_centre,
            )
            o = np.zeros((3, 1))

            try:
                d1, d2 = line_sphere_intersect(
                    self.eye_centre,
                    self.aver_eye_radius,
                    o,
                    selected_position / np.linalg.norm(selected_position),
                )
                new_position_min = o + min([d1, d2]) * (
                    selected_position / np.linalg.norm(selected_position)
                )
                new_position_max = o + max([d1, d2]) * (
                    selected_position / np.linalg.norm(selected_position)
                )
                new_radius_min = (
                    self.pupil_radius / selected_position[2, 0]
                ) * new_position_min[2, 0]
                new_radius_max = (
                    self.pupil_radius / selected_position[2, 0]
                ) * new_position_max[2, 0]

                new_gaze_min = new_position_min - self.eye_centre
                new_gaze_min = new_gaze_min / np.linalg.norm(new_gaze_min)

                new_gaze_max = new_position_max - self.eye_centre
                new_gaze_max = new_gaze_max / np.linalg.norm(new_gaze_max)
                self.pupil_new_position_min, self.pupil_new_position_max = (
                    new_position_min,
                    new_position_max,
                )
                self.pupil_new_radius_min, self.pupil_new_radius_max = (
                    new_radius_min,
                    new_radius_max,
                )
                self.pupil_new_gaze_min, self.pupil_new_gaze_max = (
                    new_gaze_min,
                    new_gaze_max,
                )
                consistence = True

            except (NoIntersectionError):
                # print("Cannot find line-sphere interception. Old pupil
                # parameters are used.")
                new_position_min, new_position_max = (
                    selected_position,
                    selected_position,
                )
                new_gaze_min, new_gaze_max = selected_gaze, selected_gaze
                new_radius_min, new_radius_max = (
                    self.pupil_radius,
                    self.pupil_radius,
                )
                consistence = False

            return (
                [new_position_min, new_position_max],
                [new_gaze_min, new_gaze_max],
                [new_radius_min, new_radius_max],
                consistence,
            )

    def plot_gaze_lines(self, ax):
        t = np.linspace(-1000, 1000, 1000)
        a = np.vstack((self.ellipse_centres, self.ellipse_centres))
        n = np.vstack(
            (
                self.unprojected_gaze_vectors[0][:, 0:2],
                self.unprojected_gaze_vectors[1][:, 0:2],
            )
        )  # [:, 0:2] takes only 2D projection

        for i in range(a.shape[0]):
            a_each = a[i, :]
            n_each = n[i, :]

            points = np.array(a_each).reshape(2, 1) + (
                t * n_each[0:2].reshape(2, 1)
            )
            ax.plot(points[0, :], points[1, :])
        ax.set_xlim(0, self.image_shape[1])
        ax.set_ylim(self.image_shape[0], 0)
        return ax

    def select_pupil_from_single_observation(
        self, gazes, positions, eye_centre_camera_frame
    ):
        # gazes : list ~ [gaze_vector_pos~(3,1), gaze_vector_neg~(3,1)]
        # positions : list~[pupil_position_pos~(3,1),pupil_position_neg~(3,1)]
        # eye_centre_camera_frame ~ numpy array~(3,1)

        selected_gaze = gazes[0]
        selected_position = positions[0]
        projected_centre = reproject(
            eye_centre_camera_frame, self.focal_length
        )
        projected_gaze = (
            reproject(selected_position + selected_gaze, self.focal_length)
            - projected_centre
        )
        projected_position = reproject(selected_position, self.focal_length)
        if (
            np.dot(projected_gaze.T, (projected_position - projected_centre))
            > 0
        ):
            return selected_gaze, selected_position
        else:
            return gazes[1], positions[1]

    @staticmethod
    def stacking_from_nx1_to_mxn(
        stacked_arrays_list, stacked_vectors_list, dims_list
    ):
        list_as_array = np.array([stacked_arrays_list])
        new_stacked_arrays_list = []
        if np.all(list_as_array == None):
            for stacked_array, stacked_vector, n in zip(
                stacked_arrays_list, stacked_vectors_list, dims_list
            ):
                stacked_array = stacked_vector.reshape(1, n)
                new_stacked_arrays_list.append(stacked_array)
        elif np.all(list_as_array != None):
            for stacked_array, stacked_vector, n in zip(
                stacked_arrays_list, stacked_vectors_list, dims_list
            ):
                stacked_array = np.vstack(
                    (stacked_array, stacked_vector.reshape(1, n))
                )
                new_stacked_arrays_list.append(stacked_array)
        elif np.any(list_as_array == None):
            print("Error list =\n", stacked_arrays_list)
            raise TypeError(
                "Some lists are initialized, some are not ('None').\
                    Error has happened!"
            )
        else:
            print("Error list =\n", stacked_arrays_list)
            raise TypeError("Unknown Error Occurred.")
        return new_stacked_arrays_list

    @staticmethod
    def convert_vec2angle31(n1):
        """
        Inputs:
            n1 = numpy array with shape (3,1)
        """
        assert n1.shape == (3, 1)
        n1 = n1 / np.linalg.norm(n1)
        n1_x, n1_y, n1_z_abs = n1[0, 0], n1[1, 0], np.abs(n1[2, 0])
        # x-augulation
        if n1_x > 0:
            x_angle = np.arctan(n1_z_abs / n1_x)
        else:
            x_angle = np.pi - np.arctan(n1_z_abs / np.abs(n1_x))
        # y-angulation
        if n1_y > 0:
            y_angle = np.arctan(n1_z_abs / n1_y)
        else:
            y_angle = np.pi - np.arctan(n1_z_abs / np.abs(n1_y))
        x_angle = np.rad2deg(x_angle)
        y_angle = np.rad2deg(y_angle)
        return [x_angle, y_angle]
