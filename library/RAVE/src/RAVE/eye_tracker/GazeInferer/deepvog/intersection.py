import numpy as np
from tqdm import tqdm
from multiprocessing import Queue, Process


class NoIntersectionError(Exception):
    pass


# vector (m,n) , m = number of examples, n = dimensionality
# a = coordinates of the vector
# n = orientation of the vector
def intersect(a, n):
    # default normalisation of vectors n
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    num_lines = a.shape[0]
    dim = a.shape[1]
    Identity = np.eye(dim)
    R_sum = 0
    q_sum = 0
    for i in range(num_lines):
        R = Identity - np.matmul(n[i].reshape(dim, 1), n[i].reshape(1, dim))

        q = np.matmul(R, a[i].reshape(dim, 1))
        q_sum = q_sum + q
        R_sum = R_sum + R
    p = np.matmul(np.linalg.inv(R_sum), q_sum)
    return p


def calc_distance(a, n, p):
    num_lines = a.shape[0]
    dim = a.shape[1]
    Identity = np.eye(dim)
    D_sum = 0
    for i in range(num_lines):
        D_1 = (a[i].reshape(dim, 1) - p.reshape(dim, 1)).T
        D_2 = Identity - np.matmul(n[i].reshape(dim, 1), n[i].reshape(1, dim))
        D_3 = D_1.T
        D = np.matmul(np.matmul(D_1, D_2), D_3)
        D_sum = D_sum + D
    D_sum = D_sum / num_lines
    return D_sum


def fit_ransac(
    a, n, max_iters=2000, samples_to_fit=20, min_distance=2000, nb_workers=5
):
    # https://stackoverflow.com/a/45829852
    q = Queue()
    processes = []
    rets = []
    for id in range(nb_workers):
        p = Process(
            target=_fit_ransac,
            args=(
                q,
                a,
                n,
                max_iters // nb_workers,
                samples_to_fit,
                min_distance,
                id,
            ),
        )
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get()  # will block
        rets.append(ret)
    for p in processes:
        p.join()
    best_distance, best_model = np.inf, None
    for ret in rets:
        distance, model = ret
        if distance < best_distance:
            best_distance = distance
            best_model = model
    return best_model


def _fit_ransac(queue, a, n, max_iters, samples_to_fit, min_distance, id):
    np.random.seed(id)
    num_lines = a.shape[0]

    best_model = None
    best_distance = min_distance
    for i in tqdm(range(max_iters), desc="Fitting", leave=False):
        # print("\rRANSAC: Currently {0}".format(i), flush=True)
        sampling_index = np.random.choice(
            num_lines, size=samples_to_fit, replace=False
        )
        a_sampled = a[sampling_index, :]
        n_sampled = n[sampling_index, :]
        model_sampled = intersect(a_sampled, n_sampled)
        sampled_distance = calc_distance(a, n, model_sampled)
        # print(sampled_distance)
        if sampled_distance > min_distance:
            continue
        else:
            if sampled_distance < best_distance:
                best_model = model_sampled
                best_distance = sampled_distance
    # if best_model is None:
    #     best_model = model_sampled
    queue.put((best_distance, best_model))


def line_sphere_intersect(c, r, o, line):
    # c = numpy array (3,1). Centre of the eyeball
    # r = scaler. Radius of the eyeball
    # o = numpy array (3,1). Origin of the line
    # line = numpy array (3,1). Directional unit vector of the line
    # return [d1, d2] : auxilary variables of the parametrised line x = o + dl
    # the closer one to the camera is chosen
    line = line / np.linalg.norm(line)
    delta = (
        np.square(np.dot(line.T, (o - c)))
        - np.dot((o - c).T, (o - c))
        + np.square(r)
    )
    if delta < 0:
        raise NoIntersectionError
    else:
        d1 = -np.dot(line.T, (o - c)) + np.sqrt(delta)
        d2 = -np.dot(line.T, (o - c)) - np.sqrt(delta)
    return [d1, d2]
