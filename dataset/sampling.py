import numpy as np
import bisect




def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_sample(points, npoint):
    """
    :param points:
    :param npoint:
    :return:
    """
    idx = np.random.choice(len(points), size=npoint, replace=False)
    return points[idx, :]


def farthest_point_sample(points, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = points[centroids.astype(np.int32)]
    return points


def farthest_point_sample_with_index(points, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    index = list()
    for i in range(npoint):
        centroids[i] = farthest
        index.append(farthest)
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = points[centroids.astype(np.int32)]
    return points, index


def random_sample_with_index(points, npoint):
    idx = np.random.choice(len(points), size=npoint, replace=False)
    return points[idx, :], idx


def sample_points(objects, num_points):
    points = []
    triangles = []

    for obj in objects:
        curr_points = []
        curr_triangles = []

        areas = np.cross(obj[:, 1] - obj[:, 0], obj[:, 2] - obj[:, 0])
        areas = np.linalg.norm(areas, axis=1) / 2.0
        prefix_sum = np.cumsum(areas)
        total_area = prefix_sum[-1]

        for _ in range(num_points):
            # pick random triangle based on area
            rand = np.random.uniform(high=total_area)
            if rand >= total_area:
                idx = len(obj) - 1  # can happen due to floating point rounding
            else:
                idx = bisect.bisect_right(prefix_sum, rand)

            # pick random point in triangle
            a, b, c = obj[idx]
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 >= 1.0:
                r1 = 1 - r1
                r2 = 1 - r2
            p = a + r1 * (b - a) + r2 * (c - a)

            curr_points.append(p)
            curr_triangles.append(obj[idx])

        points.append(curr_points)
        triangles.append(curr_triangles)

    points = np.array(points)
    triangles = np.array(triangles)

    return points, triangles
