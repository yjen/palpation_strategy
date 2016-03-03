import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
DEBUG = True
DEBUG2 = True

# generate test data test data


def generate_test_data():
    plane_param_noise = 1
    surface_noise = 0.05
    probe_offset_loc = 2.0
    probe_offset_noise = 0.0001
    approach_angle_noise = 0.5

    if DEBUG:
        fig = plt.figure()

    num_samples = 100
    # generate random plane parameters
    a = np.random.normal(scale=plane_param_noise, size=4)
    a /= np.linalg.norm(a[:3])

    # generate random points on a random plane
    points = np.ones([4, num_samples])
    points[0:2, :] = np.random.uniform(-1.0, 1.0, size=[2, num_samples])
    points[2, :] = np.array([-1.0*a[0]/a[2], -1.0*a[1]/a[2], -1.0*a[3]/a[2]]).dot(points[0:3, :])

    if DEBUG:
        fig.suptitle("Stages of generating test data")
        ax = fig.add_subplot(221, projection='3d', aspect='auto')
        ax.scatter(points[0, :], points[1, :], points[2, :])
        ax.set_title('Random points on random plane')
        # set_axes_equal(ax)

    # add gaussian noise to plane
    noisy_points = points.copy()
    noisy_points[:3, :] += np.random.normal(scale=surface_noise, size=[3, num_samples])

    if DEBUG:
        ax = fig.add_subplot(222, projection='3d', aspect='auto')
        ax.scatter(noisy_points[0, :], noisy_points[1, :], noisy_points[2, :])
        ax.set_title('Noisy points on a plane')
        set_axes_equal(ax)

    # create a probe offset
    d = np.random.normal(loc=probe_offset_loc, scale=probe_offset_noise)
    # create a gaussian approach angle centered around plane normal
    approach_angles = np.random.normal(scale = approach_angle_noise, size=[3, num_samples]) + np.transpose(np.tile(a[0:3], (num_samples, 1)))
    approach_angles = approach_angles/np.linalg.norm(approach_angles, axis=0)

    if DEBUG:
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(noisy_points[0, :], noisy_points[1, :], noisy_points[2, :])
        ax.quiver(noisy_points[0], noisy_points[1], noisy_points[2], approach_angles[0], approach_angles[1], approach_angles[2], length=d, arrow_length_ratio=0.05)
        # set_axes_equal(ax)
        ax.set_title('Noisy points and random probe approach angles')

    # apply probe offset to points to generate final test points
    test_points = noisy_points[:3, :].copy()
    test_points -= d*approach_angles

    if DEBUG:
        ax = fig.add_subplot(224, projection='3d')
        ax.scatter(test_points[0, :], test_points[1, :], noisy_points[2, :])
        ax.set_title('Final Generated Test Points')
        # set_axes_equal(ax)

    if DEBUG:
        plt.show()

    return test_points, approach_angles, d, a


def objective(X, R, d):
    matrix = X+d*R
    return min(np.linalg.eig(matrix.dot(np.transpose(matrix)))[0])



def compute_registration(test_points, approach_angles, min_offset, max_offset):
    num_samples = np.shape(test_points)[1]
    X = np.ones([4, num_samples])
    R = np.zeros([4, num_samples])
    X[:3,:] = test_points
    R[:3,:] = approach_angles

    d = 0
    if DEBUG2:
        x = np.arange(0.0, 5.0, 0.0001)
        y = [objective(X, R, a) for a in x]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title('Objective as a function of probe offset')
        plt.show()
        print "Estimated probe offset: ", x[np.argmin(y)]


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Note: Copied from:
    http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]; x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]; y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]; z_mean = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


if __name__ == '__main__':
    test_points, approach_angles, d, a = generate_test_data()
    compute_registration(test_points, approach_angles, 0, 1)
    print "True probe offset: ", d
