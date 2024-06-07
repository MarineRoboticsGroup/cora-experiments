import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# Load the .tum file
def load_tum_file(file_path):
    data = np.loadtxt(file_path)
    return data[:, 1:4], data[:, 4:8]  # positions and orientations (quaternions)

# Convert quaternion to Euler angles (optional for visualization)
def quaternion_to_euler(quaternion):
    r = R.from_quat(quaternion)
    return r.as_euler('xyz', degrees=True)

# Convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(quaternion):
    r = R.from_quat(quaternion)
    return r.as_matrix()

# Parameters
# /home/alan/range-only-slam-mission-control/cora-experiments/data/mrclam6/cora_0.tum
file_path = '/home/alan/range-only-slam-mission-control/cora-experiments/data/mrclam6/cora_0.tum'
# gtsam_gt_pose_gt_landmarks_A.tum
# file_path = '/home/alan/range-only-slam-mission-control/cora-experiments/data/mrclam6/gtsam_gt_pose_gt_landmarks_A.tum'

# /tmp/plaza1/cora_0.tum
file_path = '/tmp/plaza1/cora_0.tum'


positions, quaternions = load_tum_file(file_path)

# Prepare the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(min(positions[:, 0]), max(positions[:, 0]))
ax.set_ylim(min(positions[:, 1]), max(positions[:, 1]))
ax.set_zlim(min(positions[:, 2]), max(positions[:, 2]))

# set axes to be equal (square)
ax.set_box_aspect([1,1,1])

# Plot elements
line, = ax.plot([], [], [], 'b-', label='Trajectory')
point, = ax.plot([], [], [], 'ro')

quiver_x = ax.quiver([], [], [], [], [], [], color='r', length=0.1, normalize=True)
quiver_y = ax.quiver([], [], [], [], [], [], color='g', length=0.1, normalize=True)
# quiver_z = ax.quiver([], [], [], [], [], [], color='b', length=0.1, normalize=True)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    # return line, point, quiver_x, quiver_y, quiver_z

def update(num, positions, quaternions, line, point):
    global quiver_x
    global quiver_y
    # global quiver_z
    quiver_x.remove()
    quiver_y.remove()
    # quiver_z.remove()

    line.set_data(positions[:num, 0], positions[:num, 1])
    line.set_3d_properties(positions[:num, 2])
    point.set_data(positions[num-1:num, 0], positions[num-1:num, 1])
    point.set_3d_properties(positions[num-1:num, 2])
    
    # Update quivers
    start = positions[num-1]
    quaternion = quaternions[num-1]
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    direction_x = rotation_matrix[:, 0]  # x-axis
    direction_y = rotation_matrix[:, 1]  # y-axis
    direction_z = rotation_matrix[:, 2]  # z-axis

    # check that z is always [0,0,1]
    assert np.allclose(direction_z, [0, 0, 1])

    # check that x and y are orthogonal
    assert np.allclose(np.dot(direction_x, direction_y), 0)

    # print(f"Direction x: {direction_x}")
    # print(f"Direction y: {direction_y}")

    # update quivers using set_UVC
    quiver_x = ax.quiver(*start, *direction_x, color='r', length=5.5)
    quiver_y = ax.quiver(*start, *direction_y, color='g', length=5.5)
    # quiver_z = ax.quiver(*start, *direction_z, color='b', length=0.05)
    

ani = FuncAnimation(fig, update, frames=len(positions), fargs=[positions, quaternions, line, point], init_func=init, interval=10, blit=False)

# set axes to be equal (square)
ax.set_box_aspect([1,1,1])
plt.legend()
plt.show()
