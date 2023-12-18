import pybullet as p
import pybullet_data
import numpy as np
import bayes3d as b
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL
import os
import jax.numpy as jnp


# pybullet helper/util functions
def object_pose_in_camera_frame(object_id, view_matrix):
    object_pos, object_orn = p.getBasePositionAndOrientation(object_id) # world frame
    world2cam = np.array(view_matrix).reshape([4,4]).T # world --> cam 
    object_transform_matrix = np.eye(4)
    object_transform_matrix[:3, :3] = np.reshape(p.getMatrixFromQuaternion(object_orn), (3, 3))
    object_transform_matrix[:3, 3] = object_pos
    cam2world = world2cam @ object_transform_matrix
    cam2world[1:3] *= -1
    return cam2world

def cam_pose_from_view_matrix(view_matrix):
    # cam2world
    world2cam = np.array(view_matrix).reshape([4,4]).T
    world2cam[1:3] *= -1
    cam2world  = np.linalg.inv(world2cam)
    return cam2world

def view_matrix_from_cam_pose(cam_pose):
    world2cam = np.linalg.inv(cam_pose)
    world2cam[1:3] *= -1
    return tuple(world2cam.T.reshape(world2cam.size))


######################################



# Initialize the PyBullet physics simulation
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# initialize list to store object_ids
object_ids = []


# Sim params
sim_time = 10 # simulation time
fps_data = 60 # ground truth saving FPS
fps = 4*fps_data # physics engine FPS --> make this a multiple of fps_data

# friction_coefficient = 0.75 # Adjust this value as needed
lateral_friction_coefficient = 0.5
spinning_friction_coefficient = 0.1

gravity = -9.81

# camera intrisics params
width = 480
height = 360
field_of_view = 60
near = 0.1
far = 100

# camera_pose (define it explicitly or leave it as None)
# If cam pose is None, modify it based on computeViewMatrixFromYawPitchRoll below
cam_pose = None


box_mass = 1

box_position = [-4.75, 0, 2.501]


box_start_velocity = [5, 0, 0]
mesh_scale = np.array([1,1,1]) * 0.2
box_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"), meshScale=mesh_scale)
box_id = p.createMultiBody(box_mass, box_shape, basePosition=box_position)

# Set initial angular velocity
# For example, rotating around the z-axis
#initialAngularVelocity = [0, 5, 0]  # Adjust as needed [0,0,0]
initialAngularVelocity = [0, 5, 0]

p.resetBaseVelocity(box_id, box_start_velocity, angularVelocity=initialAngularVelocity) # rot vel

unevenInertia = [10, 1, 1] 
#unevenInertia = [1, 1, 1] 

p.changeDynamics(box_id, -1, restitution = 1, localInertiaDiagonal=unevenInertia)
object_ids.append(box_id)



p.setGravity(0, 0, gravity)
floor_id = p.loadURDF("plane.urdf")
p.changeDynamics(floor_id, -1, lateralFriction=lateral_friction_coefficient, spinningFriction=spinning_friction_coefficient)


rgbs = []
depths = []
gt_poses = []

if cam_pose is None:
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=6, yaw=0, pitch=-5, roll=0,
                                                        upAxisIndex=2)
    cam_pose = cam_pose_from_view_matrix(view_matrix)
else:
    view_matrix = view_matrix_from_cam_pose(cam_pose)
proj_matrix = p.computeProjectionMatrixFOV(fov=field_of_view, aspect=float(width)/height, nearVal=near, farVal=far)
floor_cam_pose = object_pose_in_camera_frame(floor_id, view_matrix)

num_timesteps = int(fps * sim_time)
save_data_frequency = int(fps / fps_data)
p.setTimeStep(1.0/fps)
for i in range(num_timesteps):
    p.stepSimulation()
    if i%save_data_frequency == 0:
    
        (_, _, px, d, _) = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]
        rgbs.append(rgb_array)

        pose_arr = []
        for i,obj_id in enumerate(object_ids):
            pose_arr.append(object_pose_in_camera_frame(obj_id, view_matrix))
        gt_poses.append(pose_arr)

gt_poses = jnp.array(gt_poses)
rgbs = jnp.array(rgbs)

p.disconnect()

focal = (height/2) / np.tan(np.deg2rad(field_of_view) / 2.0)
intrinsics = b.Intrinsics(
    height,
    width,
    focal,
    focal,
    width/2,
    height/2,
    near,
    far
)

def save_video(frames, file_name, framerate=30):
    """
    frames: PIL Image OR a list of N np.arrays (H x W x 3)
    framerate: frames per second
    """
    if type(frames[0]) == PIL.Image.Image:
      frames = [np.array(frames[i]) for i in range(len(frames))]
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=True)
    mywriter = animation.FFMpegWriter(fps=60)
    anim.save(file_name, writer=mywriter)


save_video(rgbs, 'test_bullet_rendering_roll_unfair.mp4')


np.save('gt_poses_roll_unfair', gt_poses)



