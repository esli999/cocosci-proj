## Tracker from Bayes3D

import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os

# Can be helpful for debugging:
# jax.config.update('jax_enable_checks', True) 


# camera intrisics params
width = 480
height = 360
field_of_view = 60
near = 0.1
far = 100

# FOV is based on height and it scales proportionately to width
focal = (height/2) / np.tan(np.deg2rad(field_of_view) / 2.0)
intrinsics = b.Intrinsics(
    height,
    width,
    focal,
    focal,
    width/2,
    height/2,
    near,
    far)

# scale intrinsics for gpu memory
SCALE = 0.2
intrinsics = b.scale_camera_parameters(intrinsics, SCALE)
b.setup_renderer(intrinsics)
b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"))


poses = np.load('gt_poses.npy')[:,0,:,:]
observed_images = np.load('depth_seq.npy')


translation_deltas = b.utils.make_translation_grid_enumeration(-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 5, 5, 5)
rotation_deltas = jax.vmap(lambda key: b.distributions.gaussian_vmf_zero_mean(key, 0.00001, 800.0))(
    jax.random.split(jax.random.PRNGKey(30), 100)
)

likelihood = jax.vmap(b.threedp3_likelihood_old, in_axes=(None, 0, None, None, None, None, None))

def update_pose_estimate(pose_estimate, gt_image):
    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, translation_deltas)
    rendered_images = jax.vmap(b.RENDERER.render, in_axes=(0, None))(proposals[:,None, ...], jnp.array([0]))
    weights_new = likelihood(gt_image, rendered_images, 0.05, 0.1, 10**3, 0.1, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    rendered_images = jax.vmap(b.RENDERER.render, in_axes=(0, None))(proposals[:, None, ...], jnp.array([0]))
    weights_new = likelihood(gt_image, rendered_images, 0.05, 0.1, 10**3, 0.1, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

inference_program = jax.jit(lambda p,x: jax.lax.scan(update_pose_estimate, p,x)[1])
inferred_poses = inference_program(poses[0], observed_images)

start = time.time()
pose_estimates_over_time = inference_program(poses[0], observed_images)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", poses.shape[0] / (end - start))

rerendered_images = b.RENDERER.render_many(pose_estimates_over_time[:, None, ...], jnp.array([0]))

viz_images = [
    b.viz.multi_panel(
        [
            b.viz.scale_image(b.viz.get_depth_image(d[:,:,2]), 3),
            b.viz.scale_image(b.viz.get_depth_image(r[:,:,2]), 3)
            ],
        labels=["Observed", "Rerendered"],
        label_fontsize=20
    )
    for (r, d) in zip(rerendered_images, observed_images)
]
b.make_gif_from_pil_images(viz_images, "test_tracker_basic.gif")



#from IPython import embed; embed()

