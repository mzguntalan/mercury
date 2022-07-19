from jax import numpy as jnp, random, vmap, jit


@jit
def rotation_matrix(angles):
    angles = jnp.reshape(angles, [3, 1])
    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)

    cos_yaw, cos_pitch, cos_roll = cos_angles[0], cos_angles[1], cos_angles[2]
    sin_yaw, sin_pitch, sin_roll = sin_angles[0], sin_angles[1], sin_angles[2]

    zero = jnp.zeros([1], "float32")
    one = jnp.ones([1], "float32")

    rot_yaw = jnp.reshape(
        jnp.concatenate(
            [cos_yaw, -sin_yaw, zero, sin_yaw, cos_yaw, zero, zero, zero, one], axis=-1
        ),
        [3, 3],
    )

    rot_pitch = jnp.reshape(
        jnp.concatenate(
            [cos_pitch, zero, sin_pitch, zero, one, zero, -sin_pitch, zero, cos_pitch],
            axis=-1,
        ),
        [3, 3],
    )

    rot_roll = jnp.reshape(
        jnp.concatenate(
            [one, zero, zero, zero, cos_roll, -sin_roll, zero, sin_roll, cos_roll],
            axis=-1,
        ),
        [3, 3],
    )

    rot = rot_yaw @ rot_pitch @ rot_roll
    return rot


@jit
def rotate(point_cloud, angles):
    angles = jnp.reshape(angles, [3, 1])
    rot = rotation_matrix(angles)
    rotated = point_cloud @ rot.T

    return rotated


@jit
def rotate_by_matrix(point_cloud, rot):
    point_cloud = jnp.expand_dims(point_cloud, axis=-1)
    rotated = rot @ point_cloud

    rotated = jnp.squeeze(rotated, axis=-1)
    return rotated


batched_rotation_matrix = jit(vmap(rotation_matrix, in_axes=0))


@jit
def randomly_rotate(key, point_cloud):
    angles = random.uniform(key, [3, 1], "float32", 0, 2 * jnp.pi)

    return rotate(point_cloud, angles)


batched_rotate = jit(vmap(rotate, in_axes=(0, 0), out_axes=(0)))


@jit
def batched_randomly_rotate(key, point_clouds):
    angles = random.uniform(
        key, [point_clouds.shape[0], 3, 1], "float32", 0, 2 * jnp.pi
    )

    return batched_rotate(point_clouds, angles)
