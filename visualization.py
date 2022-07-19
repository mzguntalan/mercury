from jax import numpy as jnp, jit
import numpy as np


@jit
def shading_1(point_cloud):
    point_of_darkness = jnp.array([[-1.5, 1.5, -1.5]], "float32")
    distances = point_cloud - point_of_darkness
    distances = jnp.linalg.norm(distances, axis=-1)

    return distances


@jit
def shading_2(point_cloud):
    point_of_darkness = jnp.array([[0, 1.5, -1.5]], "float32")
    distances = point_cloud - point_of_darkness
    distances = jnp.linalg.norm(distances, axis=-1)

    return distances


@jit
def adjust_frame(A):
    adjustment = np.array([[3.0, 0.0, 0.0]], "float32")
    adj = A + adjustment

    return adj
