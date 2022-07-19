from jax import numpy as jnp, vmap, jit
import numpy as np

from rotations import rotate


@jit
def canonicalize_ellipsoid(point_cloud: jnp.ndarray) -> jnp.ndarray:
    empirical_cov = point_cloud.T @ point_cloud

    eigenvalues, eigenvectors = jnp.linalg.eigh(empirical_cov)
    indices = jnp.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, indices]
    canonicalized = point_cloud @ eigenvectors_sorted

    signs = jnp.sign(canonicalized)
    index_of_max = jnp.argmax(jnp.abs(canonicalized), axis=0)[None, :]
    sign = jnp.take_along_axis(signs, index_of_max, axis=0)
    canonicalized = canonicalized * sign

    return canonicalized


@jit
def householder_transform(
    normal_vector: jnp.ndarray, point_cloud: jnp.ndarray
) -> jnp.ndarray:
    normal_unit = normal_vector / jnp.linalg.norm(normal_vector, 2)
    normal_unit = jnp.expand_dims(normal_unit, axis=-1)

    reflection = 2 * (normal_unit @ normal_unit.T)
    reflection = jnp.identity(reflection.shape[0], dtype="float32") - reflection
    reflected = reflection @ jnp.expand_dims(point_cloud, axis=-1)
    reflected = jnp.squeeze(reflected, axis=-1)

    return reflected


reflect_point_cloud_for_each_unit_vector = vmap(
    householder_transform, in_axes=[0, None]
)


@jit
def canonicalize_symmetry(point_cloud: jnp.ndarray) -> jnp.ndarray:
    unit_vectors = jnp.identity(3, dtype="float32")
    target_vectors = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"
    )

    reflected = reflect_point_cloud_for_each_unit_vector(
        unit_vectors, point_cloud
    )  # [u, s, d]
    stacked_point_cloud = jnp.stack([point_cloud, point_cloud, point_cloud], axis=0)

    distances = chamfer_distance(reflected, stacked_point_cloud)
    arg_min = jnp.argmin(distances, axis=0)

    rotate_about_this_axis = target_vectors[arg_min]
    rotation_angles = jnp.pi / 2 * rotate_about_this_axis
    point_cloud = rotate(point_cloud, rotation_angles)

    return point_cloud


@jit
def canonicalize(point_cloud):
    return canonicalize_symmetry(canonicalize_ellipsoid(point_cloud))


def canonicalize_to_reference(point_cloud, canonicalized_reference):
    angles_x = [
        np.array([0.0, 0.0, 0.0], "float32"),
        np.array([jnp.pi, 0.0, 0.0], "float32"),
    ]
    angles_y = [
        np.array([0.0, 0.0, 0.0], "float32"),
        np.array([0.0, jnp.pi, 0.0], "float32"),
        np.array(
            [0.0, 0.5 * jnp.pi, 0.0], "float32"
        ),  # more freedom since this will rotate normal the plane of symmetry
    ]
    angles_z = [
        np.array([0.0, 0.0, 0.0], "float32"),
        np.array([0.0, 0.0, jnp.pi], "float32"),
    ]

    canon = canonicalize(point_cloud)

    rot_angle = angles_x[0] + angles_y[0] + angles_z[0]
    rot_canon = rotate(canon, rot_angle)
    min_distance = chamfer_distance(rot_canon, canonicalized_reference)
    canon_final = canon

    for rot_x in angles_x:
        for rot_y in angles_y:
            for rot_z in angles_z:
                rot_angle = rot_x + rot_y + rot_z
                rot_canon = rotate(canon, rot_angle)
                distance = chamfer_distance(rot_canon, canonicalized_reference)
                if distance < min_distance:
                    min_distance = distance
                    canon_final = rot_canon

    return canon_final


@jit
def chamfer_distance(point_set_a, point_set_b):
    difference = jnp.subtract(
        jnp.expand_dims(point_set_a, axis=-2),
        jnp.expand_dims(point_set_b, axis=-3),
    )

    squared_distances = jnp.einsum("...i,...i->...", difference, difference)
    minimum_squared_distance_from_a_to_b = jnp.min(squared_distances, axis=-1)
    minimum_squared_distance_from_b_to_a = jnp.min(squared_distances, axis=-2)

    return jnp.add(
        jnp.mean(minimum_squared_distance_from_a_to_b, axis=-1),
        jnp.mean(minimum_squared_distance_from_b_to_a, axis=-1),
    )
