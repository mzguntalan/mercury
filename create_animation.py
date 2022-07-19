from jax import numpy as jnp, random, jit
from jax.random import KeyArray
from data import dataset_batches
from canonicalization import canonicalize, canonicalize_to_reference
from tqdm import tqdm
from rotations import rotate
from visualization import adjust_frame, shading_1, shading_2
import matplotlib.pyplot as plt
import imageio
import os
from typing import Optional, Union
from functools import partial


def main():
    key = random.PRNGKey(0)
    for obj in [
        "airplane",
        "telephone",
        "rifle",
        "table",
        "chair",
        "sofa",
        "guitar",
        "bench",
        "speaker",
        "mug",
        "cabinet",
    ]:
        batched_create_animation_for_object(key, "demo", obj, 6, 5, 30)


@partial(jit, static_argnums=[1, 2])
def rotations_for_animation(
    key: KeyArray, num_rotations: int, num_frames_between_rotations: int
) -> jnp.ndarray:
    key, rot_key = random.split(key)
    rotations = random.uniform(rot_key, [num_rotations, 3, 1], "float32", 0, 2 * jnp.pi)
    rotations = jnp.concatenate([rotations, rotations[:1]], axis=0)  # loop back

    rotations = jnp.concatenate(
        [
            jnp.linspace(rotations[i], rotations[i + 1], num_frames_between_rotations)
            for i in range(rotations.shape[0] - 1)
        ]
    )  # create in betweens for adjacent pairs of rotations

    return rotations


def create_animation_for_object(
    key: KeyArray,
    point_cloud: jnp.ndarray,
    reference: jnp.ndarray,
    num_rotations: int,
    num_frames_between_rotations: int,
    prefix_filename: str,
) -> None:
    rotations_for_object = rotations_for_animation(
        key, num_rotations, num_frames_between_rotations
    )

    for f_id, rotation in tqdm(
        enumerate(rotations_for_object), total=rotations_for_object.shape[0]
    ):
        rotated_object = rotate(point_cloud, rotation)
        canonicalized_object = canonicalize_to_reference(rotated_object, reference)
        plot_2_point_clouds(
            rotated_object,
            canonicalized_object,
            _filename_temp_png(prefix_filename, f_id),
        )

    # gif
    with imageio.get_writer(filename_animation(prefix_filename), mode="I") as writer:
        for i in tqdm(
            range(0, rotations_for_object.shape[0]), total=rotations_for_object.shape[0]
        ):
            target_name = f"{_filename_temp_png(prefix_filename, i)}"
            image = imageio.imread(target_name)
            writer.append_data(image)
            os.remove(target_name)


def batched_create_animation_for_object(
    key: KeyArray,
    prefix_filename: str,
    object_class: str,
    num_objects_to_create_animation_for: int,
    num_rotations: int,
    num_frames_between_rotations: int,
) -> None:
    key, data_key = random.split(key, num=2)
    batches = dataset_batches(
        data_key, num_objects_to_create_animation_for, object_class
    )

    batch = next(batches)
    # let the first object be the reference

    reference = canonicalize(batch[0])

    for i, point_cloud in enumerate(batch):
        key, animation_key = random.split(key)
        create_animation_for_object(
            animation_key,
            point_cloud,
            reference,
            num_rotations,
            num_frames_between_rotations,
            f"{prefix_filename}-{object_class}-{i}",
        )


def plot_2_point_clouds(
    point_cloud_1: jnp.ndarray,
    point_cloud_2: jnp.ndarray,
    filename: Union[str, None] = None,
    dpi: int = 72,
) -> str:
    point_cloud_2 = adjust_frame(point_cloud_2)

    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    point_cloud = jnp.concatenate([point_cloud_1, point_cloud_2], axis=0)
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.set_ylim([-1.5, 1.5])
    ax.set_xlim([-1.5, 4.5])
    ax.set_zlim([-1.5, 1.5])

    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.set_zlabel("$Z$", fontsize=20)
    ax.set_box_aspect([2, 1, 1])

    ax.scatter(
        x,
        y,
        z,
        c=(
            jnp.concatenate(
                [
                    shading_1(point_cloud[: point_cloud_1.shape[0]]),
                    shading_2(point_cloud[point_cloud_1.shape[0] :]),
                ],
                axis=0,
            )
        ),
        cmap=plt.magma(),
    )

    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close()
    else:
        plt.show()
        plt.close()

    return filename


def _filename_temp_png(name: str, idx: int) -> str:
    return f"./demo/{name}-{idx}.png"


def filename_animation(name: str) -> str:
    return f"./demo/{name}-animation.gif"


if __name__ == "__main__":
    main()
