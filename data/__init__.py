import h5py
from jax import jit, numpy as jnp, vmap, random
import jax
import json
from itertools import chain
from rotations import batched_randomly_rotate
import numpy as np


def h5_name(name):
    return f"./data/shapenet/train{name}.h5"


def json_name(name):
    return f"./data/shapenet/train{name}_id2name.json"


def filter_dataset(dataset, categories, category="airplane"):
    mask = [i == category for i in categories]
    return dataset[
        mask,
    ]


def datasets(names, category="airplane"):
    dataset_list = []
    for name in names:
        with open(json_name(name)) as f:
            categories = json.load(f)

        f = h5py.File(h5_name(name), "r")
        dataset = jnp.array(f["data"][()])
        f.close()

        dataset = filter_dataset(dataset, categories, category)

        dataset_list.append(dataset)

    dataset = jnp.concatenate(dataset_list, axis=0)

    return dataset


def prepare_batch(key, batch):
    # put to gpu
    batch = jax.device_put(batch, jax.devices()[0])
    batch = batched_mean_center(batch)

    batch = batched_randomly_rotate(key, batch)

    return batch


@jit
def mean_center(point_cloud):
    mean_point = jnp.mean(point_cloud[:, :3], axis=0)
    point_cloud = point_cloud - jnp.expand_dims(mean_point, axis=-2)
    return point_cloud


batched_mean_center = jit(vmap(mean_center, in_axes=0))


def batch_dataset(key, batch_size, dataset):
    permute_key, prep_key = random.split(key)
    dataset = random.permutation(permute_key, dataset, axis=0)
    num_batches = dataset.shape[0] // batch_size
    dataset = dataset[: num_batches * batch_size]
    dataset = jnp.reshape(dataset, [num_batches, batch_size, -1, 3])
    for batch in dataset:
        yield prepare_batch(prep_key, batch)


_IDs = np.reshape(np.arange(0, 17 + 1), [-1, 2])


def dataset_batches(key, batch_size, category="airplane"):
    generators = []
    for a, b in _IDs:
        names = [str(a), str(b)]
        key, subkey = random.split(key)
        generators.append(batch_dataset(subkey, batch_size, datasets(names, category)))

    return chain(*generators)
