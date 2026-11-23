import glob
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Callable

import equinox as eqx
import jax.numpy as jnp


def _fields(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def _to_json(obj):
    if is_dataclass(obj):
        return {"type": type(obj).__name__, **_fields(obj)}
    if callable(obj):
        return obj.__name__
    return jnp.asarray(obj).tolist()


class SavedParams:
    def _to_json_str(self, **kwargs) -> str:
        return json.dumps(_fields(self), default=_to_json, sort_keys=True, **kwargs)

    @property
    def params_hash(self) -> str:
        return hashlib.sha1(self._to_json_str().encode()).hexdigest()[:7]

    def find_dir(self, root: str) -> str | None:
        matches = glob.glob(f"{root}/*_{self.params_hash}")
        return matches[0] if matches else None

    def make_dir(self, root: str, write: Callable[[str], None]) -> str:
        os.makedirs(root, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=root, prefix=".")
        try:
            with open(f"{tmp_dir}/params.json", "w") as f:
                f.write(self._to_json_str(indent=2))
            write(tmp_dir)
        except BaseException:
            shutil.rmtree(tmp_dir)
            raise

        existing_dir = self.find_dir(root)
        if existing_dir is not None:
            shutil.rmtree(tmp_dir)
            return existing_dir

        path = f"{root}/{datetime.now():%Y%m%d_T%H-%M-%S}_{self.params_hash}"
        os.rename(tmp_dir, path)
        return path


def save_fit(model_dir: str, model: eqx.Module, **arrays):
    eqx.tree_serialise_leaves(f"{model_dir}/model.eqx", model)
    for name, array in arrays.items():
        jnp.save(f"{model_dir}/{name}.npy", array)


def load_fit(model_dir: str, like_model: eqx.Module, *names: str) -> tuple:
    model = eqx.tree_deserialise_leaves(f"{model_dir}/model.eqx", like_model)
    return model, *(jnp.load(f"{model_dir}/{name}.npy") for name in names)


def load_or_fit(
    params: SavedParams,
    root: str,
    model: eqx.Module,
    fit_fn: Callable[[], tuple],
    *names: str,
) -> tuple:
    model_dir = params.find_dir(root)
    if model_dir is not None:
        return model_dir, *load_fit(model_dir, model, *names)

    model, *arrays = fit_fn()
    model_dir = params.make_dir(
        root, lambda path: save_fit(path, model, **dict(zip(names, arrays)))
    )

    return model_dir, model, *arrays
