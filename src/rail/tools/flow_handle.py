"""Rail-specific data management"""

from typing import Any
import os
import tables_io
from pzflow import Flow

from rail.core.data import ModelHandle, ModelLike


def flow_model_read(modelfile: str) -> ModelLike:
    """Default function to read model files, simply used pickle.load"""
    flow = Flow(file=modelfile)
    return flow


def flow_model_write(model: ModelLike, path: str) -> None:
    """Write the model, this default implementation uses pickle"""
    model.save(path)


class FlowHandle(ModelHandle):
    """
    A wrapper around a file that describes a PZFlow object
    """
    default_model_read = flow_model_read
    default_model_write = flow_model_write

    suffix = 'pkl'

    @classmethod
    def _open(cls, path, **kwargs):  #pylint: disable=unused-argument
        if kwargs.get('mode', 'r') == 'w':  #pragma: no cover
            raise NotImplementedError("Use FlowHandle.write(), not FlowHandle.open(mode='w')")
        return cls.read(path)

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> ModelLike:
        """Read and return the data from the associated file"""
        return flow_model_read(path)

    @classmethod
    def _write(cls, data: ModelLike, path: str, **kwargs: Any) -> None:
        """Write the data to the associated file"""
        flow_model_write(data, path)
