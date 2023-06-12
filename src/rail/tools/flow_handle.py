"""Rail-specific data management"""

import os
import tables_io

from rail.core.data import ModelHandle

class FlowDict(dict):
    """
    A specialized dict to keep track of individual flow objects: this is just a dict these additional features

    1. Keys are paths
    2. Values are flow objects, this is checked at runtime.
    3. There is a read(path, force=False) method that reads a flow object and inserts it into the dictionary
    4. There is a single static instance of this class
    """

    def __setitem__(self, key, value):
        """ Add a key-value pair, and check to make sure that the value is a `Flow` object """
        from pzflow import Flow
        if not isinstance(value, Flow):  #pragma: no cover
            raise TypeError(f"Only values of type Flow can be added to a FlowFactory, not {type(value)}")
        return dict.__setitem__(self, key, value)

    def read(self, path, force=False):
        """ Read a `Flow` object from disk and add it to this dictionary """
        from pzflow import Flow
        if force or path not in self:
            flow = Flow(file=path)
            self.__setitem__(path, flow)
            return flow
        return self[path]  #pragma: no cover


class FlowHandle(ModelHandle):
    """
    A wrapper around a file that describes a PZFlow object
    """
    flow_factory = FlowDict()

    suffix = 'pkl'

    @classmethod
    def _open(cls, path, **kwargs):  #pylint: disable=unused-argument
        if kwargs.get('mode', 'r') == 'w':  #pragma: no cover
            raise NotImplementedError("Use FlowHandle.write(), not FlowHandle.open(mode='w')")
        return cls.flow_factory.read(path)

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return cls.flow_factory.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        return data.save(path)

