from abc import ABC, abstractmethod
from frozendict import frozendict
from sklearn.utils import check_random_state
from .functions import _Function

class Input(ABC):

    def __init__(self, name, value, type=None):
        self.name = name
        self.value = value
        self.type = type or type(value)


def ArgSampler():

    def __init__(self, arg_range_dict, kwarg_range_dict):
        self.arg_range_dict = arg_range_dict
        self.kwarg_range_dict = kwarg_range_dict
    
    def sample(self, random_state=None, n=1, seed=0):
        samples = list()

        random_state = random_state or check_random_state(seed)
        for i in range(n):
            args = {
                arg: random_state.uniform(*arg_range)
                for arg, arg_range
                in self.arg_range_dict.items()
            }
            kwargs = {
                kwarg: random_state.uniform(*kwarg_range)
                for kwarg, kwarg_range
                in self.arg_range_dict.items()
            }

            samples.append(
                {**args, **kwargs}
            )
        return samples


class ConstantFNInput(_Function):

    def __init__(self, function, name, args, kwargs, output_type=None):
        super(_Function, self).__init__(
            self,
            function,
            name,
            0,
            input_types=None,
            output_type=output_type
        )
        self.args = args
        self.kwargs = kwargs
    
    @property
    def value(self):
        value = self.fn(*self.args, **self.kwargs)
        return value


class BaseInputs(ABC):

    def __init__(self, values, types=None):
        self.values = values
        self.types = types or [type(value) for value in values]

    def __len__(self):
        return len(self.values)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @classmethod
    def build_idx(cls, i, prefix, suffix):
        # If prefix and suffix are None, return (i)
        # If only prefix is None, return suffix
        # If suffix is None, return "<prefix>_<i>"
        if prefix is None:
            if suffix is None:
                return str(i)
            return suffix
        if suffix is None:
            suffix = str(i)
        return "{}_{}".format(prefix, suffix)

    @classmethod
    def _flatten(cls, inputs, idx=None):
        flattened_values = list()
        flattened_idxs = list()
        if isinstance(inputs, BaseInputs):
            for i, (nested_idx, nested_value) in enumerate(inputs):
                nested_flat_idx, nested_flat_values = cls._flatten(nested_value, idx=nested_idx)
                nested_flat_idx = [
                    cls.build_idx(i, idx, nested_idx)
                    for nested_idx
                    in nested_flat_idx
                ]
                flattened_idxs.extend(nested_flat_idx)
                flattened_values.extend(nested_flat_values)        
        else:
            return [None], [inputs]
        
        return flattened_idxs, flattened_values

class NDInputs(BaseInputs):

    def __init__(self, values, types=None):
        self.values = values
        self.types = types or [values.dtype] * len(values.T)

    def __getitem__(self, idx):
        return self.values[:, idx]

    def __len__(self):
        return self.values.shape[1]

    def __iter__(self):
        for idx in range(len(self)):
            yield idx, self[idx]


class DataframeInputs(BaseInputs):

    def __init__(self, values, types=None):
        self.values = values
        self.types = types or values.dtypes.values.tolist()

    def __getitem__(self, idx):
        if idx in self.values.columns:
            return self.values[idx].to_numpy()
        return self.values.iloc[:, idx].to_numpy()
    
    def __len__(self):
        return self.values.shape[1]

    def __iter__(self):
        for idx in self.values.columns:
            yield idx, self[idx]


class DictInputs(BaseInputs):

    def __init__(self, values, types=None):
        self.values = frozendict(values)
        self.keys = list(self.values.keys())
        self.types = types or [type(value) for value in values.values()]

    def __getitem__(self, idx):
        if idx in self.keys:
            return self.values[idx]
        else:
            # Fallback to positional indexing if idx is not a key of self.values
            return self.values[self.keys[idx]]

    def __iter__(self):
        for idx in self.keys:
            yield idx, self[idx]
    
    @classmethod
    def flatten(cls, inputs):
        
        flattened_idxs, flattened_values = cls._flatten(inputs)
        
        return DictInputs(
            {
                k: v
                for k, v
                in zip(flattened_idxs, flattened_values)
            }
        )
    

class ListInputs(BaseInputs):

    def __getitem__(self, idx):
        return self.values[idx]

    def __iter__(self):
        for idx in range(len(self)):
            yield idx, self[idx]
    
    @classmethod
    def flatten(cls, inputs):
        
        flattened_idxs, flattened_values = cls._flatten(inputs)
        
        return ListInputs(flattened_values)
