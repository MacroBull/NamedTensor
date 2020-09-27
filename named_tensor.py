#! /usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Sun Sep 15 22:09:46 2019

@author: Macrobull
"""

from __future__ import absolute_import, division, unicode_literals

import warnings
import numpy as np

from collections import OrderedDict, namedtuple
from collections.abc import Collection, Hashable, Iterable


NamedIndex           :type = Hashable
NamedAxes            :type = 'Union[int, Mapping[NamedIndex, int], NamedTuple[int, ...]]'
IndexClass           :'Collection[type, ...]' = (None, int, list, slice)
ConvertableAxisClass :'Collection[type, ...]' = (None, NamedIndex, Collection, slice)
PartialAxesInfo      :type = 'Union[int, Iterable, Collection[Tuple[NamedIndex, int]]]'


### utils ###


def is_integer(value:'Any')->bool:
    r"""integer type test"""

    return isinstance(value, int) or (
        isinstance(value, np.number) and np.issubsctype(value, np.integer))


def is_namedtuple(value:'Any')->bool:
    r"""namedtuple test"""

    return isinstance(value, tuple) and hasattr(value, '_fields')


class FrozenDict(dict):
    r"""immutable 'dict'"""

    __delitem__ = property(doc='Disabled method')
    clear       = property(doc='Disabled method')
    pop         = property(doc='Disabled method')
    popitem     = property(doc='Disabled method')
    setdefault  = property(doc='Disabled method')
    update      = property(doc='Disabled method')

    def __init__(self,
                 *args, **kwargs):
        self._mutable = True
        super().__init__(*args, **kwargs)
        del self._mutable

    def __setitem__(self, key:Hashable, value:'Any'):
        if not getattr(self, '_mutable', False):
            raise AttributeError(f'{type(self).__name__!r} is immutable')
        return super().__setitem__(key, value)


class FrozenOrderedDict(FrozenDict, OrderedDict):
    r"""immutable 'OrderedDict'"""

    move_to_end = property(doc='Disabled method')


### NamedAxes ###


class SliceableFrozenOrderedDict(FrozenOrderedDict):
    r"""'SliceableFrozenOrderedDict' with slice range index support"""

    def __getitem__(self, indices:'Any')->'Any':
        if isinstance(indices, slice):
            keys = list(self.keys())
            values = list(self.values())
            start = None if indices.start is None else keys.index(indices.start)

            # HINT: stop + 1 for pandas style
            stop = None if indices.stop is None else keys.index(indices.stop)

            indices = type(indices)(start, stop, indices.step)
            return type(self)(zip(keys[indices], values[indices]))

        return super().__getitem__(indices)


def make_axes(info:PartialAxesInfo,
              cls_name:'Optional[str]'=None)->NamedAxes:
    r"""create 'NamedAxes' from 'PartialAxesInfo'"""

    if is_integer(info):
        assert info >= 0, f'naxis({info}) should be non-negative'
        assert cls_name is None, f'unnamed axes({info}) cannot have class name'

        return info

    if isinstance(info, Iterable):
        info = list(info)
        try:
            # try parsing as Mapping[NamedIndex, int] tuples
            axes = dict(info) # throw TypeError, ValueError
            assert all(map(is_integer, axes.values())), 'mapped indices should be integers'

            slots = np.zeros(len(axes), dtype=bool)
            slots[list(axes.values())] = True # throw IndexError
            assert slots.sum() == len(slots), 'bad axis indices' # HINT: all(0 <= axes < naxis)?
        except IndexError:
            raise ValueError('bad axis indices')
        except (TypeError, ValueError):
            # fallback to enumerate indices for axes
            assert all(map(lambda axis: isinstance(axis, NamedIndex), info))

            axes = dict(zip(info, range(len(info))))
        else:
            # sort name by indices
            axes_ = axes
            axes = {k: v if v >= 0 else len(axes) + v for k, v in axes.items()}
            keys = sorted(axes.keys(), key=axes.get)
            axes = OrderedDict()
            for key in keys:
                axes[key] = axes_[key]

        if cls_name is not None:
            axes = namedtuple(cls_name, axes.keys())(**axes)

        return axes

    raise TypeError(f"unsupported 'info' type({type(info)})")


def axis2index(axes:NamedAxes, axis:ConvertableAxisClass)->IndexClass:
    r"""convert named axis to int index"""

    if axis is None:
        return axis

    # NOTE: iterating until hashable
    if isinstance(axis, Collection) and not isinstance(axis, Hashable):
        iter_index = (axis2index(axes, a) for a in axis)
        if isinstance(axis, np.ndarray): # force list output for numpy array
            return list(iter_index)
        return type(axis)(iter_index)

    if isinstance(axis, slice):
        start = axis2index(axes, axis.start)
        stop = axis2index(axes, axis.stop)
        return type(axis)(start, stop, axis.step)

    if isinstance(axes, dict):
        return axes[axis]

    if is_namedtuple(axes):
        return getattr(axes, axis)

    # fallback to int axes with int axis
    assert is_integer(axes) and is_integer(axis), f'unnamed axis({axis!r}) should be integer'

    return axis


def axesiter(axes:NamedAxes)->'Iterator[NamedIndex]':
    r"""create iterator over axes"""

    if isinstance(axes, dict):
        axes = {k: v if v >= 0 else len(axes) + v for k, v in axes.items()} # HINT: non-negative
        keys = sorted(axes.keys(), key=axes.get)
        return iter(keys)

    if is_namedtuple(axes):
        keys = sorted(axes._fields, key=axes.__getattribute__)
        return iter(keys)

    return iter(range(axes))


### NamedDims ###


class FixedShapeNamedDims(SliceableFrozenOrderedDict):
    r"""Named dims class with immutable shape"""

    # class NotSet(object):
    #     r"""placeholder for not set index"""

    __slots__ = ('_dims', '_dim_axes')

    _dims     :'Tuple[NamedIndex, ...]'
    _dim_axes :'Tuple[Tuple[NamedIndex, NamedAxes], ...]'

    def __init__(self,
                 *args,
                 dict_axes:bool=False,
                 **kwargs):
        upper = OrderedDict(*args, **kwargs)
        for dim, axes in upper.items():
            if not is_namedtuple(axes):
                try:
                    assert not dict_axes
                    axes = make_axes(axes, cls_name=dim)
                except (ValueError, AssertionError):
                    axes = make_axes(axes)
            upper[dim] = SliceableFrozenOrderedDict(axes) if isinstance(axes, dict) else axes
        super().__init__(upper)
        self._dims = tuple(self.keys())
        self._dim_axes = tuple(self.values())

    @property
    def dims(self)->'Sequence[NamedIndex]':
        r"""dims getter"""

        return self._dims

    @property
    def ndim(self)->int:
        r"""dimension"""

        return len(self._dims)

    @property
    def shape(self)->'Tuple[int, ...]':
        r"""shape getter"""

        return tuple(axes if is_integer(axes) else len(axes) for axes in self._dim_axes)

    @property
    def size(self)->int:
        r"""number of total elements"""

        return np.prod([axes if is_integer(axes) else len(axes) for axes in self._dim_axes])

    @property
    def flat(self)->'Iterator[Sequence[NamedIndex]]':
        r"""flat iter of all indices"""

        axis_iters = list(map(axesiter, self._dim_axes))
        axes = list(map(next, axis_iters))
        while True:
            yield tuple(axes)
            idx_dim = self.ndim - 1
            while idx_dim >= 0:
                try:
                    axes[idx_dim] = next(axis_iters[idx_dim]) # throw StopIteration
                except StopIteration:
                    axis_iters[idx_dim] = axesiter(self._dim_axes[idx_dim])
                    axes[idx_dim] = next(axis_iters[idx_dim])
                    idx_dim -= 1
                else:
                    break
            if idx_dim < 0:
                break

    def axes(self, dim:NamedIndex)->'Sequence[NamedIndex]':
        r"""axes getter"""

        axes = self[dim] # disable idx_dim access
        # axes = self.get(dim) or self._dim_axes[dim] # dim:'Union[int, NamedIndex]'
        return tuple(axesiter(axes))

    def axes2indices(
            self,
            axes:'Union['
                'Tuple[Union[ConvertableAxisClass, ellipsis], ...], '
                'Mapping[NamedIndex, ConvertableAxisClass],'
                ']',
            )->'Any':
        r"""convert named axes to int indices"""

        if isinstance(axes, dict):
            indices = []
            for dim, axes_ in self.items():
                axis = axes.get(dim)
                if axis is None and dim in axes:
                    warnings.warn(f'it does not make sense using None(at dim {dim!r}) '
                                  'in a named index, it whould be translated into '
                                  'slice(None)(i.e. :)')
                index = slice(None) if axis is None else axis2index(axes_, axis)
                indices.append(index)
            return tuple(indices)

        axes = axes if isinstance(axes, tuple) else (axes, )
        idx_elps = naxis = len(axes)
        for idx_axis, axis in enumerate(axes):
            if isinstance(axis, type(Ellipsis)):
                assert idx_elps == naxis, 'more than one ellipsis is not allowed'

                idx_elps = idx_axis

        indices = []
        idx_axis = idx_dim = 0
        while idx_axis < idx_elps:
            axis = axes[idx_axis]
            index = None if axis is None else axis2index(self._dim_axes[idx_dim], axis)
            indices.append(index)
            idx_axis += 1
            idx_dim += index is not None

        if idx_elps < naxis:
            indices.append(axes[idx_elps])
            remainder = idx_elps + 1 - naxis
            indices_ = []
            idx_axis = idx_dim = -1
            while idx_axis >= remainder:
                axis = axes[idx_axis]
                index = None if axis is None else axis2index(self._dim_axes[idx_dim], axis)
                indices_.append(index)
                idx_axis -= 1
                idx_dim -= index is not None
            indices_.reverse()
            indices.extend(indices_)

        return tuple(indices)

    def replace_dim(self, old:NamedIndex, new:NamedIndex):
        r"""create a new dim-renamed one"""

        assert new not in self, f'new dim({new!r}) is confilicted'

        ret = OrderedDict()
        for dim, axes in self.items():
            if dim == old:
                if is_namedtuple(axes):
                    axes = namedtuple(new, axes._fields)(*axes)
                ret[new] = axes
            else:
                ret[old] = axes

        return type(self)(ret)

    def replace_axis(self, dim:NamedIndex,
                     mapping_or_old:'Union[Mapping[NamedIndex, NamedIndex], NamedIndex]',
                     new:'Optional[NamedIndex]'=None):
        r"""create a new axis-renamed one"""

        axes = self[dim] # disable idx_dim access
        # axes = self.get(dim) or self._dim_axes[dim] # dim:'Union[int, NamedIndex]'
        is_tuple_axes = is_namedtuple(axes)
        assert isinstance(axes, dict) or is_tuple_axes, (
            f'unnamed dim({dim!r}) cannot be renamed')

        axes_keys = axes._fields if is_tuple_axes else axes.keys()
        axes_iter = iter(zip(axes._fields, axes)) if is_tuple_axes else axes.items()
        axes_ = OrderedDict()

        if new is None:
            assert isinstance(mapping_or_old, dict), (
                f"'mapping_or_old'({type(mapping_or_old)}) is expected to be a dict "
                "when 'new' is None")

            mapping = mapping_or_old
            for axis, index in axes_iter:
                axis = mapping.get(axis, axis)
                assert axis not in axes_, f'axis {axis!r} in mapping is conflicted'

                axes_[axis] = index
        else:
            assert new not in axes_keys, f'new axis({new!r}) is confilicted'

            old = mapping_or_old
            for axis, index in axes_iter:
                axes_[new if axis == old else axis] = index

        axes_ = namedtuple(dim, axes_.keys())(**axes_) if is_tuple_axes else type(axes)(axes_)
        ret = OrderedDict()
        for dim_, axes in self.items():
            ret[dim_] = axes_ if dim_ == dim else axes

        return type(self)(ret)

    def tofile(self, filename:'Union[str, Path]'):
        r"""save data to file with dims info"""

        with open(filename, mode='w') as file:
            dims = ','.join(map(str, self._dims))
            file.write(f'dims={dims}\n')
            for dim, axes in self.items():
                if isinstance(axes, dict):
                    axes = ','.join(map(str, axesiter(axes)))
                    file.write(f'axes={axes}\n')
                else:
                    file.write(f'dim={axes}\n')


def dims_from_df(
        df:'pd.DataFrame',
        cls_name:'Optional[str]'=None)->'Union[OrderedDict, NamedTuple[int, ...]]':
    r"""create 'NamedDims' for 'DataFrame'"""

    dims = OrderedDict()
    for dim in df.columns:
        if np.issubsctype(df[dim], np.inexact): # skip non-discrete
            continue

        axes = np.unique(df[dim])
        if np.issubsctype(axes, np.integer):
            axes = axes.max().item() - min(0, axes.min().item()) + 1
        else:
            axes.sort()
            axes = OrderedDict(zip(axes, range(len(axes))))

        if cls_name is not None:
            axes = namedtuple(cls_name, axes.keys())(**axes)

        dims[dim] = axes
    return FixedShapeNamedDims(dims)


### NamedTensor ###


class FixedShapeNamedTensor(object):
    r"""
    Named tensor class with immutable shape

    How to represent a tensor with named dims and axes?
    Suppose we have a 3x4 tensor, where:
    	its dimension(ndim) is 2
    	its shape is (3, 4)
    	dim 0 has 3 axes
    	dim 1 has 4 axes
    If we assign names for dims and axes like:
    	dim 0 for Fruit, with axes as Apple, Banana, Coconut
    	dim 1 for Size, with axes as Tiny, Small, Medium, Large
    A probability distribution tensor of a random chosen fruit can be written like:
    							Size
				       	Tiny	Small	Medium	Large
    			Apple 	0.000	0.006	0.196	0.222
    	Fruit	Banana 	0.040	0.102	0.005	0.068
    			Coconut 0.172	0.163	0.000	0.026
    A NamedTensor can be accessed like:
    	prob[Apple, Small] = 0.006
    	prob[{'Size': Small, 'Fruit': Apple}] = 0.006
    	prob[Coconut] =
    				Size
    		Tiny	Small	Medium	Large
    		0.172	0.163	0.000	0.026
    """

    DISABLED_METHODS :'Collection[str]' = ('resize', )

    PartialDimsInfo  :type = 'Collection[Union[NamedIndex, Tuple[NamedIndex, PartialAxesInfo]]]'
    DataClass        :type = np.ndarray

    __slots__ = ('name', '_dims', '_data')

    name :str # = '' # optional name for this tensor

    _dims :FixedShapeNamedDims
    _data :DataClass

    def __init__(self,
                 data:'Optional[DataClass]'=None,
                 dims:'Optional[PartialDimsInfo]'=None,
                 **kwargs):
        assert not(dims is None and data is None), 'either data or dims should be given'

        if dims is not None:
            if not isinstance(dims, dict):
                dims_ = OrderedDict()
                for idx_dim, dim in enumerate(dims):
                    if isinstance(dim, tuple):
                        dim, axes = dim
                        dims_[dim] = axes
                    else:
                        assert data is not None, 'shape cannot be infered without data'

                        dims_[dim] = data.shape[idx_dim]
                dims = dims_
            self._dims = FixedShapeNamedDims(dims)
        if data is None:
            self.create(**kwargs)
        else:
            self._data = data
            if dims is None:
                self._dims = FixedShapeNamedDims(enumerate(data.shape))

        self.name = ''
        self._check_consistency()

    def __repr__(self)->str:
        name_str = f'name={self.name}, ' if self.name else ''
        return (f'<{type(self).__name__} '
                f'{name_str}shape={self._data.shape}, dims={tuple(self._dims.keys())}'
                '>')

    def __str__(self)->str:
        return str(self._data) # TODO

    def __format__(self, fmt:str)->str:
        return format(self._data, fmt) # TODO

    def __getattr__(self, key:str)->'Any':
        if key in self.DISABLED_METHODS:
            raise AttributeError(f'{type(self).__name__!r} is immutable')
        return getattr(self._data, key) # TODO

    def __iter__(self)->'Iterator':
        return iter(self._data)

    def __len__(self)->int:
        return len(self._data)

    def __contains__(self, k:'Any')->bool:
        return k in self._data

    def __getitem__(self, axes:'Any')->'Union[Number, DataClass]':
        return self._data[self._dims.axes2indices(axes)]

    def __setitem__(self, axes:'Any', value:'Any'):
        self._data[self._dims.axes2indices(axes)] = value

    def __lt__(self, other:DataClass)->'Union[bool, DataClass]':
        if isinstance(other, FixedShapeNamedTensor):
            return self._data < other._data
        return self._data < other

    def __le__(self, other:DataClass)->'Union[bool, DataClass]':
        if isinstance(other, FixedShapeNamedTensor):
            return self._data <= other._data
        return self._data <= other

    def __eq__(self, other:DataClass)->'Union[bool, DataClass]':
        if isinstance(other, FixedShapeNamedTensor):
            return self._data == other._data
        return self._data == other

    def __ne__(self, other:DataClass)->'Union[bool, DataClass]':
        if isinstance(other, FixedShapeNamedTensor):
            return self._data != other._data
        return self._data != other

    def __gt__(self, other:DataClass)->'Union[bool, DataClass]':
        if isinstance(other, FixedShapeNamedTensor):
            return self._data > other._data
        return self._data > other

    def __ge__(self, other:DataClass)->'Union[bool, DataClass]':
        if isinstance(other, FixedShapeNamedTensor):
            return self._data >= other._data
        return self._data >= other

    @property
    def dims(self)->FixedShapeNamedDims:
        return self._dims

    @property
    def data(self)->DataClass:
        return self._data

    @data.setter
    def data(self, value:np.ndarray):
        assert value.shape == self._data.shape, (
            f'shape mismatch({self.shape} vs {value.shape}) when assigning')

        self._data = value

    def create(self,
               fill_value:'Optional[Number]'=None,
               **kwargs):
        r"""create data with dims info"""

        shape = self._dims.shape
        if fill_value is None:
            data = np.empty(shape, **kwargs)
        else:
            data = np.full(shape, fill_value, **kwargs)

        assert isinstance(data, self.DataClass), (
            f'overload for create is required for non-default data class({type(data)})')

        self._data = data
        return self.data

    def tofile(self, filename:'Union[str, Path]',
               *args, **kwargs):
        r"""save data to file with dims info"""

        self._data.tofile(filename, *args, **kwargs)
        dims_filename = filename + '.ini' # '.meta'
        self._dims.tofile(dims_filename)

        with open(dims_filename, mode='a') as file:
            file.write(f'dtype={self._data.dtype.name}\n')

    def replace_dim(self,
                    *args, **kwargs):
        r"""shortcut for dims.rename_dim"""

        return type(self)(data=self._data, dims=self._dims.replace_dim(*args, **kwargs))

    def replace_axis(self,
                     *args, **kwargs):
        r"""shortcut for dims.rename_axis"""

        return type(self)(data=self._data, dims=self._dims.replace_axis(*args, **kwargs))

    def _check_consistency(self):
        # assert self._dims.ndim == self._data.ndim
        assert self._dims.shape == self._data.shape


if __name__ == '__main__':
    shape = 2, 3, 4, 5, 6, 7, 8, 9
    t = FixedShapeNamedTensor(data=np.arange(np.prod(shape)).reshape(shape))
    print(repr(t))
    print(t.shape)

    t = FixedShapeNamedTensor(
        data=np.arange(np.prod(shape)).reshape(shape),
        dims=(
            ('funky', {0: 1, 1: 0}.items()),
            ('Channels', 'RGB'),
            ('q', (('x', 1), ('y', 2), ('z', 3), ('w', 0))),
            None,
            -42,
            b'bytes',
            ('N', 8),
            ('rot', range(-4, 5)),
        ))
    print(repr(t))
    print(t.shape)
    print(t[0].shape)

    # element access
    print(t[0, 'R', 'x', 1, 2, 3, -4, -4], t.data[1, 0, 1, 1, 2, 3, -4, 0])
    # slice
    print(t[:, np.newaxis, 'G', ..., 0, -1, 2, -3])
    # named access, None as slice(None)/:
    print(t[{'Channels': 'G', 'q': None, -42: 0, b'bytes': -1, 'N': 2, 'rot': -3}])
    print(t.data[:, 1, ..., 0, -1, 2, 1])
    # ranged dims
    print(t.dims['Channels':'N'].shape, t[0, ..., 0, 0].shape)

    prob = np.random.rand(3, 4).astype(np.float32) ** 2
    prob /= prob.sum()
    prob = FixedShapeNamedTensor(
        prob,
        dims=(
            ('Fruit', ('Apple', 'Banana', 'Coconut')),
            ('Size', ('Tiny', 'Small', 'Medium', 'Large')),
        )
    )
    print(repr(prob))
    print(prob['Apple', 'Small'])
    print(prob[{'Size': 'Small', 'Fruit': 'Apple'}])
    print(prob['Coconut'])

    for i, v in zip(prob.dims.flat, prob.flat):
        print(f'{i!s:30s}: {v:.3f}')

    for i, v, _ in zip(t.dims.flat, t.flat, range(20)):
        print(f'{i!s:30s}: {v:d}')

    prob.tofile('/tmp/prob.bin')
    t.tofile('/tmp/t.txt', sep=' ')

    print()
    print(open('/tmp/prob.bin.ini').read())
    print(open('/tmp/t.txt.ini').read())
