'''Abstract base classes for phase-space vectors
'''

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from hamiltonian_models.integrators.base import TimeDataList

class PhaseSpace(ABC):
    '''A |PhaseSpace| describes the underlying vector space of a |HamiltonianSystem|

    Input
    -----
    dim
        Dimension of the |PhaseSpace|. It is necessarily even.
    gdim
        Physical dimension of the system. dim is necessarily a multiple of gdim.
    '''
    def __init__(self, dim, gdim):
        assert isinstance(self._type_PhaseSpaceVector, type)
        assert isinstance(self._types_vec, tuple) and all(isinstance(_type_vec, type) for _type_vec in self._types_vec)
        assert dim%2 == 0
        assert isinstance(gdim, int) and np.mod(dim, gdim) == 0
        self.dim = dim
        self.gdim = gdim

    @abstractmethod
    def ones(self):
        '''Returns the vector of all ones.'''
        pass

    def zeros(self):
        '''Returns the vector of all zeros.'''
        return self.ones().scal(0.)

    def new_vector(self, vec_q, vec_p):
        '''Generates a new |PhaseSpaceVector| with given components vec_q, vec_p.
        '''
        return self._type_PhaseSpaceVector(self, vec_q, vec_p)

    @abstractmethod
    def dofs(self, coordinate):
        '''Returns the indices which relate to the given coordinate.'''
        assert(isinstance(coordinate, int)) and 0 <= coordinate < self.gdim
        pass


class PhaseSpaceVector(ABC):
    '''A representation of a phase space vector composed of two parts, vec_q and vec_p.'''
    def __init__(self, space, vec_q, vec_p):
        self.space = space
        self.vec_q = vec_q
        self.vec_p = vec_p

    @abstractmethod
    def _component_dim(self, comp_vec):
        '''Returns the dimension, i.e. the size, of the given component vector.'''
        pass

    @abstractmethod
    def dot(self, other):
        '''Returns the dot product of self and other.'''
        assert self.space.dim == other.space.dim
        pass

    @abstractmethod
    def to_numpy(self):
        '''Returns the vector as numpy vector.'''
        pass
    
    @abstractproperty
    def ndim(self):
        '''Returns the number of dimensions of the vector.'''
        pass

    def __add__(self, other):
        assert isinstance(other, PhaseSpaceVector) and self.space.dim == other.space.dim
        return self.__class__(self.space, self.vec_q + other.vec_q, self.vec_p + other.vec_p)

    def __sub__(self, other):
        assert isinstance(other, PhaseSpaceVector) and self.space.dim == other.space.dim
        return self.__class__(self.space, self.vec_q - other.vec_q, self.vec_p - other.vec_p)

    def __mul__(self, other):
        if isinstance(other, PhaseSpaceVector):
            assert self.space.dim == other.space.dim
            return self.vec_q * other.vec_q + self.vec_p * other.vec_p
        elif isinstance(other, float):
            res = self.copy()
            return res.scal(other)
        else:
            raise NotImplementedError()

    def __iadd__(self, other):
        assert isinstance(other, PhaseSpaceVector) and self.space.dim == other.space.dim
        self.vec_q += other.vec_q
        self.vec_p += other.vec_p
        return self

    def __isub__(self, other):
        assert isinstance(other, PhaseSpaceVector) and self.space.dim == other.space.dim
        self.vec_q -= other.vec_q
        self.vec_p -= other.vec_p
        return self

    def __neg__(self):
        self.vec_q = -self.vec_q
        self.vec_p = -self.vec_p
        return self

    def __pos__(self):
        pass

    def l2_norm2(self):
        '''Returns squared l2 norm of self.'''
        return self.dot(self)

    def l2_norm(self):
        '''Returns l2 norm of self.'''
        return np.sqrt(self.l2_norm2())

    def scal(self, scal):
        self.vec_q = scal * self.vec_q
        self.vec_p = scal * self.vec_p
        return self

    def __setattr__(self, attr, value):
        # ensure for vec_q and vec_p to be of self.space._types_vec and to have correct dimensions
        if attr in ('vec_q', 'vec_p'):
            assert self._component_dim(value) == self.space.dim//2
        super(PhaseSpaceVector, self).__setattr__(attr, value)

    def copy(self):
        return self.__class__(self.space, self.vec_q.copy(), self.vec_p.copy())
    
    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice))
        if self.ndim == 1:
            if idx == 0:
                return self
            else:
                raise IndexError
        elif self.ndim == 2:
            return self.__class__(self.space, self.vec_q[idx], self.vec_p[idx])


class PhaseSpaceVectorList(TimeDataList):
    '''A list of PhaseSpaceVectors which enables to get the component vectors of all items in the list
    comfortably with one command.'''
    def __init__(self, td_t=None, td_data=None):
        if td_data:
            assert all(isinstance(item, PhaseSpaceVector) for item in td_data)
        super().__init__(td_t=td_t, td_data=td_data)

    def all_vec_q(self):
        '''Return all vec_q components of items in the list.'''
        return (item.vec_q for item in self._data)

    def all_vec_p(self):
        '''Return all vec_p components of items in the list.'''
        return (item.vec_p for item in self._data)

    def all_data_to_numpy(self):
        '''Return representation as numpy array.'''
        return np.vstack(tuple(item.to_numpy() for item in self.all_data()))
