'''Abstract base classes for phase-space vectors
'''

import numpy as np

from hamiltonian_models.vectors import PhaseSpace, PhaseSpaceVector

class NumpyPhaseSpaceVector(PhaseSpaceVector):
    '''A numpy implementation of PhaseSpaceVector.'''
    def __init__(self, space, vec_q, vec_p):
        if isinstance(vec_q, (float, int)):
            vec_q = np.array([vec_q], dtype=float)
        if isinstance(vec_p, (float, int)):
            vec_p = np.array([vec_p], dtype=float)
        super().__init__(space, vec_q, vec_p)

    def _component_dim(self, comp_vec):
        return comp_vec.shape[-1]

    def dot(self, other):
        if self.ndim > 1:
            raise NotImplementedError('dot not vectorized yet.')
        super().dot(other)
        return np.inner(self.vec_q, other.vec_q) + np.inner(self.vec_p, other.vec_p)

    def to_numpy(self):
        return np.hstack([self.vec_q, self.vec_p])
    
    @property
    def ndim(self):
        return self.vec_q.ndim

class NumpyPhaseSpace(PhaseSpace):
    '''A numpy implementation of PhaseSpace.'''
    _type_PhaseSpaceVector = NumpyPhaseSpaceVector
    _types_vec = (np.ndarray,)

    def __init__(self, dim, gdim, dofs=None):
        '''parameters:
        dim, gdim
            see PhaseSpace
        dofs
            a numpy ndarray of shape (gdim, dim/gdim) which describes which degree of freedom (DOF)
            belongs to which physical dimension
        '''
        super().__init__(dim, gdim)
        if gdim == 1 and not dofs:
            dofs = np.array([range(dim)])
        assert isinstance(dofs, np.ndarray) and dofs.shape == (gdim, dim/gdim)
        self._dofs = dofs
    
    def ones(self):
        return NumpyPhaseSpaceVector(self, np.ones(self.dim//2), np.ones(self.dim//2))

    def dofs(self, coordinate):
        super().dofs(coordinate)
        return self._dofs[coordinate]

    def from_numpy(self, np_vec):
        len_np_vec = np_vec.shape[-1]
        assert len_np_vec % 2 == 0, 'vector has to be even-dimensional in axis -1'
        return NumpyPhaseSpaceVector(self, np_vec[..., :len_np_vec//2], np_vec[..., len_np_vec//2:])
