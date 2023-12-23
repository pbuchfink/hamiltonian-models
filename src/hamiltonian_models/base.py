# Abstract base classes for Hamiltonian systems

from abc import abstractmethod
import numpy as np
from hamiltonian_models.dyn_sys import DynamicalSystem
from hamiltonian_models.integrators.base import HookFunction, TimeDataList
from hamiltonian_models.vectors import PhaseSpaceVector

class HamiltonianSystem(DynamicalSystem):
    '''A parametric Hamiltonian system.

    Parameters
    ----------
    phase_space
        The |PhaseSpace| of the Hamiltonian system.
    '''
    def __init__(self, phase_space, **kw):
        self.phase_space = phase_space
        super(HamiltonianSystem, self).__init__(**kw)

    @abstractmethod
    def Ham(self, x, mu):
        '''The (parameter-dependent) Hamiltonian function.

        Parameters
        ----------
        x
            A |PhaseSpaceVector| to evaluate the Hamiltonian for.
        mu
            A |dict| describing the parameters to evaluate the Hamiltonian for.
        '''
        assert isinstance(x, PhaseSpaceVector) and self.check_parameters(mu, check_under=True)
        pass

    @abstractmethod
    def gradHam(self, x, mu):
        '''The (parameter-dependent) gradient of the Hamiltonian function.

        Parameters
        ----------
        x
            A |PhaseSpaceVector| to evaluate the gradient of the Hamiltonian function for.
        mu
            A |dict| describing the parameters to evaluate the gradient of the Hamiltonian function for.
        '''
        assert isinstance(x, PhaseSpaceVector) and self.check_parameters(mu, check_under=True)
        pass

    @abstractmethod
    def J_y(self, x, mu, y):
        '''Applies the (parameter-dependent) Poisson structure to a |PhaseSpaceVector|.

        Parameters
        ----------
        x
            A |PhaseSpaceVector| to evaluate the Poisson structure for.
        mu
            A |dict| describing the parameters to evaluate the Poisson structure for.
        y
            A |PhaseSpaceVector| to apply the Poisson structure to.

        Returns
        -------
        The product J(x, mu) * y
        '''
        assert isinstance(x, PhaseSpaceVector) and self.check_parameters(mu, check_under=True)
        pass

    def dxdt(self, x, mu):
        '''The RHS of the canonical Hamiltonian system:
            d/dt x(t, mu) = J * gradHam(x(t, mu), mu)

        Parameters
        ----------
        x
            The vector to evaluate the RHS for.
        mu
            The parameter vector.
        '''
        assert isinstance(x, PhaseSpaceVector) and self.check_parameters(mu, check_under=True)
        return self.J_y(x, mu, self.gradHam(x, mu))


class Vectorized():
    """Indicates that all operations are implemented vectorized."""
    ...


class CanonicalHamiltonianSystem(HamiltonianSystem):
    '''A parametric canonical Hamiltonian system.
    '''
    def J_y(self, x, mu, y):
        '''State- and parameter independent canonical Poisson structure.
        '''
        super().J_y(x, mu, y)
        y.vec_q, y.vec_p = y.vec_p, -y.vec_q
        return y


class QuadraticHamiltonianSystem(HamiltonianSystem,Vectorized):
    '''A |HamiltonianSystem| with a quadratic Hamiltoninan specified by the quadratic part H_op and the linear part h_op.
    '''
    def __init__(self, phase_space, **kw):
        super(QuadraticHamiltonianSystem, self).__init__(phase_space=phase_space, is_linear=True)

    def Ham(self, x, mu):
        super().Ham(x, mu)
        if x.ndim == 2:
            return [self.Ham(x_elem, mu) for x_elem in x]
        elif x.ndim == 1:
            return x.dot(self.H_op_y(mu, x).scal(.5) + self.h_op(mu))
        else:
            raise NotImplementedError()

    def gradHam(self, x, mu):
        super().gradHam(x, mu)
        return self.H_op_y(mu, x) + self.h_op(mu)

    def hessianHam_y(self, x, mu, y):
        '''Hessian of Hamiltonian applied to y'''
        return self.H_op_y(mu, y)

    def dxdt(self, x, mu):
        return self.J_y(x, mu, self.H_op_y(mu, x) + self.h_op(mu))

    @abstractmethod
    def H_op_y(self, mu, y):
        '''Specifies the quadratic part of the Hamiltonian applied to y:
            H_op(mu) * y
        '''
        assert self.check_parameters(mu, check_under=True) and isinstance(y, PhaseSpaceVector)
        pass

    @abstractmethod
    def h_op(self, mu):
        '''Specifies the linear part of the Hamiltonian.
        '''
        assert self.check_parameters(mu, check_under=True)
        pass


class QuadraticHamiltonianHookFunction(HookFunction):
    def __init__(self, ham_sys):
        super(QuadraticHamiltonianHookFunction, self).__init__(name='QuadraticHamiltonian')
        assert isinstance(ham_sys, QuadraticHamiltonianSystem)
        self.ham_sys = ham_sys
        self.output = TimeDataList()
        self._cached_H_op_y = TimeDataList()
        self._cached_h_op = TimeDataList()

        self.ham_sys_H_op_y = ham_sys.H_op_y
        self.ham_sys_h_op = ham_sys.h_op

    def setup(self, x0, mu, nt):
        # hook H_op_y and h_op to cache operators
        self.ham_sys.H_op_y = self._hook_H_op_y
        self.ham_sys.h_op = self._hook_h_op

    def eval(self, x, mu):
        # use cached operators to evaluate Hamiltonian
        assert isinstance(x, PhaseSpaceVector) and self.ham_sys.check_parameters(mu, check_under=True)
        assert len(self._cached_H_op_y) > 0, 'No cached H_op_y available.'
        assert len(self._cached_h_op) > 0, 'No cached h_op available.'
        t_H, H_op_y = self._cached_H_op_y.pop()
        t_h, h_op = self._cached_h_op.pop()
        if not np.isclose(t_H, mu['_t']):
            H_op_y = self.ham_sys.H_op_y(mu, x)
        if not np.isclose(t_h, mu['_t']):
            h_op = self.ham_sys.h_op(mu)
        
        self.output.append(mu['_t'], x.dot(H_op_y.scal(.5) + h_op))

    def finalize(self):
        # reset hooked functions
        self.ham_sys.H_op_y = self.ham_sys_H_op_y
        self.ham_sys.h_op = self.ham_sys_h_op

    def _hook_H_op_y(self, mu, y):
        H_op_y = self.ham_sys_H_op_y(mu, y)
        while(len(self._cached_H_op_y) > 0):
            self._cached_H_op_y.pop()
        self._cached_H_op_y.append(mu['_t'], H_op_y)
        return H_op_y

    def _hook_h_op(self, mu):
        h_op = self.ham_sys_h_op(mu)
        while(len(self._cached_h_op) > 0):
            self._cached_h_op.pop()
        self._cached_h_op.append(mu['_t'], h_op)
        return h_op


class HamiltonianHookFunction(HookFunction):
    def __init__(self, ham_sys):
        super(HamiltonianHookFunction, self).__init__(name='Hamiltonian')
        assert isinstance(ham_sys, HamiltonianSystem)
        self.ham_sys = ham_sys
        self.output = TimeDataList()

    def setup(self, x0, mu, nt):
        pass

    def eval(self, x, mu):
        self.output.append(mu['_t'], self.ham_sys.Ham(x, mu))

    def finalize(self):
        pass
