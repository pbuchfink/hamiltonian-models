
from abc import ABC, abstractmethod
import numpy as np
from hamiltonian_models.integrators.base import Integrator, FixedTimeStepWidthIntegrator
from hamiltonian_models.integrators.implicit_midpoint import ImplicitMidpointIntegrator
from hamiltonian_models.integrators.stormer_verlet import SeparableStormerVerletIntegrator

class DynamicalSystem(ABC):
    '''A dynamical system as initial-value problem.
    
    Parameters
    ----------
    is_linear
        Flag, whether the right-hand side of the system is linear.
    '''
    def __init__(self, is_linear):
        self.is_linear = is_linear

        # empty set for parameter names
        self._parameters = set()
        self.available_integrators = [
            SeparableStormerVerletIntegrator,
            ImplicitMidpointIntegrator
        ]

    @abstractmethod
    def initial_value(self, mu):
        '''The (parameter-dependent) initial value.

        Parameters
        ----------
        mu
            A |dict| describing the parameters to evaluate the initial value for.
        '''
        assert self.check_parameters(mu, check_under=True)
        pass

    @abstractmethod
    def dxdt(self, x, mu):
        '''The RHS of the canonical Hamiltonian system:
            d/dt x(t, mu) = f(x, mu, t)
        where t is included in mu as parameter _t

        Parameters
        ----------
        x
            The vector to evaluate the RHS for.
        mu
            The parameter vector.
        '''
        assert self.check_parameters(mu, check_under=True)
        pass

    @abstractmethod
    def solve(self, t_0, t_end, integrator, mu):
        '''Solve dynamical system for all t in [t_0, t_end] with time-step width dt for the parameter vector mu.

        Parameters
        ----------
        t_0
            The initial time.
        t_end
            The final time.
        integrator
            The integrator to use to solve the system. Make sure that the integrator |is_applicable| to the system.
        mu
            The parameter vector.
        '''
        assert t_0 < t_end
        assert isinstance(integrator, Integrator) and integrator.is_applicable(self)
        assert self.check_parameters(mu, check_under=False)

        # pre-assemble operators
        if hasattr(self, 'preassemble') and isinstance(integrator, FixedTimeStepWidthIntegrator):
            self.preassemble(mu, integrator._dt)
        pass

    def list_applicable_integrators(self):
        return [integrator for integrator in self.available_integrators if integrator.is_applicable(self)]

    def check_parameters(self, mu, check_under=True):
        return isinstance(mu, dict) and \
               all([(param in mu.keys()) or ((not check_under) and param[0] == '_') for param in self._parameters])

    def update_mu(self, mu, update):
        '''Updates mu. Might be used to recompute time-dependent terms of the dynamical system.

        Parameters
        ----------
        mu
            The (old / current) parameter vector.
        update
            The entries to be updates as dict.
        '''
        assert self.check_parameters(mu, check_under=False)
        assert isinstance(update, dict) 
        mu.update(update)
        return mu
