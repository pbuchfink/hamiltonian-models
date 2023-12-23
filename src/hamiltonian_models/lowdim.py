# Canonical, TIME-INDEPENDENT Hamiltonian Systems with LOW-DIMENSIONAL phase-space

import numpy as np
from hamiltonian_models.base import CanonicalHamiltonianSystem, QuadraticHamiltonianSystem, QuadraticHamiltonianHookFunction, HamiltonianHookFunction
from hamiltonian_models.integrators.base import FixedTimeStepWidthIntegrator, TimeDataList
from hamiltonian_models.vectors import PhaseSpaceVector, PhaseSpaceVectorList
from hamiltonian_models.vectors_numpy import NumpyPhaseSpace

class HarmonicOscillator(QuadraticHamiltonianSystem, CanonicalHamiltonianSystem):
    '''A |QuadraticHamiltonianSystem| describing a harmonic oscillator.

        m d^2/d(t^2) q(t, mu) + k q(t, mu) = f(mu)

    The parameter vector consists of the parameter components:
        mu['m']: mass m
        mu['k']: spring stiffness k
        mu['f']: external force f (constant in time)
        mu['q0']: initial displacement
        mu['p0']: initial momentum
    '''
    is_separable = True
    def __init__(self):
        super().__init__(
            NumpyPhaseSpace(2, 2, np.array([[0], [1]])),
        )
        self.available_integrators.append(HarmonicOscillatorClosedFormulaIntegrator)

    def H_op_y(self, mu, y):
        super().H_op_y(mu, y)
        res = y.copy()
        y.vec_q *= mu['k']
        y.vec_p *= 1/mu['m']
        return res

    def h_op(self, mu):
        super().h_op(mu)
        return self.phase_space.new_vector(-mu['f'], 0)

    def initial_value(self, mu):
        super().initial_value(mu)
        return self.phase_space.new_vector(mu['q0'], mu['p0'])

    def _inv_M_dt_Df_y(self, x, mu, dt, y):
        y.vec_q /= (1 - dt/2 * mu['k'])
        y.vec_p /= (1 - dt/2 / mu['m'])
        return y

    def solve(self, t_0, t_end, integrator, mu):
        super().solve(t_0, t_end, integrator, mu)

        if isinstance(integrator, HarmonicOscillatorClosedFormulaIntegrator):
            return integrator.solve(self, mu, t_0, t_end)
        else:
            Ham_hook_fcn = QuadraticHamiltonianHookFunction(self)

            td_x = integrator.solve(self, mu, t_0, t_end, hook_fcns=Ham_hook_fcn)

            return td_x, Ham_hook_fcn.output

class HarmonicOscillatorClosedFormulaIntegrator(FixedTimeStepWidthIntegrator):
    @staticmethod
    def is_applicable(dyn_sys):
        return isinstance(dyn_sys, HarmonicOscillator)

    def solve(self, dyn_sys, mu, t_0, t_end, hook_fcns=None):
        super().solve(dyn_sys, mu, t_0, t_end, hook_fcns)
        assert hook_fcns is None

        sol_t = np.arange(t_0, t_end, self._dt)
        x0 = dyn_sys.initial_value(mu)
        omega0 = np.sqrt(mu['k']/mu['m'])

        td_x = PhaseSpaceVectorList()
        td_Ham = TimeDataList()
        phase_space = dyn_sys.phase_space
        nt = len(sol_t)
        for i, t in enumerate(sol_t):
            td_x.append(
                t,
                phase_space.new_vector(
                    x0.vec_q*np.cos(omega0*(t - t_0)) + (1/(mu['m']*omega0))*x0.vec_p*np.sin(omega0*(t - t_0)),
                    -mu['m']*omega0*x0.vec_q*np.sin(omega0*(t - t_0)) + x0.vec_p*np.cos(omega0*(t - t_0))
                )
            )
            td_Ham.append(t, dyn_sys.Ham(td_x._data[-1], mu))

            self.print_progress(i-1, i, nt)

        self.print_progress(nt-1, nt, nt)

        return td_x, td_Ham

class SimplePendulum(CanonicalHamiltonianSystem):
    '''A |CanonicalHamiltonianSystem| describing the dynamics of a simple (mathematical) pendulum.

        m l d^2/(dt^2) q(t, mu) + m g sin(q(t, mu)) = 0

    The paramter vector consists of the parameter components
        mu['m']: mass m
        mu['g']: gravitational acceleration g
        mu['l']: pendulum length l
        mu['q0']: initial angle
        mu['p0']: initial momentum
    '''
    is_separable = True
    def __init__(self):
        super().__init__(
            NumpyPhaseSpace(2, 2, np.array([[0], [1]])),
            is_linear=False
        )

    def Ham(self, x, mu):
        super().Ham(x, mu)
        return x.vec_p**2/(2*mu['m']*mu['l']**2) + mu['m']*mu['g']*mu['l']*(1-np.cos(x.vec_q))

    def gradHam(self, x, mu):
        super().gradHam(x, mu)
        return self.phase_space.new_vector(
            mu['m']*mu['g']*mu['l']*np.sin(x.vec_q),
            x.vec_p/(mu['m']*mu['l']**2)
        )

    def initial_value(self, mu):
        super().initial_value(mu)
        return self.phase_space.new_vector(mu['q0'], mu['p0'])

    def _inv_M_dt_Df_y(self, x, mu, dt, y):
        '''Returns the product
                inv(I - dt/2 * Df(x, mu)) * y
            where
                Df(x, mu) = J * [[m*g*l*cos(x.vec_p), 0], [0, 1/(m*l^2)]]
            is the derivative of dxdt with respect to x and J = [[0,1], [-1,0]] is the canonical Poisson matrix.
            The inverse is derived via LR decomposition and explicit inversion of the matrix.
        '''
        assert isinstance(x, PhaseSpaceVector) and self.check_parameters(mu, check_under=True) and isinstance(y, PhaseSpaceVector)
        a = dt/2*mu['m']*mu['g']*mu['l']*np.cos(x.vec_p)
        b = dt/(2*mu['m']*mu['l']**2)
        # multiplication with inv(L)
        det_L = 1-a*b
        y.vec_p *= 1/det_L
        y.vec_q -= a/det_L * y.vec_p
        # multiplication with inv(R)
        y.vec_p -= b * y.vec_q
        return y

    def solve(self, t_0, t_end, integrator, mu):
        super().solve(t_0, t_end, integrator, mu)

        Ham_hook_fcn = HamiltonianHookFunction(self)

        td_x = integrator.solve(self, mu, t_0, t_end, hook_fcns=Ham_hook_fcn)

        return td_x, Ham_hook_fcn.output
