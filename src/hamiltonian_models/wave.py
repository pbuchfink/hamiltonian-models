import os
from os.path import splitext

import numpy as np
from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkGroup
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from hamiltonian_models.base import (CanonicalHamiltonianSystem,
                                     HamiltonianHookFunction,
                                     QuadraticHamiltonianHookFunction,
                                     QuadraticHamiltonianSystem, Vectorized)
from hamiltonian_models.integrators.base import (FixedTimeStepWidthIntegrator,
                                                 TimeDataList)
from hamiltonian_models.vectors import PhaseSpaceVectorList
from hamiltonian_models.vectors_numpy import NumpyPhaseSpace


class LinearWaveEquationProblem(QuadraticHamiltonianSystem, CanonicalHamiltonianSystem):
    """One-dimensional linear wave equation.

    The problem is to solve::

        ∂^2_(t^2) u(x, t, μ) = c^2 ∂^2_(x^2) u(x, t, μ)

    for an unknown, time-dependent displacement-field u(x, t, μ)
        on the interval [-l/2, l/2]
        for all time steps t ∈ [0, T]
        for a fixed (but arbitrary) parameter vector μ ∈ P in the parameter space P
    with the initial conditions

                    u(x, 0, μ) = u_0(x, μ)
                ∂_t u(x, 0, μ) = w_0(x, μ)

    for all x ∈ Ω and with homogeneous Dirichlet boundray conditions

                    u(x, t, μ) = 0, x ∈ {-l/2, l/2}

    for all t ∈ [0, T] where
        l is the length of the domain
        c is the (constant) wave speed
        u_0(x, μ) is the initial displacement
        w_0(x, μ) is the initial velocity

    Upon discretization with finite differences, the problem results in a second-order system of ODEs::

            d^2/dt^2 q(t, μ) = D_xx(μ) q(t, μ)
                     q(0, μ) = q_0(μ)
                d/dt q(0, μ) = v_0(μ)

    where Dirichlet conditions are implemented via index reduction of the corresponding algebraic equations.
    The system is reformulated to a quadratic Hamiltonian system with::

        ∂_t x(t, μ) = J * (H(μ) * x(t, μ) + h(t, μ))
            x(0, μ) = x_0(μ)

    with the first-order system matrices::

        J       = [[0, Identity], [-Identity, 0]]
        H(μ)    = [[-D_xx(μ), 0], [0, I]]
        h(t, μ) = 0
        x_0(μ)  = [q_0(μ), p_0(μ)]

    Parameters
    ----------
    l
        Length of the domain.
    n_x
        Number of elements in the axial direction of the beam.
    """
    is_separable = True
    def __init__(self, l, n_x):
        assert n_x >= 2
        self.l = l
        self.dx = l/(n_x-1)
        self.n_x = n_x
        self.grid = np.linspace(-l/2, l/2, n_x)
        super(LinearWaveEquationProblem, self).__init__(
            phase_space=NumpyPhaseSpace(2*n_x, 1)
        )
        self._parameters = {'c'}

    def preassemble(self, mu, dt):
        self._assembled_mu = mu

        # second-order central difference scheme
        central_fd = diags(
            [
                np.hstack([0, -2 * np.ones(self.n_x-2), 0]),
                np.hstack([0, np.ones(self.n_x-2)]),
                np.hstack([np.ones(self.n_x-2), 0])
            ],
            [0, 1, -1],
            format='csr',
        )

        self._assembled_Dxx = (mu['c'])**2/self.dx * central_fd
        self._assembled_M = self.dx * eye(self.n_x, format='csr')
        self._assembled_M_dt_Dxx = self._assembled_M - (dt/2)**2 * self._assembled_Dxx

    def H_op_y(self, mu, y):
        res = y.copy()
        res.vec_q = -y.vec_q @ self._assembled_Dxx.T
        res.vec_p = spsolve(self._assembled_M, y.vec_p.T).T
        return res

    def h_op(self, mu):
        h = self.phase_space.zeros()
        return h

    def H_product_y(self, mu, y):
        res = self.H_op_y(mu, y)
        res.vec_q[..., [0, -1]] += y.vec_q[..., [0, -1]]
        res.vec_p[..., [0, -1]] += y.vec_p[..., [0, -1]]
        return res

    def _inv_M_dt_Df_y(self, x, mu, dt, y):
        '''Returns the product
                inv(M(mu) - dt/2 * Df(x, mu)) * y
            where
                M(mu) is the first-order system mass matrix which is identity for non-Dirichlet DOFs and zero else
            and
                Df(x, mu) = J * [[-D_xx, 0],[0, I]] is the derivative of dxdt with respect to x
                and J = [[0,I], [-I,0]] is the canonical Poisson matrix
            The inverse is derived via LR decomposition of M(mu) - dt/2 * Df(x, mu) such that
                inv(M(mu) - dt/2 * Df(x, mu)) = inv(R_I_dt_A_impl) * inv(L_I_dt_A_impl)
        '''
        if x.ndim > 1 or y.ndim > 1:
            raise NotImplementedError('_inv_M_dt_Df_y is not vectorized yet.')

        assert self._assembled_mu and mu is self._assembled_mu

        # 1.) Solve
        #   L_I_dt_A_impl * res = y
        # with
        #   L_I_dt_A_impl = [[In,         Zn],
        #                    [-dt/2*D_xx, In]]
        res = y.copy()
        res.vec_p += dt/2 * self._assembled_Dxx @ res.vec_q

        y = res.copy()
        # 2.) Solve
        #   R_I_dt_A_impl * res = y
        # with
        #   R_I_dt_A_impl = [[In,              -dt/2 * In],
        #                    [Zn, M_so - (dt/2)**2 * D_xx]]
        res.vec_p = spsolve(self._assembled_M_dt_Dxx, y.vec_p)
        res.vec_q += dt/2 * res.vec_p

        y = res.copy()
        # 3.) Compute
        #   res = S * y
        # with
        #   S = block_diag([In, M_so])
        res.vec_p = self._assembled_M @ y.vec_p

        return res

    def M_dt_Df_y(self, x, mu, dt, y):
        '''Returns the product
                (M(mu) - dt/2 * Df(x, mu)) * y
            where
                M(mu) is the first-order system mass matrix which is identity for non-Dirichlet DOFs and zero else
            and
                Df(x, mu) = J * [[-D_xx, 0],[0, I]] is the derivative of dxdt with respect to x
                and J = [[0,I], [-I,0]] is the canonical Poisson matrix
        '''

        assert self._assembled_mu and mu is self._assembled_mu

        res = self.J_y(x, mu, self.hessianHam_y(x, mu, y)).scal(-dt/2)
        res += self._mass_y(mu, y)
        return res

    def _mass_y(self, mu, y):
        '''Returns the product
            M(mu) * y
        where
            M(mu) is the first-order system mass matrix which is identity for non-Dirichlet DOFs and zero else.
        This function is not used since the system is linear but the function is given for the sake of completion.
        '''
        return y

    def solve(self, t_0, t_end, integrator, mu):
        super().solve(t_0, t_end, integrator, mu)

        # stability check
        if integrator.is_explicit:
            assert integrator.check_stability(2 * mu['c'] / self.dx)

        if not isinstance(integrator, TravellingWaveClosedFormulaIntegrator):
            Ham_hook_fcn = QuadraticHamiltonianHookFunction(self)

            td_x = integrator.solve(self, mu, t_0, t_end, hook_fcns=Ham_hook_fcn)

            return td_x, Ham_hook_fcn.output
        else:
            return integrator.solve(self, mu, t_0, t_end)

    def visualize(self, filename, sol_x):
        '''
            Output results of solve() to pvd file (as input for paraview)
            for displacemnet and momentum.
        '''
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        split_filename = splitext(filename)
        n = int(np.ceil(np.log10(len(sol_x))))
        # plot displacement and momentum
        z = np.array([0])
        file_path = split_filename[0]
        g = VtkGroup(file_path)
        for i, (t, x) in enumerate(sol_x):
            file_name_i = gridToVTK(file_path + ('{0:0' + str(n) + 'd}').format(i), self.grid, z, z, pointData={"q" : x.vec_q, "p" : x.vec_p})
            g.addFile(filepath=file_name_i, sim_time=t)
        g.save()


class OscillatingModeLinearWaveProblem(LinearWaveEquationProblem):
    def initial_value(self, mu):
        super().initial_value(mu)

        x0 = self.phase_space.zeros()
        bump = lambda xi: np.cos(xi / self.l * np.pi)
        x0.vec_q = bump(self.grid) 
        return x0


class FixedEndsLinearWaveProblem(LinearWaveEquationProblem):
    def __init__(self, l, n_x):
        super().__init__(l, n_x)
        self._parameters.add('q0_supp')

    def initial_value(self, mu):
        super().initial_value(mu)
        assert 0 < mu['q0_supp'] <= self.l

        x0 = self.phase_space.zeros()
        bump = lambda xi, w: (xi >= -w/2) * (xi <= w/2) * 1/2 * (np.exp(-(6*xi)**2/w**2) - np.exp(-(6*w/2)**2/w**2))
        x0.vec_q = bump(self.grid, mu['q0_supp'])
        return x0

class TravellingBumpLinearWaveProblem(LinearWaveEquationProblem):
    def __init__(self, l, n_x):
        super().__init__(l, n_x)
        self._parameters.add('q0_supp')
    
    def initial_value_profile_as_function_handle(self, mu):
        '''Returns initial value profile and derivative as function handle.'''
        l=self.l
        h = lambda s: (0 <= s) * (s <= 1) * (1 - 3/2 * s**2 + 3/4 * s**3) + (1 < s) * (s <= 2) * ((2-s)**3)/4
        ds_h = lambda s: (0 <= s) * (s <= 1) * (- 3 * s + 9/4 * s**2) - (1 < s) * (s <= 2) * 3 * ((2-s)**2)/4
        bump = lambda xi: h(np.abs(4*(xi + l/2 - mu['q0_supp']/2)/mu['q0_supp']))
        ddt_bump = lambda xi: -4 * mu['c'] / mu['q0_supp'] \
            * ds_h(np.abs(4*(xi + l/2 - mu['q0_supp']/2)/mu['q0_supp'])) \
            * (-1 + 2*((xi + l/2 - mu['q0_supp']/2) > 0))
        return bump, ddt_bump

    def initial_value(self, mu):
        super().initial_value(mu)
        assert 0 < mu['q0_supp'] <= self.l

        bump, ddt_bump = self.initial_value_profile_as_function_handle(mu)
        x0 = self.phase_space.zeros()
        x0.vec_q = bump(self.grid)
        x0.vec_p = self._assembled_M @ ddt_bump(self.grid)
        x0.vec_p[0] = x0.vec_p[-1] = 0 #enforces Dirichlet zero
        return x0


class AnotherTravellingBumpLinearWaveProblem(LinearWaveEquationProblem):
    def initial_value_profile_as_function_handle(self, mu):
        '''Returns initial value profile and derivative as function handle.'''
        l=self.l
        h = lambda s: (0 <= s) * (s <= 1) * (1 - 3/2 * s**2 + 3/4 * s**3) + (1 < s) * (s <= 2) * ((2-s)**3)/4
        ds_h = lambda s: (0 <= s) * (s <= 1) * (- 3 * s + 9/4 * s**2) - (1 < s) * (s <= 2) * 3 * ((2-s)**2)/4
        bump = lambda xi: h(
            4*np.abs((xi + l/2 - mu['c']/2))
            / mu['c']
        )
        ddt_bump = lambda xi: -4 * (
            ds_h(
                4*np.abs((xi + l/2 - mu['c']/2))
                / mu['c']
            )
            * (-1 + 2 * ((xi + l/2 - mu['c']/2) > 0))
        )
        return bump, ddt_bump

    def initial_value(self, mu):
        super().initial_value(mu)

        bump, ddt_bump = self.initial_value_profile_as_function_handle(mu)
        x0 = self.phase_space.zeros()
        x0.vec_q = bump(self.grid)
        x0.vec_p = self._assembled_M @ ddt_bump(self.grid)
        x0.vec_p[0] = x0.vec_p[-1] = 0 #enforces Dirichlet zero
        return x0


class TravellingFrontLinearWaveProblem(LinearWaveEquationProblem):
    def __init__(self, l, n_x):
        super().__init__(l, n_x)
        self._parameters.add('ramp_width')
    
    def initial_value_profile_as_function_handle(self, mu):
        '''Returns initial value profile and derivative as function handle.'''
        l = self.l
        # kink
        #
        #   \
        #    \
        #     \
        #    _ \__________________________
        #   |   |                         |
        # -l/2  -l/2 + mu['ramp_width]   l/2
        h = lambda s: (0 <= s) * (s <= 1) * (1 - 3/2 * s**2 + 3/4 * s**3) + (1 < s) * (s <= 2) * ((2-s)**3)/4
        ds_h = lambda s: (0 <= s) * (s <= 1) * (- 3 * s + 9/4 * s**2) - (1 < s) * (s <= 2) * 3 * ((2-s)**2)/4
        front = lambda xi: -h(2 - 2*(xi + l/2) / mu['ramp_width']) + ((xi + l/2) <= mu['ramp_width'])
        ddt_front = lambda xi: -2*mu['c']/mu['ramp_width'] * ds_h(2 - 2*(xi + l/2) / mu['ramp_width'])
        return front, ddt_front

    def initial_value(self, mu):
        super().initial_value(mu)
        assert 0 < mu['ramp_width'] <= self.l

        front, ddt_front = self.initial_value_profile_as_function_handle(mu)
        x0 = self.phase_space.zeros()
        x0.vec_q = front(self.grid)
        x0.vec_p = self._assembled_M @ ddt_front(self.grid)
        x0.vec_p[0] = x0.vec_p[-1] = 0 #enforces Dirichlet zero
        return x0

class SineGordonProblem(CanonicalHamiltonianSystem,Vectorized):
    """One-dimensional sine-Gordon equation. A non-liear wave problem.

    The problem is to solve::

        ∂^2_(t^2) u(x, t, μ) = c^2 ∂^2_(x^2) u(x, t, μ) - g(u(x, t, μ))

    for an unknown, time-dependent displacement-field u(x, t, μ)
        on the interval [-25, 25]
        for all time steps t ∈ [0, T]
        for a fixed (but arbitrary) parameter vector μ ∈ P in the parameter space P
        with g(u) = sin(u)
    with the initial conditions

                    u(x, 0, μ) = u_0(x, μ)
                ∂_t u(x, 0, μ) = w_0(x, μ)

    for all x ∈ Ω and with Dirichlet boundray conditions

                    u(-25, t, μ) = 0, 
                    u( 25, t, μ) = 2*pi,

    for all t ∈ [0, T] where
        c is the (constant) wave speed
        u_0(x, μ) is the initial displacement
        w_0(x, μ) is the initial velocity

    Upon discretization with finite differences, the problem results in a second-order system of ODEs::

            d^2/dt^2 q(t, μ) = D_xx(μ) q(t, μ) - sin(q(t, μ))
                     q(0, μ) = q_0(μ)
                d/dt q(0, μ) = v_0(μ)

    where Dirichlet conditions are implemented via index reduction of the corresponding algebraic equations.
    The system is reformulated to a quadratic Hamiltonian system with::

        ∂_t x(t, μ) = J * (H(μ) * x(t, μ) + h(x(t, μ)))
            x(0, μ) = x_0(μ)

    with the first-order system matrices::

        J          = [[0, Identity], [-Identity, 0]]
        H(μ)       = [[-D_xx(μ), 0], [0, I]]
        h(x(t, μ)) = [sin(q(t, μ)), 0]
        x_0(μ)     = [q_0(μ), p_0(μ)]

    Parameters
    ----------
    n_x
        Number of elements in the axial direction of the beam.
    """
    is_separable = True
    def __init__(self, n_x):
        assert n_x >= 2
        self.l = 50 # initial value does not work otherwise
        self.dx = self.l/(n_x-1)
        self.n_x = n_x
        self.grid = np.linspace(-self.l/2, self.l/2, n_x)
        super(SineGordonProblem, self).__init__(
            phase_space=NumpyPhaseSpace(2*n_x, 1),
            is_linear=False
        )
        self._parameters = {'v'}

    def preassemble(self, mu, dt):
        self._assembled_mu = mu

        # second-order central difference scheme
        central_fd = diags(
            [
                np.hstack([0, -2 * np.ones(self.n_x-2), 0]),
                np.hstack([0, np.ones(self.n_x-2)]),
                np.hstack([np.ones(self.n_x-2), 0])
            ],
            [0, 1, -1],
            format='csr',
        )

        self._assembled_Dxx = 1/self.dx * central_fd
        self._assembled_M = self.dx * eye(self.n_x, format='csr')
        self._assembled_M_dt_Dxx = self._assembled_M - (dt/2)**2 * self._assembled_Dxx

    def H_op_y(self, mu, y):
        res = y.copy()
        res.vec_q = -y.vec_q @ self._assembled_Dxx.T
        res.vec_p = spsolve(self._assembled_M, y.vec_p.T).T
        return res

    def gradHam(self, x, mu):
        res = self.H_op_y(mu, x)
        res.vec_q[..., 1:-1] += self.dx * np.sin(x.vec_q[..., 1:-1]) # only in interior, not on Dirichlet nodes
        return res

    def Ham(self, x, mu):
        super().Ham(x, mu)
        if x.ndim == 2:
            return [self.Ham(x_elem, mu) for x_elem in x]
        elif x.ndim == 1:
            return 1/2 * self.H_op_y(mu, x).dot(x) + self.dx * np.sum(1 - np.cos(x.vec_q[1:-1]))
        else:
            raise NotImplementedError()

    def hessianHam_y(self, x, mu, y):
        res = self.H_op_y(mu, y)
        res.vec_q[..., 1:-1] += self.dx * np.cos(x.vec_q[..., 1:-1]) * y.vec_q[..., 1:-1]
        return res

    def H_product_y(self, mu, y):
        res = self.H_op_y(mu, y)
        res.vec_q[..., [0, -1]] += y.vec_q[..., [0, -1]]
        res.vec_p[..., [0, -1]] += y.vec_p[..., [0, -1]]
        return res
    
    def initial_value_profile_as_function_handle(self, mu):
        '''Returns initial value profile and derivative as function handle.'''
        l = self.l
        # kink
        #              ____________________
        #             /
        #            /
        #           /
        #    ______/
        #   |    |    |    |              |
        # -l/2 -l/3 -l/6   0             l/2
        xi_0 = -l/4
        denom = np.sqrt(1-mu['v']**2)
        kink = lambda xi: 4 * np.arctan(np.exp((xi - xi_0)/denom))
        ddt_kink = lambda xi: -4 * mu['v']/denom * np.exp((xi - xi_0) / denom) / (1 + np.exp((xi - xi_0) / denom)**2) * (-l/2 < xi) * (xi < l/2)
        return kink, ddt_kink

    def initial_value(self, mu):
        super().initial_value(mu)
        
        kink, ddt_kink = self.initial_value_profile_as_function_handle(mu)
        x0 = self.phase_space.zeros()
        x0.vec_q = kink(self.grid)
        x0.vec_p = self._assembled_M @ ddt_kink(self.grid)
        return x0

    def _inv_M_dt_Df_y(self, x, mu, dt, y):
        '''Returns the product
                inv(M(mu) - dt/2 * Df(x, mu)) * y
            where
                M(mu) is the first-order system mass matrix which is identity for non-Dirichlet DOFs and zero else
            and
                Df(x, mu) = J * [[-D_xx + COS(q), 0],[0, I]] is the derivative of dxdt with respect to x
                and J = [[0,I], [-I,0]] is the canonical Poisson matrix
                and COS(q) = diag([0, cos(q_1), ..., cos(q_n), 0]) is a matrix with cos(q_i) on the i-th diagonal entry for inner nodes
            The inverse is derived via LR decomposition of M(mu) - dt/2 * Df(x, mu) such that
                inv(M(mu) - dt/2 * Df(x, mu)) = inv(R_I_dt_A_impl) * inv(L_I_dt_A_impl)
        '''
        if x.ndim > 1 or y.ndim > 1:
            raise NotImplementedError('_inv_M_dt_Df_y is not vectorized yet.')

        assert self._assembled_mu and mu is self._assembled_mu

        COS = self.dx * diags(np.hstack([0, np.cos(x.vec_q[1:-1]), 0]))
        D_xx_COS = self._assembled_Dxx - COS
        M_dt_D_xx_COS = self._assembled_M_dt_Dxx + (dt/2)**2 * COS

        # 1.) Solve
        #   L_I_dt_A_impl * res = y
        # with
        #   L_I_dt_A_impl = [[In,                    Zn],
        #                    [-dt/2*(D_xx - COS(q)), In]]
        res = y.copy()
        res.vec_p += dt/2 * D_xx_COS @ res.vec_q

        y = res.copy()
        # 2.) Solve
        #   R_I_dt_A_impl * res = y
        # with
        #   R_I_dt_A_impl = [[In,                          -dt/2 * In],
        #                    [Zn, M_so - (dt/2)**2 * (D_xx - COS(q))]]
        res.vec_p = spsolve(M_dt_D_xx_COS, y.vec_p)
        res.vec_q += dt/2 * res.vec_p

        y = res.copy()
        # 3.) Compute
        #   res = S * y
        # with
        #   S = block_diag([In, M_so])
        res.vec_p = self._assembled_M @ y.vec_p

        return res

    def M_dt_Df_y(self, x, mu, dt, y):
        '''Returns the product
                (M(mu) - dt/2 * Df(x, mu)) * y
            where
                M(mu) is the first-order system mass matrix which is identity for non-Dirichlet DOFs and zero else
            and
                Df(x, mu) = J * [[-D_xx, 0],[0, I]] is the derivative of dxdt with respect to x
                and J = [[0,I], [-I,0]] is the canonical Poisson matrix
        '''

        assert self._assembled_mu and mu is self._assembled_mu

        res = self.J_y(x, mu, self.hessianHam_y(x, mu, y)).scal(-dt/2)
        res += self._mass_y(mu, y)
        return res

    def _mass_y(self, mu, y):
        '''Returns the product
            M(mu) * y
        where
            M(mu) is the first-order system mass matrix which is identity for non-Dirichlet DOFs and zero else.
        This function is not used since the system is linear but the function is given for the sake of completion.
        '''
        return y

    def solve(self, t_0, t_end, integrator, mu):
        super().solve(t_0, t_end, integrator, mu)

        # stability check of linear part of the equations!
        if integrator.is_explicit:
            assert integrator.check_stability(2 * 1 / self.dx)

        if not isinstance(integrator, TravellingWaveClosedFormulaIntegrator):
            Ham_hook_fcn = HamiltonianHookFunction(self)

            td_x = integrator.solve(self, mu, t_0, t_end, hook_fcns=Ham_hook_fcn)

            return td_x, Ham_hook_fcn.output
        else:
            return integrator.solve(self, mu, t_0, t_end)

    def visualize(self, filename, sol_x):
        '''
            Output results of solve() to pvd file (as input for paraview)
            for displacemnet and momentum.
        '''
        split_filename = splitext(filename)
        n = int(np.ceil(np.log10(len(sol_x))))
        # plot displacement and momentum
        z = np.array([0])
        file_path = split_filename[0]
        g = VtkGroup(file_path)
        for i, (t, x) in enumerate(sol_x):
            file_name_i = gridToVTK(file_path + ('{0:0' + str(n) + 'd}').format(i), self.grid, z, z, pointData={"q" : x.vec_q, "p" : x.vec_p})
            g.addFile(filepath=file_name_i, sim_time=t)
        g.save()

class TravellingWaveClosedFormulaIntegrator(FixedTimeStepWidthIntegrator):
    @staticmethod
    def is_applicable(dyn_sys):
        return isinstance(dyn_sys, (TravellingBumpLinearWaveProblem, TravellingFrontLinearWaveProblem, SineGordonProblem))

    def solve(self, dyn_sys, mu, t_0, t_end, hook_fcns=None):
        super().solve(dyn_sys, mu, t_0, t_end, hook_fcns)
        assert hook_fcns is None

        sol_t = np.arange(t_0, t_end, self._dt)
        x0 = dyn_sys.initial_value(mu)
        grid = dyn_sys.grid
        q0, v0 = dyn_sys.initial_value_profile_as_function_handle(mu)
        if isinstance(dyn_sys, SineGordonProblem):
            wave_speed = mu['v']
        else:
            wave_speed = mu['c']

        # batched computation
        xi_values = grid.reshape((1, -1)) - wave_speed * sol_t.reshape((-1, 1))
        q_values = q0(xi_values)
        p_values = np.ascontiguousarray((dyn_sys._assembled_M @ v0(xi_values).T).T)

        # sanity check
        assert abs(x0.vec_q - q_values[0]).max() == 0
        assert abs(x0.vec_p - p_values[0]).max() == 0

        td_x = PhaseSpaceVectorList()
        td_Ham = TimeDataList()
        phase_space = dyn_sys.phase_space
        nt = len(sol_t)
        for i, t in enumerate(sol_t):
            td_x.append(
                t,
                phase_space.new_vector(q_values[i], p_values[i])
            )
            td_Ham.append(t, dyn_sys.Ham(td_x._data[-1], mu))

            self.print_progress(i-1, i, nt)

        self.print_progress(nt-1, nt, nt)

        return td_x, td_Ham
