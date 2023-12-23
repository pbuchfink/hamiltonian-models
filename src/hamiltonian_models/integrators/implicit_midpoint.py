'''Implicit midpoint integrator for linear and non-linear dynamical systems.
    It numerically integrates the (parametric) initial value problem (IVP)

        M(mu) * (d/dt x(t, mu)) = f(x(t, mu), mu),
                      x(t0, mu) = x0(mu)

    with the implicit midpoint scheme (and Newton's method, if the system is non-linear). Also mu might depend on t (for non-autonomous systems).
    The underlying equations are
        1.) implicit midpoint scheme solves the IVP iteratively (iteration index i)
            x_0 = x0(mu)
            M * (x_{i+1} - x_i) = dt * f(x_{i+1/2}, mu_{i+1/2})
        where x_{i+1/2} = (x_i + x_{i+1}) / 2 and similarly mu for the time t
        2.) Newton's method solves
            r_i(x_{i+1}) = 0
        for the implicit midpoint residual
            r_i(x) = M * (x - x_i) - dt * f((x - x_i)/2, mu_{i+1/2})
        iteratively (iteration index k) with
            x_{i+1}^{(k+1)} = x_{i+1}^{(k)} - inv(Dr_i(x_{i+1}^{(k)}, mu_{i+1/2})) * r_i(x_{i+1}^{(k)}, mu_{i+1/2})
        where Dr_i(x, mu) is the Jacobian with respect to x of the residual r_i
            Dr_i(x, mu)*y = M * y - dt/2 * Df(x, mu)*y
        where Df(x,mu) is the Jacobian with respect to x of f.

Required properties and methods:
    properties:
        is_linear = True / False
    methods
        _inv_M_dt_Df_y(self, x, mu, dt, y)
            returns the evlauation of the term
                inv(M - dt/2 * Df(x,mu)) * y
        _mass_y(self, mu, y)
            returns the evaluation of the term
                M(mu) * y

            
Optional properties and methods:
    methods:
        preassemble
            if present, the operators are preassembled
'''

import warnings
import numpy as np
from hamiltonian_models.integrators.base import FixedTimeStepWidthIntegrator, HookFunction
from hamiltonian_models.vectors import PhaseSpaceVectorList

class ImplicitMidpointIntegrator(FixedTimeStepWidthIntegrator):
    '''Implicit time integration with the implicit midpoint scheme.
    '''
    is_explicit = False
    def __init__(self, dt, verbose=True, n_newton_max=100, abs_tol_newton=1e-3):
        super().__init__(dt, verbose)
        self.n_newton_max = n_newton_max
        self.abs_tol_newton = abs_tol_newton

    @staticmethod
    def is_applicable(dyn_sys):
        return hasattr(dyn_sys, '_inv_M_dt_Df_y') \
            and (dyn_sys.is_linear or hasattr(dyn_sys, '_mass_y'))

    def solve(self, dyn_sys, mu, t_0, t_end, hook_fcns=()):
        super().solve(dyn_sys, mu, t_0, t_end, hook_fcns)
        if isinstance(hook_fcns, HookFunction):
            hook_fcns = (hook_fcns,)

        dt = self._dt
        sol_t = np.arange(t_0, t_end, dt)
        nt = len(sol_t)
        td_x = PhaseSpaceVectorList()
        mu = dyn_sys.update_mu(mu, {'_t': sol_t[0] + dt/2})
        td_x.append(t_0, dyn_sys.initial_value(mu))

        # setup all hook functions
        for hook_fcn in hook_fcns:
            hook_fcn.setup(td_x._data[-1], mu, nt)

        # compute solution for every time step
        for i,t in enumerate(sol_t[:-1]):
            x_old = td_x._data[-1]
            x_new = x_old.copy()
            
            mu = dyn_sys.update_mu(mu, {'_t': t + dt})
            # since x_new = x_old the commands
            #   res = dyn_sys._mass_y(mu, x_new - x_old) + dyn_sys.dxdt((x_new+x_old) * .5, mu) * (-dt)
            # simplify to
            res = dyn_sys.dxdt(x_new, mu) * (-dt)

            # solve for x_new
            if dyn_sys.is_linear:
                x_new -= self.apply_inv_jacobian(dyn_sys, x_new, x_old, mu, res)
            else:
                # apply Newton's method to find solution of current time step
                it_new = 0

                while res.l2_norm() > self.abs_tol_newton and it_new < self.n_newton_max:
                    x_new -= self.apply_inv_jacobian(dyn_sys, x_new, x_old, mu, res)
                    res = self.residual(dyn_sys, x_new, x_old, mu)
                    it_new += 1

                if it_new == self.n_newton_max:
                    warnings.warn('Maximal number of Newton iterations reached!', RuntimeWarning)

            # append solution
            td_x.append(t+dt, x_new)

            # evaluate each hook_fcn
            for hook_fcn in hook_fcns:
                hook_fcn.eval(td_x._data[-1], mu)

            self.print_progress(i-1, i, nt)

        for hook_fcn in hook_fcns:
            hook_fcn.finalize()
        
        self.print_progress(nt-1, nt, nt)

        return td_x

    def residual(self, dyn_sys, x_new, x_old, mu):
        return dyn_sys._mass_y(mu, x_new - x_old) + dyn_sys.dxdt((x_new+x_old) * .5, mu) * (-self._dt)

    def apply_inv_jacobian(self, dyn_sys, x_new, x_old, mu, res):
        return dyn_sys._inv_M_dt_Df_y((x_new+x_old) * .5, mu, self._dt, res)

    #TODO: integrate more into the framework
    def apply_jacobian(self, dyn_sys, x_new, x_old, mu, y):
        return dyn_sys._mass_y(mu, y) + dyn_sys.apply_dxdt_jacobian((x_new+x_old) * .5, mu, y) * (-self._dt / 2)
