'''Störmer-Verlet integration for separable Hamiltonians.
Required properties and methods:
    properties:
        is_separable = True
            This is the case, when the Hamiltonian can be written as
                H(q, p) = K(p) + U(q)
            or for time-dependent systems
                H(t, q, p) = K(p) + U(t, q)
    methods
        gradHam
            the gradient of the Hamiltonian
Optional properties and methods:
    methods:
        preassemble
            if present, the operators are preassembled
'''

import numpy as np
from hamiltonian_models.integrators.base import FixedTimeStepWidthIntegrator, HookFunction
from hamiltonian_models.vectors import PhaseSpaceVectorList

class SeparableStormerVerletIntegrator(FixedTimeStepWidthIntegrator):
    '''Explicit time integration with the Störmer-Verlet scheme for separable system.
    Two formulations are implemented
        1.) q staggered (with intermediate q_{i+1/2} step)
        1.) p staggered (with intermediate p_{i+1/2} step)
    '''
    is_explicit = True
    def __init__(self, dt, use_staggered='q', verbose=True):
        '''
        use_staggered
            Decides, whether to use a staggered grid for q or p
        '''
        super().__init__(dt, verbose)
        assert use_staggered in ('q', 'p')
        self.use_staggered = use_staggered

    @staticmethod
    def is_applicable(dyn_sys):
        return hasattr(dyn_sys, 'gradHam') \
            and hasattr(dyn_sys, 'is_separable') and dyn_sys.is_separable

    def check_stability(self, max_freq):
        '''Checks stability of the system based on the highest frequency which can be computed from a generalized eigenvalue problem.
        max_freq
            highest frequency in the system.
        '''
        return self._dt <= 4 / max_freq

    def solve(self, dyn_sys, mu, t_0, t_end, hook_fcns=()):
        super().solve(dyn_sys, mu, t_0, t_end, hook_fcns)
        if isinstance(hook_fcns, HookFunction):
            hook_fcns = (hook_fcns,)

        dt = self._dt
        sol_t = np.arange(t_0, t_end, dt)
        nt = len(sol_t)
        td_x = PhaseSpaceVectorList()
        mu = dyn_sys.update_mu(mu, {'_t': sol_t[0]})
        td_x.append(t_0, dyn_sys.initial_value(mu))

        # setup all hook functions
        for hook_fcn in hook_fcns:
            hook_fcn.setup(td_x._data[-1], mu, nt)

        # decide q or p staggered
        if self.use_staggered == 'q':
            stormer_verlet_step = self._stormer_verlet_step_q_staggered
        elif self.use_staggered == 'p':
            stormer_verlet_step = self._stormer_verlet_step_p_staggered

        # compute solution for every time step
        for i, t in enumerate(sol_t[:-1]):
            _, x_old = td_x[-1] #TODO: neat method to avoid access to _data

            x_new = stormer_verlet_step(dyn_sys, x_old.copy(), mu, dt)
            mu = dyn_sys.update_mu(mu, {'_t': t + dt})

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

    def _stormer_verlet_step_q_staggered(self, dyn_sys, x, mu, dt):
        # compute q_{n+1/2}
        x.vec_q += dt/2 * dyn_sys.gradHam(x, mu).vec_p
        mu = dyn_sys.update_mu(mu, {'_t': mu['_t'] + dt/2})
        # compute p_{n+1}
        x.vec_p -= dt * dyn_sys.gradHam(x, mu).vec_q
        # compute q_{n+1}
        x.vec_q += dt/2 * dyn_sys.gradHam(x, mu).vec_p
        # mu = dyn_sys.update_mu(mu, {'_t': mu['_t'] + dt/2}) #not required as mu is update in loop

        return x

    def _stormer_verlet_step_p_staggered(self, dyn_sys, x, mu, dt):
        # compute p_{n+1/2}
        x.vec_p -= dt/2 * dyn_sys.gradHam(x, mu).vec_q
        # compute q_{n+1}
        x.vec_q += dt * dyn_sys.gradHam(x, mu).vec_p
        mu = dyn_sys.update_mu(mu, {'_t': mu['_t'] + dt})
        # compute p_{n+1}
        x.vec_p -= dt/2 * dyn_sys.gradHam(x, mu).vec_q
        
        return x

    def residual(self, dyn_sys, x_new, x_old, mu):
        if self.use_staggered == 'q':
            return x_new - self._stormer_verlet_step_q_staggered(dyn_sys, x_old.copy(), mu, self._dt)
        elif self.use_staggered == 'p':
            return x_new - self._stormer_verlet_step_p_staggered(dyn_sys, x_old.copy(), mu, self._dt)

    def apply_inv_jacobian(self, dyn_sys, x_new, x_old, mu, res):
        return res
