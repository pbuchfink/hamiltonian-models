# Demo for simple TIME-INDEPENDENT Hamiltonian Systems

import numpy as np
import matplotlib.pyplot as plt
from hamiltonian_models.lowdim import HarmonicOscillator, HarmonicOscillatorClosedFormulaIntegrator, SimplePendulum
from hamiltonian_models.integrators.base import TimeDataList
from hamiltonian_models.integrators.implicit_midpoint import ImplicitMidpointIntegrator
from hamiltonian_models.integrators.stormer_verlet import SeparableStormerVerletIntegrator

def plot_Ham_sys(td_xs, td_Hams, titles, suptitle):
    if not isinstance(td_xs, (list, tuple,)):
        td_xs = (td_xs,)
        td_Hams = (td_Hams,)
        titles = (titles,)
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    # plot phase-space diagram and Hamiltonian over time
    legend = []
    for td_x, td_Ham, title in zip(td_xs, td_Hams, titles):
        hp = ax[0].plot(list(td_x.all_vec_q()), list(td_x.all_vec_p()), label=title + ' solution trajectory')
        ax[0].scatter(td_x._data[0].vec_q, td_x._data[0].vec_p, marker='o', label=title + ' initial value', color=hp[0].get_color())
        ax[0].scatter(td_x._data[-1].vec_q, td_x._data[-1].vec_p, marker='s', label=title + ' final value', color=hp[0].get_color())
        ax[1].plot(td_Ham._t, td_Ham._data - td_Ham._data[0])
    # phase-space diagram description
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('phase-space diagram')
    ax[0].set_xlabel(r'displacement $q(t)$')
    ax[0].set_ylabel(r'momentum $p(t)$')
    ax[0].legend()
    # Hamiltonian over time description
    ax[1].set_title('Hamiltonian vs. time')
    ax[1].set_xlabel(r'time $t$')
    ax[1].set_ylabel(r'Ham. $H(x(t,\mu), \mu) - H(x(t_0,\mu), \mu)$')
    ax[1].set_yscale('log')
    # overall title
    fig.suptitle(suptitle, fontsize=16)
    plt.show()

if __name__ == "__main__":
    # choose numerical integrator
    num_integrator_class = SeparableStormerVerletIntegrator
    # num_integrator_class = ImplicitMidpointIntegrator

    ## Harmonic oscillator
    model = HarmonicOscillator()
    mu = {'m': 1., 'k': 1., 'f': 0., 'q0': 1., 'p0': 0.}
    dt = np.pi/1e3
    # compute solution for all t in [0, pi] with closed formula
    td_x_closed, td_Ham_closed = model.solve(0, np.pi, HarmonicOscillatorClosedFormulaIntegrator(dt), mu)
    # compute solution for all t in [0, pi] with numerical integrator
    num_integrator = num_integrator_class(dt) # , use_staggered='p'
    td_x_num, td_Ham_num = model.solve(0, np.pi, num_integrator, mu)
    # plot solution
    plot_Ham_sys([td_x_closed, td_x_num], [td_Ham_closed, td_Ham_num], ['closed formula', str(num_integrator)], 'Harmonic oscillator')

    ## Simple pendulum, swinging case
    model = SimplePendulum()
    mu = {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi/2, 'p0': 0}
    dt = 4/2e3
    # compute solution for all t in [0, 4]
    num_integrator = num_integrator_class(dt)
    td_x, td_Ham = model.solve(0, 4, num_integrator, mu)
    # plot solution
    plot_Ham_sys(td_x, td_Ham, str(num_integrator), 'Simple pendulum, swinging case')

    ## Simple pendulum, rotating case
    model = SimplePendulum()
    mu = {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi, 'p0': 1.}
    # compute solution for all t in [0, 4]
    num_integrator = num_integrator_class(dt)
    td_x, td_Ham = model.solve(0, 4, num_integrator, mu)
    # plot solution
    plot_Ham_sys(td_x, td_Ham, str(num_integrator), 'Simple pendulum, rotating case')

