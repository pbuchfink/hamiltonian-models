'''
    Demo file for linear wave equation.
    Visualization files are exported as vtr which can be opened with Paraview
        best visualiztion results with the filter 'Plot Over Line'
'''
import argparse, sys
import numpy as np
from hamiltonian_models.integrators.implicit_midpoint import ImplicitMidpointIntegrator
from hamiltonian_models.integrators.stormer_verlet import SeparableStormerVerletIntegrator
from hamiltonian_models.wave import OscillatingModeLinearWaveProblem, SineGordonProblem, TravellingBumpLinearWaveProblem, TravellingFrontLinearWaveProblem, TravellingWaveClosedFormulaIntegrator

parser = argparse.ArgumentParser()

parser.add_argument(
    '--integrator',
    help='Choose integrator.',
    choices=['implicit_midpoint', 'stoermer_verlet_q', 'stoermer_verlet_p'],
    default='implicit_midpoint'
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.integrator == 'implicit_midpoint':
        integrator = ImplicitMidpointIntegrator(1.)
    elif args.integrator == 'stoermer_verlet_q':
        integrator = SeparableStormerVerletIntegrator(1., use_staggered='q')
    elif args.integrator == 'stoermer_verlet_p':
        integrator = SeparableStormerVerletIntegrator(1., use_staggered='p')
    else:
        raise ValueError('Unknown integrator: {}'.format(args.integrator))

    ## linear wave equation, standing wave with one fixed node on each boundary
    l = 1
    n_x = 500
    model = OscillatingModeLinearWaveProblem(l, n_x)
    mu = {'c': .5}
    # compute solution for all t in [0, T]
    T, dt = 10, .01
    integrator._dt = dt
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/lin_wave_mode_oscillation.pvd', td_x)

    ## linear wave equation with transport-dominated behaviour
    ## due to localized support in the initial value
    l = 1
    n_x = 514
    model = TravellingBumpLinearWaveProblem(l, n_x)
    mu = {'c': 0.45, 'q0_supp': 0.45}
    # compute solution for all t in [0, T]
    T, dt = 1, 1/100
    integrator._dt = dt
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/lin_wave_transport.pvd', td_x)
    ## exlicit solution
    integrator = TravellingWaveClosedFormulaIntegrator(dt)
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/lin_wave_transport_expl.pvd', td_x)

    ## linear wave equation with transport-dominated behaviour
    l = 1
    n_x = 4*512+2
    model = TravellingFrontLinearWaveProblem(l, n_x)
    mu = {'ramp_width': 0.4, 'c': 0.4}
    # compute solution for all t in [0, T]
    T, dt = 1, 1/4000
    integrator._dt = dt
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/lin_wave_front.pvd', td_x)
    ## exlicit solution
    dt_expl = np.diff(td_x[::50]._t[:2])[0]
    integrator = TravellingWaveClosedFormulaIntegrator(dt_expl)
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/lin_wave_front_expl.pvd', td_x)

    ## sine-Gordon wave equation
    ## parameters from Peng and Mohseni, Structure-Preserving Model Reduction of Forced Hamiltonian Systems, 2016
    n_x = 4*512+2
    model = SineGordonProblem(n_x)
    mu = {'c': 1, 'v': .2}
    # compute solution for all t in [0, T]
    T, dt = 150, .0125
    integrator._dt = dt
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/sine_gordon.pvd', td_x[::50])
    ## exlicit solution
    dt_expl = np.diff(td_x[::50]._t[:2])[0]
    dt_expl = 0.625
    integrator = TravellingWaveClosedFormulaIntegrator(dt_expl)
    td_x, td_Ham = model.solve(0, T, integrator, mu)
    # plot solution
    model.visualize('paraview/sine_gordon_expl.pvd', td_x)
