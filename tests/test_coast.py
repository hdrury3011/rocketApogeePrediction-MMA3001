import numpy as np
import pytest

from apogee import analytic_apogee_height, analytic_time_to_apogee
from atmosphere import constant_atmosphere, isa
from coast import simulate_apogee, reference_apogee, make_rhs
from config import S_REF
from drag import DragModel
from integrators import euler_step, rk4_step

G = 9.81
RHO = 1.1
CD0 = 0.42


def constant_cd(mach):
    return CD0


def k_of(mass):
    return RHO * CD0 * S_REF / (2 * mass)


def test_constant_properties_match_analytic_solution():
    """Verification: with constant rho and C_D the model reduces to the benchmark."""
    mass, v0, h0 = 30.0, 280.0, 548.64
    atm = constant_atmosphere(RHO, 340.0)
    t_a, h_a = simulate_apogee(mass, constant_cd, atm, v0, h0, dt=0.1)
    assert np.isclose(h_a - h0, analytic_apogee_height(v0, k_of(mass), G), rtol=1e-8)
    assert np.isclose(t_a, analytic_time_to_apogee(v0, k_of(mass), G), rtol=1e-8)


def test_reference_solver_agrees_with_analytic_solution():
    mass, v0, h0 = 30.0, 280.0, 548.64
    atm = constant_atmosphere(RHO, 340.0)
    t_a, h_a = reference_apogee(mass, constant_cd, atm, v0, h0)
    assert np.isclose(h_a - h0, analytic_apogee_height(v0, k_of(mass), G), rtol=1e-9)


def test_vacuum_limit():
    """Huge mass makes drag negligible: apogee tends to the drag-free value."""
    v0, h0 = 280.0, 548.64
    _, h_a = simulate_apogee(1e9, constant_cd, constant_atmosphere(RHO, 340.0), v0, h0, dt=0.1)
    assert np.isclose(h_a - h0, v0**2 / (2 * G), rtol=1e-6)


def test_heavier_rocket_flies_higher():
    atm = constant_atmosphere(RHO, 340.0)
    h = [simulate_apogee(m, constant_cd, atm, 280.0, 0.0)[1] for m in (10, 20, 40, 80)]
    assert all(np.diff(h) > 0)


def test_more_flap_lowers_apogee():
    mach = np.linspace(0.1, 0.9, 5)
    flap = np.array([0.0, 100.0])
    cd = np.column_stack([np.full(5, 0.4), np.full(5, 0.6)])
    model = DragModel(mach, flap, cd)
    apogees = [simulate_apogee(30.0, model, isa, 280.0, 548.64, flap=f)[1] for f in (0, 50, 100)]
    assert apogees[0] > apogees[1] > apogees[2]


def test_flap_schedule_is_used():
    """Deploying later gives an apogee between never and always deployed."""
    mach = np.linspace(0.1, 0.9, 5)
    flap = np.array([0.0, 100.0])
    model = DragModel(mach, flap, np.column_stack([np.full(5, 0.4), np.full(5, 0.6)]))
    never = simulate_apogee(30.0, model, isa, 280.0, 548.64, flap=0.0)[1]
    always = simulate_apogee(30.0, model, isa, 280.0, 548.64, flap=100.0)[1]
    late = simulate_apogee(30.0, model, isa, 280.0, 548.64,
                           flap=lambda t: 100.0 if t > 8.0 else 0.0)[1]
    assert always < late < never


def test_euler_less_accurate_than_rk4_with_variable_density():
    mass, v0, h0 = 30.0, 280.0, 548.64
    _, h_ref = reference_apogee(mass, constant_cd, isa, v0, h0)
    err_e = abs(simulate_apogee(mass, constant_cd, isa, v0, h0, dt=0.1, step=euler_step)[1] - h_ref)
    err_r = abs(simulate_apogee(mass, constant_cd, isa, v0, h0, dt=0.1, step=rk4_step)[1] - h_ref)
    assert err_r < 1e-3 * err_e


def test_evaluation_counter():
    counter = [0]
    simulate_apogee(30.0, constant_cd, isa, 280.0, 548.64, dt=1.0, counter=counter)
    assert counter[0] > 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_apogee(-1.0, constant_cd, isa, 280.0, 548.64)
    with pytest.raises(ValueError):
        simulate_apogee(30.0, constant_cd, isa, -5.0, 548.64)
    with pytest.raises(ValueError):
        make_rhs(0.0, constant_cd, isa)
