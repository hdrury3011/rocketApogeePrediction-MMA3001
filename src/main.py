"""Entry point: runs the whole apogee-prediction study end to end.

Usage (from the repository root, with the virtual environment active)::

    python src/main.py --data "data/Project Data Set.csv"

Optional: add scattered (noisy) CFD runs with ``--extra-data PATH``, or pass a
merged csv that has a 'Source' column as ``--data``; section 9 then fits a
regression drag model to all samples.

Steps
-----
1. Verification: integrators against the analytical constant-k solution.
2. Convergence and cost of Euler and RK4 on the full model (ISA + C_D table).
3. Calibration of the unknown rocket mass to the 3450 m no-airbrake apogee.
4. Validation against the 11000 ft airbrake apogee and flap/deployment study.
5. Sensitivity, modelling-assumption and performance studies.
6. Angle-of-attack sensitivity: apogee, calibrated mass and airbrake
   authority at the pitch angles whose CFD table is complete.
7. Regression drag model fitted to all CFD samples (needs extra data).

Figures are saved to <outdir>/figures and a results summary to
<outdir>/results.md.
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter

from apogee import analytic_apogee_height
from atmosphere import constant_atmosphere, isa
from calibration import bisect, calibrate_mass, flap_for_apogee
from coast import reference_apogee, simulate_apogee, trajectory
from config import (APOGEE_NO_BRAKES, APOGEE_WITH_BRAKES, G, H_BURNOUT,
                    S_REF, V_BURNOUT)
from drag import (GRID_SOURCE, DragModel, cd_roll_spread, cd_vs_pitch,
                  complete_pitches, load_cd_points, load_cd_table)
from integrators import euler_step, rk4_step
from regression import (RegressionDragModel, cross_validate, select_degrees,
                        source_offset)

# Palette validated with the dataviz skill's validate_palette.js (light mode).
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply a quiet, consistent look: recessive grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    # Log axes: keep decade labels only (minor tick labels overlap).
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def new_figure(plt, ncols=1, width=7.0, height=4.4):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0]


def timed(fn, repeats=3):
    """Run fn() several times; return (result, best wall time in seconds)."""
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def md_table(headers, rows):
    """Markdown table with every column padded to one width.

    Columns whose body cells are all numbers are right-aligned, the rest
    left-aligned, so the file reads as a table in a plain text editor as well
    as when rendered.

    Raises
    ------
    ValueError
        If a row does not have one cell per header.
    """
    heads = [str(h) for h in headers]
    body = [[str(c) for c in r] for r in rows]
    if any(len(r) != len(heads) for r in body):
        raise ValueError("every row needs one cell per header")
    n = len(heads)
    widths = [max([3, len(heads[j])] + [len(r[j]) for r in body]) for j in range(n)]
    numeric = [bool(body) and all(_is_number(r[j]) for r in body) for j in range(n)]

    def line(cells):
        return '| ' + ' | '.join(c.rjust(w) if num else c.ljust(w)
                                  for c, w, num in zip(cells, widths, numeric)) + ' |'

    rule = '| ' + ' | '.join('-' * (w - 1) + ':' if num else '-' * w
                             for w, num in zip(widths, numeric)) + ' |'
    return '\n'.join([line(heads), rule] + [line(r) for r in body])


def observed_orders(dts, errs):
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1]))
            for i in range(len(errs) - 1)]


# ---------------------------------------------------------------------------
# 1. Verification against the analytical solution
# ---------------------------------------------------------------------------
def run_verification(plt, figdir):
    """Euler/RK4 error against the closed-form apogee (constant rho and C_D)."""
    rho, cd0, k = 1.225, 0.40, 8.8*10**-5
    mass = rho * cd0 * S_REF / (2 * k)                    # gives k exactly
    atm = constant_atmosphere(rho, 340.29)
    exact = analytic_apogee_height(V_BURNOUT, k, G)
    dts = [0.4 * 0.5**n for n in range(8)]

    out = {}
    for name, step in (('Euler', euler_step), ('RK4', rk4_step)):
        errs, evals = [], []
        for dt in dts:
            counter = [0]
            _, h = simulate_apogee(mass, lambda m: cd0, atm, V_BURNOUT,
                                   H_BURNOUT, dt=dt, step=step, counter=counter)
            errs.append(abs((h - H_BURNOUT) - exact))
            evals.append(counter[0])
        out[name] = dict(errs=errs, evals=evals)

    if plt is not None:
        fig, (ax1, ax2) = new_figure(plt, 2, 10.5, 4.4)
        floor = 1e-10           # below this the error is floating-point round-off
        for name, color in (('Euler', ORANGE), ('RK4', BLUE)):
            e = np.array(out[name]['errs'])
            keep = e > floor    # drop round-off points (a log axis cannot show 0)
            ax1.loglog(np.array(dts)[keep], e[keep], '-o', color=color, lw=2, ms=8,
                       label=name)
            ax2.loglog(np.array(out[name]['evals'])[keep], e[keep], '-o', color=color,
                       lw=2, ms=8, label=name)
        dref = np.array([dts[0], dts[-1]])
        ax1.loglog(dref, out['Euler']['errs'][0] * (dref / dts[0]), '--', color=INK2,
                   lw=1, label='reference slopes 1 and 4')
        dref4 = np.array([dts[0], dts[4]])
        ax1.loglog(dref4, 0.5 * out['RK4']['errs'][0] * (dref4 / dts[0])**4, '--',
                   color=INK2, lw=1)
        ax1.axhline(floor, ls=':', color=INK2, lw=1)
        ax1.text(dts[0], floor * 1.8, 'round-off floor (about 1e-10 m)', fontsize=8,
                 color=INK2, ha='right')
        style_axes(ax1, 'Apogee error vs step size (analytical benchmark)',
                   'step size dt (s)', 'absolute apogee error (m)')
        style_axes(ax2, 'Apogee error vs cost', 'right-hand-side evaluations', None)
        ax1.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='center left')
        ax2.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper right')
        fig.tight_layout()
        fig.savefig(figdir / 'fig1_verification.png', dpi=160, facecolor=SURFACE)
        plt.close(fig)

    out['dts'] = dts
    out['exact'] = exact
    # RK4 is only fitted over the first four steps: below dt ~ 0.05 s the
    # error reaches the round-off floor and the slope flattens.
    out['orders'] = {'Euler': observed_orders(dts[:6], out['Euler']['errs'][:6]),
                     'RK4': observed_orders(dts[:4], out['RK4']['errs'][:4])}
    return out


# ---------------------------------------------------------------------------
# 2-4. Full model: calibration, convergence, validation
# ---------------------------------------------------------------------------
def run_calibration(model_lin):
    """Calibrate mass with bisection and Brent; compare cost."""
    res = {}
    for method in ('bisect', 'brent'):
        (mass, n), t = timed(lambda m=method: calibrate_mass(
            APOGEE_NO_BRAKES, model_lin, isa, V_BURNOUT, H_BURNOUT,
            method=m, xtol=1e-4), repeats=1)
        res[method] = dict(mass=mass, n_sim=n, time=t)
    return res


def run_full_convergence(model_lin, mass):
    """RK4 vs Euler against a tight DOP853 reference on the full model."""
    _, h_ref = reference_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT)
    dts = [0.8 * 0.5**n for n in range(7)]
    out = {'h_ref': h_ref, 'dts': dts}
    for name, step in (('Euler', euler_step), ('RK4', rk4_step)):
        errs, evals, times = [], [], []
        for dt in dts:
            counter = [0]
            (_, h), t = timed(lambda: simulate_apogee(
                mass, model_lin, isa, V_BURNOUT, H_BURNOUT, dt=dt, step=step,
                counter=counter), repeats=1)
            errs.append(abs(h - h_ref))
            evals.append(counter[0])
            times.append(t)
        out[name] = dict(errs=errs, evals=evals, times=times)
    out['orders'] = {n: observed_orders(dts[:5], out[n]['errs'][:5]) for n in ('Euler', 'RK4')}
    return out


def run_smoothness_study(model_lin, model_cub, mass):
    """RK4 error vs dt for three C_D models of different smoothness.

    A piecewise-linear table has slope jumps at every Mach node and a cubic
    table is clamped at Mach 0.1; RK4's fourth-order accuracy assumes a
    smooth right-hand side. A smooth analytic C_D is the control case.
    """
    smooth = lambda m: 0.4 + 0.05 * np.sin(3.0 * m)          # noqa: E731
    models = {'linear table': model_lin, 'cubic table': model_cub,
              'smooth C_D (control)': smooth}
    dts = [0.8 * 0.5**n for n in range(7)]
    out = {'dts': dts}
    for name, model in models.items():
        h_ref = reference_apogee(mass, model, isa, V_BURNOUT, H_BURNOUT,
                                 rtol=1e-13, atol=1e-11)[1]
        errs = [abs(simulate_apogee(mass, model, isa, V_BURNOUT, H_BURNOUT,
                                    dt=dt, step=rk4_step)[1] - h_ref)
                for dt in dts]
        out[name] = errs
    return out


def run_validation(model_lin, mass):
    """Apogee vs flap, equivalent flap and deployment delay for the airbrake target."""
    flaps = [0, 25, 50, 75, 100]
    sweep = [simulate_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT, flap=f)[1]
             for f in flaps]
    flap_eq, _ = flap_for_apogee(APOGEE_WITH_BRAKES, mass, model_lin, isa,
                                 V_BURNOUT, H_BURNOUT)

    def apogee_with_delay(td):
        return reference_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT,
                                flap=lambda t: 100.0 if t >= td else 0.0,
                                rtol=1e-9, atol=1e-8)[1]

    t_nb, _ = reference_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT)
    td, _ = bisect(lambda td: apogee_with_delay(td) - APOGEE_WITH_BRAKES,
                   0.0, t_nb, xtol=0.01)
    return dict(flaps=flaps, sweep=sweep, flap_eq=flap_eq, delay=td,
                t_apogee_no_brakes=t_nb)


def run_sensitivity(model_lin, mass, base):
    """One-at-a-time sensitivity of the (no-airbrake) apogee."""
    def scaled_atm(s):
        return lambda h: (s * isa(h)[0], isa(h)[1])

    def apogee(**kw):
        p = dict(mass=mass, v0=V_BURNOUT, cd=model_lin, atm=isa, area=S_REF)
        p.update(kw)
        return simulate_apogee(p['mass'], p['cd'], p['atm'], p['v0'], H_BURNOUT,
                               area=p['area'])[1]

    def cd_scaled(s):
        return DragModel(model_lin.mach, model_lin.flap, model_lin.cd * s)

    cases = [
        ('mass +/-10 %', dict(mass=mass * 1.10), dict(mass=mass * 0.90)),
        ('burnout speed +/-5 m/s', dict(v0=V_BURNOUT + 5), dict(v0=V_BURNOUT - 5)),
        ('C_D +/-5 %', dict(cd=cd_scaled(1.05)), dict(cd=cd_scaled(0.95))),
        ('reference area +/-2 %', dict(area=S_REF * 1.02), dict(area=S_REF * 0.98)),
        ('air density +/-5 %', dict(atm=scaled_atm(1.05)), dict(atm=scaled_atm(0.95))),
    ]
    rows = [(label, apogee(**up) - base, apogee(**dn) - base) for label, up, dn in cases]
    return rows


def run_assumptions(model_lin, model_cub, mass, base):
    """How much do modelling choices move the apogee?"""
    out = {}
    out['cubic_vs_linear'] = [
        (f, simulate_apogee(mass, model_cub, isa, V_BURNOUT, H_BURNOUT, flap=f)[1]
         - simulate_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT, flap=f)[1])
        for f in (0.0, 50.0, 100.0)]

    # Constant-property (analytical) approximations of the same flight.
    a0 = isa(H_BURNOUT)[1]
    cd_burnout = model_lin(V_BURNOUT / a0, 0.0)
    cd_mid = model_lin(0.5, 0.0)
    rho_burnout = isa(H_BURNOUT)[0]
    rho_mid = isa(0.5 * (H_BURNOUT + base))[0]
    variants = []
    for label, rho, cd in (('rho at burnout, C_D at burnout Mach', rho_burnout, cd_burnout),
                           ('rho at mid-coast, C_D at Mach 0.5', rho_mid, cd_mid),
                           ('sea-level rho, C_D at Mach 0.5', 1.225, cd_mid)):
        k = rho * cd * S_REF / (2 * mass)
        variants.append((label, H_BURNOUT + analytic_apogee_height(V_BURNOUT, k, G)))
    out['constant'] = variants
    out['vacuum'] = H_BURNOUT + V_BURNOUT**2 / (2 * G)

    # Interpretation of the 3450 m target: altitude vs height gained.
    m_gain, _ = calibrate_mass(H_BURNOUT + APOGEE_NO_BRAKES, model_lin, isa,
                               V_BURNOUT, H_BURNOUT, bracket=(5.0, 2000.0))
    out['mass_if_height_gain'] = m_gain
    for dtarget in (-50.0, 50.0):
        m, _ = calibrate_mass(APOGEE_NO_BRAKES + dtarget, model_lin, isa,
                              V_BURNOUT, H_BURNOUT)
        out[f'mass_target_{dtarget:+.0f}'] = m
    return out


def run_performance(model_lin, mass):
    """Baseline vs optimised right-hand side, fixed-step vs adaptive."""
    # Baseline: flap passed as a function of time, which forces the full
    # C_D(Mach, flap) lookup (all table columns) on every evaluation.
    counter_b, counter_o = [0], [0]
    (_, t_b) = timed(lambda: simulate_apogee(
        mass, model_lin, isa, V_BURNOUT, H_BURNOUT, flap=lambda t: 0.0,
        counter=counter_b))
    (_, t_o) = timed(lambda: simulate_apogee(
        mass, model_lin, isa, V_BURNOUT, H_BURNOUT, flap=0.0, counter=counter_o))
    n_evals = counter_o[0] // 3                      # evaluations per run

    _, t_ref = timed(lambda: reference_apogee(
        mass, model_lin, isa, V_BURNOUT, H_BURNOUT, rtol=1e-8, atol=1e-6))

    prof = cProfile.Profile()
    prof.enable()
    simulate_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT,
                    flap=lambda t: 0.0)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats('cumulative').print_stats(8)
    return dict(t_baseline=t_b, t_optimised=t_o, n_evals=n_evals,
                t_adaptive=t_ref, profile=buf.getvalue())


# ---------------------------------------------------------------------------
# Figures that need the calibrated mass
# ---------------------------------------------------------------------------
def make_model_figures(plt, figdir, model_lin, model_cub, mass, val):
    # C_D table and interpolants
    fig, (ax,) = new_figure(plt, 1, 7.0, 4.4)
    ms = np.linspace(model_lin.mach[0], model_lin.mach[-1], 200)
    for j, (color, f) in enumerate(zip((BLUE, ORANGE, AQUA), model_lin.flap)):
        ax.plot(model_lin.mach, model_lin.cd[:, j], 'o', color=color, ms=8)
        ax.plot(ms, [model_cub(m, f) for m in ms], '-', color=color, lw=2,
                label=f'flap {f:.0f} %')
        ax.annotate(f'{f:.0f} %', (model_lin.mach[-1], model_lin.cd[-1, j]),
                    xytext=(6, 0), textcoords='offset points', color=INK, fontsize=9,
                    va='center')
    style_axes(ax, 'Drag coefficient from CFD (markers) and cubic interpolant',
               'Mach number', 'C_D (zero pitch)')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper left')
    ax.set_xlim(0.05, 1.0)
    fig.tight_layout()
    fig.savefig(figdir / 'fig2_cd_table.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # Trajectories
    fig, (ax1, ax2) = new_figure(plt, 2, 10.5, 4.4)
    for f, color in ((0.0, BLUE), (50.0, ORANGE), (100.0, AQUA)):
        t, h, v = trajectory(mass, model_lin, isa, V_BURNOUT, H_BURNOUT, flap=f)
        ax1.plot(t, h, color=color, lw=2, label=f'flap {f:.0f} %')
        ax1.plot(t[-1], h[-1], 'o', color=color, ms=8)
        ax2.plot(t, v, color=color, lw=2, label=f'flap {f:.0f} %')
    ax1.axhline(APOGEE_WITH_BRAKES, color=INK2, lw=1, ls='--')
    ax1.text(0.3, APOGEE_WITH_BRAKES + 25, 'airbrake apogee (11000 ft)', color=INK2, fontsize=9)
    style_axes(ax1, 'Altitude after burnout', 'time after burnout (s)', 'altitude (m)')
    style_axes(ax2, 'Vertical velocity', 'time after burnout (s)', 'velocity (m/s)')
    for ax in (ax1, ax2):
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(figdir / 'fig3_trajectories.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # Apogee vs flap
    fig, (ax,) = new_figure(plt, 1, 7.0, 4.4)
    fl = np.linspace(0, 100, 21)
    ap = [simulate_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT, flap=f)[1] for f in fl]
    ax.plot(fl, ap, '-', color=BLUE, lw=2)
    ax.axhline(APOGEE_WITH_BRAKES, color=INK2, lw=1, ls='--')
    ax.plot(val['flap_eq'], APOGEE_WITH_BRAKES, 'o', color=ORANGE, ms=9)
    ax.annotate(f"equivalent constant flap: {val['flap_eq']:.0f} %",
                (val['flap_eq'], APOGEE_WITH_BRAKES), xytext=(10, 14),
                textcoords='offset points', color=INK, fontsize=9)
    ax.text(1, APOGEE_WITH_BRAKES - 28, 'measured airbrake apogee, 11000 ft', color=INK2, fontsize=9)
    style_axes(ax, 'Predicted apogee vs constant flap extension',
               'flap extension (%)', 'apogee altitude (m)')
    fig.tight_layout()
    fig.savefig(figdir / 'fig4_apogee_vs_flap.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_convergence_figure(plt, figdir, conv):
    fig, (ax1, ax2) = new_figure(plt, 2, 10.5, 4.4)
    for name, color in (('Euler', ORANGE), ('RK4', BLUE)):
        ax1.loglog(conv['dts'], conv[name]['errs'], '-o', color=color, lw=2, ms=8, label=name)
        ax2.loglog(conv[name]['evals'], conv[name]['errs'], '-o', color=color, lw=2, ms=8,
                   label=name)
    style_axes(ax1, 'Full model: error vs step size', 'step size dt (s)',
               'apogee error vs DOP853 reference (m)')
    ticks = conv['dts'][::2]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f'{d:g}' for d in ticks])
    style_axes(ax2, 'Full model: error vs cost', 'right-hand-side evaluations', None)
    for ax in (ax1, ax2):
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(figdir / 'fig5_full_model_convergence.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_sensitivity_figure(plt, figdir, rows):
    fig, (ax,) = new_figure(plt, 1, 7.0, 4.0)
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[1] for r in rows], height=0.36, color=BLUE, label='parameter increased')
    ax.barh(y + 0.2, [r[2] for r in rows], height=0.36, color=ORANGE, label='parameter decreased')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1)
    style_axes(ax, 'Change in apogee from one-at-a-time perturbations',
               'apogee change (m)')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / 'fig6_sensitivity.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run_pitch_study(data_path, mass0):
    """Effect of a constant angle of attack on apogee, mass and airbrake authority.

    Uses every pitch angle whose (Mach, flap) table is complete (the coast
    spans Mach 0 to 0.83, so a table that stops at lower Mach is unusable).
    For each angle: the apogee at the zero-pitch calibrated mass, the mass that
    would be calibrated instead, and the airbrake authority (apogee with the
    flaps at 0 % minus at 100 %) at that mass.
    """
    mach_b = V_BURNOUT / isa(H_BURNOUT)[1]
    all_pitches = cd_vs_pitch(data_path, 0.1, 0.0)[0]
    pitches = complete_pitches(data_path)

    def apogee(mass, model, flap=0.0):
        return simulate_apogee(mass, model, isa, V_BURNOUT, H_BURNOUT,
                               flap=flap)[1]

    rows = []
    for p in pitches:
        model = DragModel(*load_cd_table(data_path, p, component='wind'), 'linear')
        a_mean = apogee(mass0, model)
        if p > 0:
            a_rolls = [apogee(mass0, DragModel(*load_cd_table(
                data_path, p, roll=r, component='wind'), 'linear'))
                for r in (0, 45)]
        else:
            a_rolls = [a_mean, a_mean]
        mass_p, _ = calibrate_mass(APOGEE_NO_BRAKES, model, isa, V_BURNOUT,
                                   H_BURNOUT, method='brent', xtol=1e-4)
        authority = apogee(mass_p, model, 0.0) - apogee(mass_p, model, 100.0)
        try:
            flap_eq = flap_for_apogee(APOGEE_WITH_BRAKES, mass_p, model, isa,
                                      V_BURNOUT, H_BURNOUT)[0]
        except ValueError:
            flap_eq = None      # the 11000 ft target is beyond the flaps' reach
        rows.append(dict(pitch=p, cd_burnout=model(mach_b, 0.0), apogee=a_mean,
                         apogee_rolls=a_rolls, mass=mass_p, authority=authority,
                         flap_eq=flap_eq, roll_spread=(cd_roll_spread(data_path, p, 'wind')
                                                       if p > 0 else 0.0)))
    return dict(rows=rows, all_pitches=all_pitches, mach_burnout=mach_b)


def make_pitch_figure(plt, figdir, data_path, pitch):
    fig, (ax1, ax2) = new_figure(plt, 2, 10.0, 4.4)
    series = [(0.1, 0.0, BLUE, '-'), (0.1, 100.0, ORANGE, '-'),
              (0.2, 0.0, BLUE, '--'), (0.2, 100.0, ORANGE, '--')]
    for mach, flap, colour, ls in series:
        x, y = cd_vs_pitch(data_path, mach, flap, component='wind')
        ax1.plot(x, y, color=colour, ls=ls, marker='o', ms=3.5, lw=1.6,
                 label=f'Mach {mach}, flap {flap:.0f} %')
    hi = max(r['pitch'] for r in pitch['rows'])
    ax1.axvspan(-0.3, hi + 0.3, color=GRID, alpha=0.8, lw=0)
    ax1.annotate('complete tables\n(pitch 0-2 deg)', (hi + 0.6, 1.9), fontsize=8,
                 color=INK2, va='top')
    style_axes(ax1, 'Drag along the flight path vs pitch',
               'pitch angle (deg)', 'C_D along flight direction')
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper center',
               bbox_to_anchor=(0.5, -0.16), ncol=2)

    rows = pitch['rows']
    x = np.arange(len(rows))
    base = rows[0]['apogee']
    ax2.bar(x, [r['apogee'] - base for r in rows], width=0.5, color=BLUE,
            label='mean of both roll angles')
    for k, (marker, lab) in enumerate((('v', 'roll 0 deg'), ('^', 'roll 45 deg'))):
        ax2.scatter(x, [r['apogee_rolls'][k] - base for r in rows], marker=marker,
                    color=ORANGE, zorder=3, s=36, label=lab)
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['pitch']:.0f}" for r in rows])
    style_axes(ax2, 'Apogee change at the zero-pitch calibrated mass',
               'pitch angle (deg)', 'apogee change (m)')
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper center',
               bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(figdir / 'fig7_pitch_sensitivity.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run_regression_study(data_path, extra_path, model_lin, model_cub, mass_table):
    """Fit C_D(Mach, flap) by regression to every sample and compare with the table.

    The samples are the gridded CFD runs plus the scattered (noisy) runs.
    Returns None if there are no scattered runs to add.
    """
    paths = [data_path] if extra_path is None else [data_path, extra_path]
    pts = load_cd_points(paths, pitch=0)
    grid_label = (GRID_SOURCE if (pts.source == GRID_SOURCE).any()
                  else Path(data_path).stem)
    noisy = (pts.source != grid_label).to_numpy()
    if not noisy.any():
        return None
    M, F, Y = (pts[c].to_numpy() for c in ('mach', 'flap', 'cd'))

    degrees, cv_table = select_degrees(M, F, Y, test_mask=noisy)
    reg = RegressionDragModel.fit(M, F, Y, *degrees)
    cv = cross_validate(M, F, Y, *degrees, test_mask=noisy)

    def rel_error(model):
        pred = np.array([model(m, f) for m, f in zip(M[noisy], F[noisy])])
        e = pred / Y[noisy] - 1.0
        return float(np.sqrt(np.mean(e**2))), float(e.mean())

    methods = [('table, linear', 'gridded runs', *rel_error(model_lin)),
               ('table, cubic', 'gridded runs', *rel_error(model_cub)),
               ('regression, cross-validated', 'gridded + other scattered runs',
                cv['rmse'], cv['bias'])]

    # Independent check: fit on the scattered runs alone, predict the gridded runs.
    only_noisy = RegressionDragModel.fit(M[noisy], F[noisy], Y[noisy], *degrees)
    e = only_noisy.predict(M[~noisy], F[~noisy]) / Y[~noisy] - 1.0
    independent = dict(rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()),
                       n=int((~noisy).sum()))
    offset = source_offset(M, F, Y, noisy, *degrees)

    mass_reg, _ = calibrate_mass(APOGEE_NO_BRAKES, reg, isa, V_BURNOUT, H_BURNOUT,
                                 method='brent', xtol=1e-4)
    flaps = [0, 25, 50, 75, 100]
    sweep = [simulate_apogee(mass_reg, reg, isa, V_BURNOUT, H_BURNOUT, flap=f)[1]
             for f in flaps]
    flap_eq = flap_for_apogee(APOGEE_WITH_BRAKES, mass_reg, reg, isa, V_BURNOUT,
                              H_BURNOUT)[0]
    at_table_mass = simulate_apogee(mass_table, reg, isa, V_BURNOUT, H_BURNOUT)[1]

    # Pitch 2 deg with the same degrees (both roll angles pooled).
    pts2 = load_cd_points(paths, pitch=2)
    reg2 = RegressionDragModel.fit(pts2['mach'], pts2['flap'], pts2['cd'], *degrees)
    drop2 = (simulate_apogee(mass_reg, reg2, isa, V_BURNOUT, H_BURNOUT)[1]
             - simulate_apogee(mass_reg, reg, isa, V_BURNOUT, H_BURNOUT)[1])
    counts = {p: (int((pp.source == grid_label).sum()),
                  int((pp.source != grid_label).sum()))
              for p, pp in ((0, pts), (2, pts2))}
    return dict(pts=pts, noisy=noisy, degrees=degrees, cv_table=cv_table,
                reg=reg, methods=methods, independent=independent, offset=offset,
                mass=mass_reg, mass_table=mass_table, at_table_mass=at_table_mass,
                flaps=flaps, sweep=sweep, flap_eq=flap_eq, drop2=drop2,
                counts=counts, grid_label=grid_label)


def make_regression_figure(plt, figdir, reg_study, model_lin):
    reg, pts, noisy = reg_study['reg'], reg_study['pts'], reg_study['noisy']
    fig, (ax1, ax2) = new_figure(plt, 2, 10.0, 4.6)
    bands = [(0.0, BLUE, 'flap 0 %', lambda f: f <= 5),
             (50.0, AQUA, 'flap 50 %', lambda f: np.abs(f - 50) <= 5),
             (100.0, ORANGE, 'flap 100 %', lambda f: f >= 95)]
    mm = np.linspace(*reg.mach_range, 200)
    for flap, colour, label, band in bands:
        sel = noisy & band(pts['flap'].to_numpy())
        ax1.scatter(pts['mach'][sel], pts['cd'][sel], s=9, color=colour,
                    alpha=0.35, linewidths=0)
        grid = ~noisy & (pts['flap'].to_numpy() == flap)
        ax1.scatter(pts['mach'][grid], pts['cd'][grid], s=34, color=colour,
                    edgecolor=INK, linewidths=0.8, zorder=4)
        ax1.plot(mm, reg.predict(mm, flap), color=colour, lw=2, label=label)
        ax1.plot(mm, [model_lin(m, flap) for m in mm], color=colour, lw=1,
                 ls='--')
    style_axes(ax1, 'Regression surface through all samples', 'Mach number',
               'C_D along flight direction')
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper center',
               bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax1.annotate('dots: scattered runs  |  outlined: gridded runs  |  solid: regression  |  dashed: linear table',
                 (0.5, -0.30), xycoords='axes fraction', ha='center', fontsize=7.5,
                 color=INK2)

    names = ['table,\nlinear', 'table,\ncubic', 'regression\n(cross-validated)']
    vals = [100 * m[2] for m in reg_study['methods']]
    ax2.bar(names, vals, width=0.5, color=[ORANGE, ORANGE, BLUE])
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.01, f'{v:.2f} %', ha='center', fontsize=9, color=INK)
    style_axes(ax2, 'Error on held-out scattered runs', None, 'RMS error in C_D (%)')
    ax2.tick_params(axis='x', labelsize=8.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    fig.savefig(figdir / 'fig8_regression_drag.png', dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', default='data/Project Data Set.csv',
                        help='CFD csv file (default: data/Project Data Set.csv)')
    parser.add_argument('--extra-data', default=None,
                        help='csv of scattered (noisy) CFD runs to add to the regression '
                             'fit (not needed if --data has a Source column)')
    parser.add_argument('--outdir', default='docs', help='output folder')
    parser.add_argument('--no-plots', action='store_true', help='skip figures')
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}\n"
              "Place the CFD csv there or pass --data PATH.", file=sys.stderr)
        return 1

    extra_path = Path(args.extra_data) if args.extra_data else None
    if extra_path is not None and not extra_path.is_file():
        print(f"error: extra data file not found: {extra_path}", file=sys.stderr)
        return 1

    plt = None
    figdir = Path(args.outdir) / 'figures'
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figdir.mkdir(parents=True, exist_ok=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    mach, flap, cd = load_cd_table(data_path)
    model_lin = DragModel(mach, flap, cd, 'linear')
    model_cub = DragModel(mach, flap, cd, 'cubic')
    a0 = isa(H_BURNOUT)[1]
    print(f"Burnout Mach number: {V_BURNOUT / a0:.3f}  (table covers "
          f"{mach[0]:.1f}-{mach[-1]:.1f})")

    print('1/8 verification against the analytical solution ...')
    ver = run_verification(plt, figdir)

    print('2/8 mass calibration ...')
    cal = run_calibration(model_lin)
    mass = cal['bisect']['mass']
    check_ref = reference_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT)[1]

    print('3/8 full-model convergence ...')
    conv = run_full_convergence(model_lin, mass)

    smooth = run_smoothness_study(model_lin, model_cub, mass)

    print('4/8 airbrake validation ...')
    val = run_validation(model_lin, mass)

    print('5/8 sensitivity and assumptions ...')
    sens = run_sensitivity(model_lin, mass, APOGEE_NO_BRAKES)
    assum = run_assumptions(model_lin, model_cub, mass, APOGEE_NO_BRAKES)

    print('6/8 performance ...')
    perf = run_performance(model_lin, mass)

    print('7/8 angle-of-attack sensitivity ...')
    pitch = run_pitch_study(data_path, mass)

    print('8/8 regression drag model ...')
    regs = run_regression_study(data_path, extra_path, model_lin, model_cub, mass)
    if regs is None:
        print('    skipped: no scattered CFD runs (use --extra-data or a merged csv)')

    if plt:
        make_model_figures(plt, figdir, model_lin, model_cub, mass, val)
        make_convergence_figure(plt, figdir, conv)
        make_sensitivity_figure(plt, figdir, sens)
        make_pitch_figure(plt, figdir, data_path, pitch)
        if regs:
            make_regression_figure(plt, figdir, regs, model_lin)

    # ---- results summary --------------------------------------------------
    L = []
    L.append('# Results summary (generated by `src/main.py`)\n')
    L.append(f'Burnout: v = {V_BURNOUT} m/s at {H_BURNOUT:.1f} m '
             f'(Mach {V_BURNOUT / a0:.3f}); reference area S = {S_REF:.5f} m^2.\n')
    if ver:
        L.append('## 1. Verification against the analytical solution\n')
        L.append(f"Exact apogee gain (constant rho, C_D; k = 8.8e-5): {ver['exact']:.6f} m\n")
        rows = [(f'{dt:.4f}', f"{ver['Euler']['errs'][i]:.3e}", f"{ver['RK4']['errs'][i]:.3e}")
                for i, dt in enumerate(ver['dts'])]
        L.append(md_table(['dt (s)', 'Euler error (m)', 'RK4 error (m)'], rows))
        L.append(f"\nObserved order (Euler over 6 steps, RK4 over the first 4): Euler "
                 f"{np.mean(ver['orders']['Euler']):.2f}, RK4 {np.mean(ver['orders']['RK4']):.2f}\n")
    L.append('## 2. Mass calibration (apogee 3450 m, no airbrakes)\n')
    L.append(md_table(['method', 'mass (kg)', 'simulations', 'time (s)'],
                      [(m, f"{r['mass']:.4f}", r['n_sim'], f"{r['time']:.2f}")
                       for m, r in cal.items()]))
    L.append(f"\nCheck with DOP853 reference at the calibrated mass: apogee "
             f"{check_ref:.3f} m (target {APOGEE_NO_BRAKES:.1f} m)\n")
    L.append('## 3. Full-model convergence (reference: DOP853, '
             f"apogee {conv['h_ref']:.4f} m)\n")
    rows = [(f'{dt:.4f}', f"{conv['Euler']['errs'][i]:.3e}", f"{conv['RK4']['errs'][i]:.3e}",
             conv['Euler']['evals'][i], conv['RK4']['evals'][i])
            for i, dt in enumerate(conv['dts'])]
    L.append(md_table(['dt (s)', 'Euler error (m)', 'RK4 error (m)', 'Euler evals', 'RK4 evals'], rows))
    L.append(f"\nObserved order (first 5 steps): Euler {np.mean(conv['orders']['Euler']):.2f}, "
             f"RK4 {np.mean(conv['orders']['RK4']):.2f}\n")
    L.append('### RK4 error vs smoothness of C_D(Mach) (reference: DOP853, rtol 1e-13)\n')
    names = ['linear table', 'cubic table', 'smooth C_D (control)']
    L.append(md_table(['dt (s)'] + [n + ' error (m)' for n in names],
                      [[f'{dt:.4f}'] + [f'{smooth[n][i]:.2e}' for n in names]
                       for i, dt in enumerate(smooth['dts'])]))
    L.append('\nRK4 is fourth order only for the smooth C_D; slope jumps in the '
             'interpolated table cap the observed order and set an error floor, '
             'far below the modelling uncertainty (metres).\n')
    L.append('## 4. Airbrake validation\n')
    L.append(md_table(['flap (%)', 'predicted apogee (m)'],
                      [(f, f'{a:.1f}') for f, a in zip(val['flaps'], val['sweep'])]))
    L.append(f"\nMeasured apogee with airbrakes: {APOGEE_WITH_BRAKES:.1f} m (11000 ft). "
             f"Equivalent constant flap from burnout: {val['flap_eq']:.1f} %. "
             f"Alternatively, 100 % flap deployed {val['delay']:.2f} s after burnout "
             f"(no-brake apogee reached at {val['t_apogee_no_brakes']:.1f} s).\n")
    L.append('## 5. Sensitivity of the no-airbrake apogee (m)\n')
    L.append(md_table(['perturbation', 'increase', 'decrease'],
                      [(r[0], f'{r[1]:+.1f}', f'{r[2]:+.1f}') for r in sens]))
    L.append('\n## 6. Modelling assumptions\n')
    L.append(md_table(['flap (%)', 'cubic - linear C_D interpolation (m)'],
                      [(f'{f:.0f}', f'{d:+.2f}') for f, d in assum['cubic_vs_linear']]))
    L.append('\n' + md_table(['constant-property analytical estimate', 'apogee (m)', 'error vs full model (m)'],
                             [(lab, f'{h:.0f}', f'{h - APOGEE_NO_BRAKES:+.0f}')
                              for lab, h in assum['constant']]))
    L.append(f"\nDrag-free apogee: {assum['vacuum']:.0f} m. "
             f"Mass if 3450 m were height gained instead of altitude: {assum['mass_if_height_gain']:.1f} kg. "
             f"Mass for target -50 m / +50 m: {assum['mass_target_-50']:.2f} / {assum['mass_target_+50']:.2f} kg.\n")
    L.append('## 7. Performance\n')
    L.append(md_table(['configuration', 'time per apogee run (ms)'],
                      [('baseline RK4 (full C_D lookup each evaluation)', f"{1e3 * perf['t_baseline']:.1f}"),
                       ('optimised RK4 (flap slice precomputed)', f"{1e3 * perf['t_optimised']:.1f}"),
                       ('adaptive DOP853 (rtol 1e-8)', f"{1e3 * perf['t_adaptive']:.1f}")]))
    L.append(f"\nSpeed-up from optimisation: {perf['t_baseline'] / perf['t_optimised']:.1f}x. "
             f"Right-hand-side evaluations per run: {perf['n_evals']}.\n")
    L.append('Profile of the baseline run:\n\n```\n' + perf['profile'] + '```\n')
    L.append('## 8. Angle-of-attack sensitivity (constant pitch from burnout)\n')
    rows8 = []
    for r in pitch['rows']:
        flap_txt = ('target out of reach' if r['flap_eq'] is None
                    else f"{r['flap_eq']:.1f}")
        rows8.append((f"{r['pitch']:.0f}", f"{r['cd_burnout']:.3f}",
                      f"{r['apogee']:.1f}",
                      f"{r['apogee'] - pitch['rows'][0]['apogee']:+.1f}",
                      f"{min(r['apogee_rolls']):.0f} to {max(r['apogee_rolls']):.0f}",
                      f"{r['mass']:.2f}", f"{r['authority']:.0f}", flap_txt))
    L.append(md_table(['pitch (deg)', f"C_D at Mach {pitch['mach_burnout']:.2f}, flap 0",
                       'apogee at 0-deg mass (m)', 'change (m)',
                       'range over roll (m)', 'recalibrated mass (kg)',
                       'airbrake authority (m)', 'flap for 11000 ft (%)'], rows8))
    usable = ', '.join(f"{r['pitch']:.0f}" for r in pitch['rows'])
    L.append(f"\nOnly pitch angles {usable} deg have a complete (Mach, flap) table. "
             f"The file also contains {', '.join(f'{p:.0f}' for p in pitch['all_pitches'] if p not in [r['pitch'] for r in pitch['rows']])} deg, "
             "but only at Mach 0.1-0.4 or lower, so they cannot be used for a coast "
             "that starts at Mach 0.83. Airbrake authority is the apogee with the "
             "flaps at 0 % minus at 100 %, using the mass recalibrated at that pitch. "
             "Drag is taken along the flight direction, C_axial cos(pitch) + "
             "C_normal sin(pitch); the csv column 'Total C_D' is the whole force "
             "magnitude and would overstate the drag at non-zero pitch. The 0-deg "
             "row uses the axial coefficient alone, which differs from 'Total C_D' "
             "by under 0.1 %, hence 3450.2 m and not 3450.0 m.\n")
    if regs:
        L.append('\n## 9. Regression drag model from all CFD samples\n')
        n_g0, n_s0 = regs['counts'][0]
        n_g2, n_s2 = regs['counts'][2]
        L.append(md_table(['pitch (deg)', 'gridded samples', 'scattered samples', 'total'],
                          [(0, n_g0, n_s0, n_g0 + n_s0), (2, n_g2, n_s2, n_g2 + n_s2)]))
        dm, df_ = regs['degrees']
        L.append(f"\nSamples at the same Mach number and flap (the two roll angles of a gridded run) are averaged, "
                 f"so the gridded runs count {n_g0} values at pitch 0, not the {2 * n_g0} rows in the file. "
                 f"Polynomial degrees chosen by 5-fold cross-validation, repeated 3 times, with the "
                 f"one-standard-error rule: degree {dm} in Mach and {df_} in flap "
                 f"({regs['reg'].n_params} coefficients).\n")
        best = sorted(regs['cv_table'], key=lambda t: t['rmse'])[:5]
        pick = [t for t in regs['cv_table'] if (t['deg_mach'], t['deg_flap']) == regs['degrees']][0]
        shown = best + ([pick] if pick not in best else [])
        L.append(md_table(['deg Mach', 'deg flap', 'coefficients', 'CV RMS error (%)', 'CV bias (%)', 'selected'],
                          [(t['deg_mach'], t['deg_flap'], t['n_params'], f"{100 * t['rmse']:.3f}",
                            f"{100 * t['bias']:+.3f}", 'yes' if t is pick else '') for t in shown]))
        L.append('\n### Error on the scattered runs\n')
        L.append(md_table(['drag model', 'fitted to', 'RMS error (%)', 'bias (%)'],
                          [(a, b, f'{100 * c:.3f}', f'{100 * d:+.3f}') for a, b, c, d in regs['methods']]))
        ind = regs['independent']
        L.append(f"\nIndependent check: a regression fitted to the scattered runs alone predicts the "
                 f"{ind['n']} gridded values with RMS error {100 * ind['rmse']:.2f} % (bias {100 * ind['bias']:+.2f} %). "
                 f"Offset of the scattered runs from the gridded runs after accounting for Mach and flap: "
                 f"{100 * regs['offset'][0]:+.2f} % +/- {100 * regs['offset'][1]:.2f} % (one standard error).\n")
        L.append('### Effect on the apogee results\n')
        L.append(md_table(['quantity', 'table (gridded runs)', 'regression (all samples)'],
                          [('calibrated mass for 3450 m (kg)', f"{regs['mass_table']:.3f}", f"{regs['mass']:.3f}"),
                           ('apogee at the table mass (m)', f'{APOGEE_NO_BRAKES:.1f}', f"{regs['at_table_mass']:.1f}"),
                           ('constant flap for 11000 ft (%)', f"{val['flap_eq']:.1f}", f"{regs['flap_eq']:.1f}"),
                           ('apogee with 100 % flap (m)', f"{val['sweep'][-1]:.1f}", f"{regs['sweep'][-1]:.1f}"),
                           ('apogee change at 2 deg pitch (m)', f"{pitch['rows'][-1]['apogee'] - pitch['rows'][0]['apogee']:+.1f}",
                            f"{regs['drop2']:+.1f}")]))
        L.append("\nThe 2-deg row compares both pitches at the same mass: for the table the zero-pitch calibrated "
                 "mass, for the regression the regression-calibrated mass.\n")
    text = '\n'.join(L)
    (Path(args.outdir) / 'results.md').write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
