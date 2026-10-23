"""Entry point: runs the whole apogee-prediction study end to end.

Usage (from the repository root, with the virtual environment active)::

    python src/main.py --data data/project_data_set.csv

Steps
-----
1. Verification: integrators against the analytical constant-k solution.
2. Convergence and cost of Euler and RK4 on the full model (ISA + C_D table).
3. Calibration of the unknown rocket mass to the 3450 m no-airbrake apogee.
4. Validation against the 11000 ft airbrake apogee and flap/deployment study.
5. Sensitivity, modelling-assumption and performance studies.

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
from drag import DragModel, load_cd_table
from integrators import euler_step, rk4_step

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


def md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---'] * len(headers)) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(lines)


def observed_orders(dts, errs):
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1]))
            for i in range(len(errs) - 1)]


# ---------------------------------------------------------------------------
# 1. Verification against the analytical solution
# ---------------------------------------------------------------------------
def run_verification(plt, figdir):
    """Euler/RK4 error against the closed-form apogee (constant rho and C_D)."""
    rho, cd0, k = 1.225, 0.40, 1.0e-4
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
        for name, color in (('Euler', ORANGE), ('RK4', BLUE)):
            ax1.loglog(dts, out[name]['errs'], '-o', color=color, lw=2, ms=8, label=name)
            ax2.loglog(out[name]['evals'], out[name]['errs'], '-o', color=color,
                       lw=2, ms=8, label=name)
        dref = np.array([dts[0], dts[-1]])
        ax1.loglog(dref, out['Euler']['errs'][0] * (dref / dts[0]), '--', color=INK2,
                   lw=1, label='slope 1 and 4 (reference)')
        ax1.loglog(dref, out['RK4']['errs'][2] * (dref / dts[2])**4, '--', color=INK2, lw=1)
        style_axes(ax1, 'Apogee error vs step size (analytical benchmark)',
                   'step size dt (s)', 'absolute apogee error (m)')
        style_axes(ax2, 'Apogee error vs cost', 'right-hand-side evaluations', None)
        for ax in (ax1, ax2):
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
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


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', default='data/Project Data Set.csv',
                        help='CFD csv file (default: data/Project Data Set.csv)')
    parser.add_argument('--outdir', default='docs', help='output folder')
    parser.add_argument('--no-plots', action='store_true', help='skip figures')
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}\n"
              "Place the CFD csv there or pass --data PATH.", file=sys.stderr)
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

    print('1/6 verification against the analytical solution ...')
    ver = run_verification(plt, figdir)

    print('2/6 mass calibration ...')
    cal = run_calibration(model_lin)
    mass = cal['bisect']['mass']
    check_ref = reference_apogee(mass, model_lin, isa, V_BURNOUT, H_BURNOUT)[1]

    print('3/6 full-model convergence ...')
    conv = run_full_convergence(model_lin, mass)

    smooth = run_smoothness_study(model_lin, model_cub, mass)

    print('4/6 airbrake validation ...')
    val = run_validation(model_lin, mass)

    print('5/6 sensitivity and assumptions ...')
    sens = run_sensitivity(model_lin, mass, APOGEE_NO_BRAKES)
    assum = run_assumptions(model_lin, model_cub, mass, APOGEE_NO_BRAKES)

    print('6/6 performance ...')
    perf = run_performance(model_lin, mass)

    if plt:
        make_model_figures(plt, figdir, model_lin, model_cub, mass, val)
        make_convergence_figure(plt, figdir, conv)
        make_sensitivity_figure(plt, figdir, sens)

    # ---- results summary --------------------------------------------------
    L = []
    L.append('# Results summary (generated by `src/main.py`)\n')
    L.append(f'Burnout: v = {V_BURNOUT} m/s at {H_BURNOUT:.1f} m '
             f'(Mach {V_BURNOUT / a0:.3f}); reference area S = {S_REF:.5f} m^2.\n')
    if ver:
        L.append('## 1. Verification against the analytical solution\n')
        L.append(f"Exact apogee gain (constant rho, C_D; k = 1e-4): {ver['exact']:.6f} m\n")
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
    text = '\n'.join(L)
    (Path(args.outdir) / 'results.md').write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
