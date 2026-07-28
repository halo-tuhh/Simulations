#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================
 NITROUS OXIDE ROCKET-TANK FILLING SIMULATION
=====================================================================================
 A lumped control-volume (0-D network) transient simulation of filling a rocket
 oxidiser tank with nitrous oxide (N2O) from a reserve/supply tank through a line
 of pipes and on/off valves.

 Physics / methods
 -----------------
 * Every pipe, valve section and tank is a control volume (CV / "node") with two
   conserved quantities per species and one energy quantity:  m_N2O, m_air, U.
 * Fluid properties for N2O come from CoolProp (Helmholtz EOS, real fluid,
   two-phase capable).  The residual air that initially fills the line is treated
   as an ideal gas occupying the gas/ullage volume (Dalton partial pressures).
 * Two-phase N2O flow through the valve orifices uses the Homogeneous Equilibrium
   Model (HEM): isentropic expansion of the upstream stagnation state to the throat,
   mass flux  G = rho_t * sqrt(2 (h0 - h_t)),  maximised over throat pressure.
   The maximum of G identifies the CHOKED condition (critical flow); the code
   alerts you whenever a valve is choked.
 * Air-dominated nodes (before N2O arrives) use the compressible ideal-gas nozzle
   equation for the orifice, also with choking detection.
 * Valve orifices additionally respect the supplied Kv (flow-coefficient) limit
   (single-phase incompressible sizing); the actual flow is the more restrictive
   of the HEM/gas orifice result and the Kv result.
 * Pipe / valve-section pressure losses use Darcy-Weisbach friction (Haaland f).
 * The ODE system (m_N2O, m_air, U for every node) is advanced with the EXPLICIT
   (forward) EULER method.
 * SI units everywhere.  Temperatures in KELVIN.

 Flow path (as requested):
   reserve tank -> valve1 -> pipe1 -> valve2 -> pipe2 -> valve3 -> pipe3
                -> valve4 -> pipe4 -> rocket tank -> valve5
 Each valve = [small pipe section A] --[orifice]-- [small pipe section B].

 Outputs: time histories of pressure and temperature in every pipe, every valve
 and the rocket tank, plus the rising liquid level in the rocket tank, the position
 of every valve, and a report of any choked-flow events.
=====================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp import AbstractState

# =====================================================================================
#  1.  USER INPUTS  --  edit everything in this block
# =====================================================================================

# ---- Fluids -------------------------------------------------------------------------
FLUID = "NitrousOxide"          # CoolProp name for N2O
P_ATM = 1.0e5                    # [Pa]  initial air pressure in line & rocket tank

# ---- Simulation control -------------------------------------------------------------
# STABILITY / PERFORMANCE NOTE:
#   Explicit Euler on compressible flow in small volumes is only conditionally stable.
#   The limiting volumes are the tiny valve sections, so DT must be small.
#   DT = 1e-4 s is stable for the default geometry.  If you see "NUMERICAL
#   INSTABILITY", NaNs, or wild oscillations, reduce DT (try 5e-5 or 2e-5).
#   Runtime scales with steps AND with how many valves pass two-phase N2O (the HEM
#   flash is the cost).  ~4 ms/step here -> a few minutes for tens of thousands of
#   steps.  Filling to a high level is a multi-second physical process for small
#   orifices; enlarge d_orifice or accept longer runs.
T_TOTAL = 10.0                  # [s]   total simulated time (+ 10.0 s extension when STOP_ON_ROCKET_FILL enabled)
DT      = 1.0e-4               # [s]   Euler time step
RECORD_EVERY = 50             # store/plot every N-th step (keeps memory reasonable)
PRINT_EVERY = 1000    # print terminal time every N-th step (set to 1 for every step)

# ---- Ambient / initial line temperature ---------------------------------------------
T_AMBIENT = 293.15             # [K]   initial temperature of line & rocket-tank air

# ---- Reserve (supply) tank ----------------------------------------------------------
RESERVE = dict(
    V         = 0.010,         # [m^3]  internal volume (50 L)
    T         = T_AMBIENT,        # [K]    temperature of stored N2O
    P         = 5748577.277,   # [Pa]   None -> use saturation pressure at T (self-pressurised)
    liq_fill  = 1,             # [-]    initial liquid volume fraction (rest is N2O vapour)
)

# ---- Rocket tank (vertical concentric annular tank) ------------------------------
ROCKET = dict(
    D_outer = 0.104,          # [m]  outer diameter of the concentric annulus
    D_inner = 0.050,          # [m]  inner diameter of the concentric annulus
    H = 1.000,                # [m]  height
    T = T_AMBIENT,            # [K]  initial air temperature
)

# ---- Main pipes 1..4  (circular) ----------------------------------------------------
#  length [m], inner diameter [m]
PIPES = [
    dict(L=0.10, D=0.010),     # pipe1
    dict(L=5.00, D=0.010),     # pipe2
    dict(L=1.50, D=0.010),     # pipe3
    dict(L=0.10, D=0.010),     # pipe4
]

# ---- Valves 1..5 --------------------------------------------------------------------
#  Each valve is two small pipe sections (sec_L, sec_D) around an orifice (d_orifice).
#  Kv  = valve flow coefficient (metric units, m^3/h at 1 bar) -> used as a single-phase limit.
#  Cd  = orifice discharge coefficient (dimensionless, ~0.6-0.85).
VALVES = [
    dict(Kv=3.00, d_orifice=0.0100, Cd=0.80, sec_L=0.02, sec_D=0.010),  # valve1
    dict(Kv=1.50, d_orifice=0.0100, Cd=0.65, sec_L=0.02, sec_D=0.010),  # valve2
    dict(Kv=2.00, d_orifice=0.0100, Cd=0.80, sec_L=0.02, sec_D=0.010),  # valve3
    dict(Kv=1.60, d_orifice=0.0100, Cd=0.80, sec_L=0.02, sec_D=0.010),  # valve4
    dict(Kv=0.10, d_orifice=0.0010, Cd=0.60, sec_L=0.01, sec_D=0.001),  # valve5
]

# ---- Valve schedules ----------------------------------------------------------------
#  For each valve, a list of (time_seconds, state) events.  state: 1 = fully open, 0 = closed.
#  The valve holds the last state whose time <= current sim time.
#  Example below: all valves open at t = 0.  Edit freely, e.g. staged opening.
#  NOTE: When STOP_ON_ROCKET_FILL is enabled, valves 1-4 close 0.5s after rocket fill target is reached.
VALVE_SCHEDULE = [
    [(0.0, 1)],                # valve1 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 0), (0.5, 1)],     # valve2 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 0), (1.0, 1)],     # valve3 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 0), (1.5, 1)],     # valve4 (closes 0.5s after fill target reached if STOP_ON_ROCKET_FILL enabled)
    [(0.0, 1)],                # valve5
]

# ---- Stopping criteria --------------------------------------------------------------
#  When STOP_ON_ROCKET_FILL is enabled, the simulation will continue until the rocket tank
#  liquid volume fraction reaches ROCKET_FILL_TARGET, then all valves 1-4 will close after
#  0.5s and the simulation will continue for 10s more to observe the pressure decay in the
#  rocket tank.
#  NOTE: When T_TOTAL is reached, the simulation will stop regardless of the rocket fill level.

STOP_ON_ROCKET_FILL = True    # [bool] enable/disable rocket fill stopping condition
ROCKET_FILL_TARGET = 0.15      # [-]    rocket tank liquid volume fraction required to end simulation

# ---- Pipe wall roughness (for friction factor) --------------------------------------
ROUGHNESS = 1.5e-6             # [m]  absolute roughness (drawn tubing ~1.5 um)

# ---- Air (ideal gas) properties -----------------------------------------------------
R_AIR   = 287.05               # [J/(kg K)]
GAMMA_A = 1.400                # [-]
CV_AIR  = R_AIR / (GAMMA_A - 1.0)
CP_AIR  = GAMMA_A * CV_AIR

# ---- Numerical guards ---------------------------------------------------------------
M_FLOOR   = 1.0e-12            # [kg]  minimum species mass kept in a node
MAX_DRAIN = 0.20               # a branch may drain at most this fraction of upstream
                               #        species mass per step (outflow limiter)
HEM_NPTS  = 24                 # throat-pressure samples in the HEM mass-flux sweep
N2O_RICH  = 0.5                # mass fraction above which a node is "N2O dominated"

# =====================================================================================
#  2.  FLUID / SATURATION SET-UP
# =====================================================================================
_AS = AbstractState("HEOS", FLUID)          # reusable CoolProp state object
TCRIT = _AS.T_critical()
TTRIP = _AS.Ttriple()

# Build N2O saturation lookup tables so the inner state solve avoids CoolProp calls.
_T_TAB = np.linspace(TTRIP + 0.5, TCRIT - 0.15, 400)
_PS, _RL, _RV, _UL, _UV, _HL, _HV, _SL, _SV = (np.zeros_like(_T_TAB) for _ in range(9))
for _i, _T in enumerate(_T_TAB):
    _AS.update(CP.QT_INPUTS, 0.0, _T)
    _PS[_i], _RL[_i], _UL[_i], _HL[_i], _SL[_i] = _AS.p(), _AS.rhomass(), _AS.umass(), _AS.hmass(), _AS.smass()
    _AS.update(CP.QT_INPUTS, 1.0, _T)
    _RV[_i], _UV[_i], _HV[_i], _SV[_i] = _AS.rhomass(), _AS.umass(), _AS.hmass(), _AS.smass()


def sat(T):
    """Interpolated N2O saturation properties at temperature T [K]."""
    return (np.interp(T, _T_TAB, _PS), np.interp(T, _T_TAB, _RL), np.interp(T, _T_TAB, _RV),
            np.interp(T, _T_TAB, _UL), np.interp(T, _T_TAB, _UV))


def n2o_Psat(T):
    return float(np.interp(T, _T_TAB, _PS))


# =====================================================================================
#  3.  NETWORK TOPOLOGY  (nodes and branches)
# =====================================================================================
#  Node types: 'reservoir' (tanks), 'pipe', 'vsec' (valve small section)
#  Branch kinds: 'friction' (Darcy-Weisbach) or 'orifice' (valve, HEM/gas + Cv + choke)

nodes   = []     # list of dicts: name,type,V,L,D, (rocket: A_cross)
branches = []    # list of dicts describing each connection


def add_node(name, ntype, V, L=0.0, D=0.0, A_cross=None):
    nodes.append(dict(name=name, type=ntype, V=V, L=L, D=D, A_cross=A_cross))
    return len(nodes) - 1


# --- reserve tank ---
i_reserve = add_node("reserve", "reservoir", RESERVE["V"])

# --- build the alternating chain of valves and pipes ---
prev = i_reserve
for k in range(4):
    v = VALVES[k]
    Vs = np.pi / 4.0 * v["sec_D"] ** 2 * v["sec_L"]
    va = add_node(f"v{k+1}a", "vsec", Vs, L=v["sec_L"], D=v["sec_D"])
    vb = add_node(f"v{k+1}b", "vsec", Vs, L=v["sec_L"], D=v["sec_D"])
    branches.append(dict(kind="friction", i=prev, j=va))
    branches.append(dict(kind="orifice", i=va, j=vb, valve=k))
    p = PIPES[k]
    Vp = np.pi / 4.0 * p["D"] ** 2 * p["L"]
    ip = add_node(f"pipe{k+1}", "pipe", Vp, L=p["L"], D=p["D"])
    branches.append(dict(kind="friction", i=vb, j=ip))
    prev = ip

# --- rocket tank ---
A_cross = np.pi / 4.0 * (ROCKET["D_outer"] ** 2 - ROCKET["D_inner"] ** 2)
V_rocket = A_cross * ROCKET["H"]
i_rocket = add_node("rocket", "reservoir", V_rocket, A_cross=A_cross)
branches.append(dict(kind="friction", i=prev, j=i_rocket))

# --- vent valve ---
v = VALVES[4]
Vs = np.pi / 4.0 * v["sec_D"] ** 2 * v["sec_L"]
v5a = add_node("v5a", "vsec", Vs, L=v["sec_L"], D=v["sec_D"])
v5b = add_node("v5b", "vsec", Vs, L=v["sec_L"], D=v["sec_D"])
branches.append(dict(kind="friction", i=i_rocket, j=v5a))
branches.append(dict(kind="orifice", i=v5a, j=v5b, valve=4))
i_vent = add_node("vent", "reservoir", 1.0e6)
branches.append(dict(kind="friction", i=v5b, j=i_vent))

N = len(nodes)

# Pre-compute friction-branch geometry (symmetric half-length split; tanks add loss only)
for b in branches:
    if b["kind"] != "friction":
        continue
    ni, nj = nodes[b["i"]], nodes[b["j"]]
    # limiting diameter / area
    Ds = [d for d in (ni["D"], nj["D"]) if d > 0]
    D_b = min(Ds) if Ds else 0.01
    b["A"] = np.pi / 4.0 * D_b ** 2
    b["D"] = D_b
    # each conduit contributes half its length; reservoirs contribute none
    L_b = 0.0
    K_minor = 0.0
    for nd in (ni, nj):
        if nd["type"] in ("pipe", "vsec"):
            L_b += 0.5 * nd["L"]
        else:
            K_minor += 0.75          # tank entrance/exit loss
    b["L"] = L_b
    b["K_minor"] = K_minor


# =====================================================================================
#  4.  STATE INVERSION:  (m_N2O, m_air, U, V)  ->  (T, P, ...)
# =====================================================================================

def brentq(f, a, b, tol=1e-6, maxit=80):
    """Minimal, dependency-light Brent root finder (kept local for portability)."""
    fa, fb = f(a), f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa * fb > 0:               # no sign change -> return closest end
        return a if abs(fa) < abs(fb) else b
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxit):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2 * xm * s, 1 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0: q = -q
            p = abs(p)
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    return b


def node_state(mN2O, mair, U, V):
    """
    Return dict with T,P,rho,h,u,V_liq,quality,phase for a node holding mN2O kg of
    N2O and mair kg of air in volume V with total internal energy U.
    References: air ideal gas u=cv*T,h=cp*T (0 K ref); N2O from CoolProp.
    """
    mtot = mN2O + mair
    out = dict(mN2O=mN2O, mair=mair, mtot=mtot, V=V, V_liq=0.0, quality=1.0, phase="gas")

    # ---- pure air ----
    if mN2O <= M_FLOOR:
        T = U / (mair * CV_AIR)
        P = mair * R_AIR * T / V
        out.update(T=T, P=P, rho=mair / V, u=U / mtot, h=CP_AIR * T)
        return out

    # ---- pure N2O ----
    if mair <= M_FLOOR:
        rho = mN2O / V
        u = U / mN2O
        try:
            _AS.update(CP.DmassUmass_INPUTS, rho, u)
            T, P, Q, h = _AS.T(), _AS.p(), _AS.Q(), _AS.hmass()
        except Exception:
            T = brentq(lambda TT: _n2o_u_rho(rho, TT) - u, TTRIP + 1, TCRIT - 0.2)
            P, Q, h = _n2o_PQh(rho, T)
        if 0.0 <= Q <= 1.0:
            _, rl, rv, _, _ = sat(T)
            V_liq = mN2O * (1 - Q) / rl
            out.update(V_liq=V_liq, quality=Q, phase="two-phase")
        out.update(T=T, P=P, rho=rho, u=u, h=h)
        return out

    # ---- mixture: N2O (CoolProp) + air (ideal gas) sharing V, common T ----
    def energy_residual(T):
        return _mix_U(mN2O, mair, T, V)[0] - U

    Tlo, Thi = max(TTRIP + 1.0, 90.0), 1000.0
    try:
        T = brentq(energy_residual, Tlo, Thi, tol=1e-5)
    except Exception:
        T = T_AMBIENT
    Utot, P, V_liq, Q, phase, h = _mix_U(mN2O, mair, T, V, want_props=True)
    out.update(T=T, P=P, rho=mtot / V, u=U / mtot, h=h, V_liq=V_liq, quality=Q, phase=phase)
    return out


def _mix_U(mN2O, mair, T, V, want_props=False):
    """Internal energy of an N2O+air mixture at temperature T in volume V.
       Returns U (and, if want_props, P,V_liq,Q,phase,h_mix)."""
    Psat, rl, rv, ul, uv = sat(T)
    # candidate two-phase liquid mass (vapour fills gas space at rho_v)
    denom = (1.0 - rv / rl)
    mass_liq = (mN2O - rv * V) / denom if denom != 0 else -1.0

    if 0.0 < mass_liq < mN2O and T < TCRIT:
        # --- N2O two-phase, air in gas space ---
        mass_vap = mN2O - mass_liq
        V_liq = mass_liq / rl
        V_gas = max(V - V_liq, 1e-12)
        U_n2o = mass_liq * ul + mass_vap * uv
        P_air = mair * R_AIR * T / V_gas
        U = U_n2o + mair * CV_AIR * T
        if not want_props:
            return U, None
        P = Psat + P_air
        h_l = ul + Psat / rl
        h_v = uv + Psat / rv
        h_air = CP_AIR * T
        h_mix = (mass_liq * h_l + mass_vap * h_v + mair * h_air) / (mN2O + mair)
        Q = mass_vap / mN2O
        return U, P, V_liq, Q, "two-phase", h_mix

    elif mass_liq <= 0.0 or T >= TCRIT:
        # --- N2O all vapour/supercritical, shares full V with air ---
        rho_n = mN2O / V
        try:
            _AS.update(CP.DmassT_INPUTS, rho_n, T)
            Pn, un, hn = _AS.p(), _AS.umass(), _AS.hmass()
        except Exception:
            Pn = rho_n * 188.9 * T          # crude ideal-gas fallback (R_N2O~188.9)
            un = 0.75e3 * T
            hn = un + Pn / rho_n
        P_air = mair * R_AIR * T / V
        U = mN2O * un + mair * CV_AIR * T
        if not want_props:
            return U, None
        P = Pn + P_air
        h_mix = (mN2O * hn + mair * CP_AIR * T) / (mN2O + mair)
        return U, P, 0.0, 1.0, "gas", h_mix

    else:
        # --- N2O essentially all liquid; air squeezed into tiny bubble ---
        V_liq = min(mN2O / rl, V * (1 - 1e-6))
        V_gas = max(V - V_liq, 1e-9)
        U_n2o = mN2O * ul
        P_air = mair * R_AIR * T / V_gas
        U = U_n2o + mair * CV_AIR * T
        if not want_props:
            return U, None
        P = Psat + P_air
        h_l = ul + Psat / rl
        h_mix = (mN2O * h_l + mair * CP_AIR * T) / (mN2O + mair)
        return U, P, V_liq, 0.0, "liquid", h_mix


def _n2o_u_rho(rho, T):
    _AS.update(CP.DmassT_INPUTS, rho, T)
    return _AS.umass()


def _n2o_PQh(rho, T):
    _AS.update(CP.DmassT_INPUTS, rho, T)
    return _AS.p(), _AS.Q(), _AS.hmass()


# =====================================================================================
#  5.  BRANCH FLOW MODELS
# =====================================================================================

def darcy_f(Re, D):
    """Darcy friction factor via Haaland (turbulent) or laminar 64/Re."""
    if Re < 1.0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re
    rr = ROUGHNESS / D
    return (-1.8 * np.log10((rr / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def friction_flow(b, si, sj):
    """Quasi-steady Darcy-Weisbach mass flow between two nodes (signed, i->j positive)."""
    Pi, Pj = si["P"], sj["P"]
    dP = Pi - Pj
    if abs(dP) < 1.0:
        return 0.0
    up = si if dP > 0 else sj
    rho = max(up["rho"], 1e-4)
    A, D, L = b["A"], b["D"], b["L"]
    # iterate friction factor with an internal velocity estimate
    mu = 2.0e-5
    f = 0.02
    v = 0.0
    for _ in range(3):
        K = f * L / max(D, 1e-6) + b["K_minor"] + 1e-6
        v = np.sqrt(2.0 * abs(dP) / (rho * K))
        Re = rho * v * D / mu
        f = darcy_f(Re, D)
    mdot = rho * A * v
    return mdot if dP > 0 else -mdot


def orifice_flow(b, si, sj, valve_open):
    """
    Valve orifice flow (i->j positive).  Returns (mdot, choked_flag).
    Uses HEM for N2O-dominated upstream, ideal-gas nozzle for air-dominated upstream,
    limited by the valve Kv.  Handles both flow directions.
    """
    if not valve_open:
        return 0.0, False
    v = VALVES[b["valve"]]
    A = np.pi / 4.0 * v["d_orifice"] ** 2
    Cd = v["Cd"]

    Pi, Pj = si["P"], sj["P"]
    if abs(Pi - Pj) < 1.0:
        return 0.0, False
    forward = Pi > Pj
    up = si if forward else sj
    P_up = up["P"]
    P_dn = sj["P"] if forward else si["P"]
    xN2O = up["mN2O"] / max(up["mtot"], 1e-30)

    if xN2O >= N2O_RICH:
        G, choked = hem_mass_flux(up, P_dn)
    else:
        G, choked = gas_mass_flux(up, P_dn)

    mdot = Cd * A * G
    kv = v.get("Kv", v.get("Cv", 0.0))
    mdot = min(mdot, kv_limit(kv, abs(P_up - P_dn), up["rho"]))
    mdot = max(mdot, 0.0)
    return (mdot, choked) if forward else (-mdot, choked)


def hem_mass_flux(up, P_dn):
    """Homogeneous Equilibrium Model mass flux [kg/(m^2 s)] with choking detection."""
    T = up["T"]
    # upstream N2O stagnation entropy & enthalpy
    try:
        if up["phase"] == "two-phase":
            _AS.update(CP.QT_INPUTS, min(max(up["quality"], 0.0), 1.0), T)
        elif up["phase"] == "liquid":
            _AS.update(CP.QT_INPUTS, 0.0, T)
        else:
            _AS.update(CP.DmassT_INPUTS, max(up["mN2O"] / up["V"], 1e-6), T)
        s0, h0 = _AS.smass(), _AS.hmass()
        Psat0 = n2o_Psat(T)
        P_up_n2o = Psat0 if up["phase"] in ("two-phase", "liquid") else _AS.p()
    except Exception:
        return 0.0, False

    P_dn = min(P_dn, P_up_n2o - 1.0)
    if P_dn <= 0:
        P_dn = 1.0
    Ps = np.linspace(P_up_n2o, max(P_dn, 1.0e3), HEM_NPTS)
    G = np.zeros_like(Ps)
    for k, Pt in enumerate(Ps):
        try:
            _AS.update(CP.PSmass_INPUTS, Pt, s0)
            rt, ht = _AS.rhomass(), _AS.hmass()
            dh = h0 - ht
            G[k] = rt * np.sqrt(2.0 * dh) if dh > 0 else 0.0
        except Exception:
            G[k] = 0.0
    kmax = int(np.argmax(G))
    # choked = mass flux peaks at a throat pressure strictly above the downstream
    # pressure (critical/back-pressure-independent flow), with a small margin so the
    # flag does not chatter right at the critical point.
    choked = (kmax <= HEM_NPTS - 2) and (G[kmax] > 1.002 * G[-1])
    return float(G[kmax]), bool(choked)


def gas_mass_flux(up, P_dn):
    """Compressible ideal-gas (air) nozzle mass flux per unit area, with choking."""
    P_up = up["P"]
    T = up["T"]
    R = R_AIR
    g = GAMMA_A
    r_crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    ratio = max(P_dn / P_up, 1e-6)
    if ratio <= r_crit:                      # choked
        G = P_up * np.sqrt(g / (R * T) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
        return G, True
    term = ratio ** (2.0 / g) - ratio ** ((g + 1.0) / g)
    G = P_up * np.sqrt(2.0 * g / ((g - 1.0) * R * T) * max(term, 0.0))
    return G, False


def kv_limit(Kv, dP, rho):
    """Single-phase incompressible mass-flow limit from valve Kv (SI conversion)."""
    if Kv <= 0 or dP <= 0:
        return 0.0
    SG = max(rho / 1000.0, 1e-6)
    Q_m3h = Kv * np.sqrt((dP / 1.0e5) / SG)  # dP in bar
    return rho * Q_m3h / 3600.0


# =====================================================================================
#  6.  VALVE SCHEDULE
# =====================================================================================

def valve_is_open(vidx, t):
    state = 0
    for (ts, s) in VALVE_SCHEDULE[vidx]:
        if t >= ts:
            state = s
    return bool(state)


# =====================================================================================
#  7.  INITIAL CONDITIONS
# =====================================================================================

mN2O = np.zeros(N)
mair = np.zeros(N)
U    = np.zeros(N)

# reserve tank: N2O (liquid + vapour) at its temperature
Tr = RESERVE["T"]
Psr, rl_r, rv_r, ul_r, uv_r = sat(Tr)
Vl = RESERVE["liq_fill"] * RESERVE["V"]
Vv = RESERVE["V"] - Vl
mliq = rl_r * Vl
mvap = rv_r * Vv
mN2O[i_reserve] = mliq + mvap
mair[i_reserve] = 0.0
U[i_reserve]    = mliq * ul_r + mvap * uv_r

# every other node: air at P_ATM, ambient temperature
for idx, nd in enumerate(nodes):
    if idx == i_reserve:
        continue
    T0 = ROCKET["T"] if idx == i_rocket else T_AMBIENT
    rho_a = P_ATM / (R_AIR * T0)
    mair[idx] = rho_a * nd["V"]
    mN2O[idx] = 0.0
    U[idx]    = mair[idx] * CV_AIR * T0

# =====================================================================================
#  8.  TIME INTEGRATION  (explicit Euler)
# =====================================================================================

nsteps = int(round(T_TOTAL / DT))
print(f"Nodes: {N} | Branches: {len(branches)} | Steps: {nsteps} (dt={DT:g} s)")
print("Running explicit-Euler integration ...")

# recording arrays
rec_t = []
rec_P = []          # [step][node]
rec_T = []
rec_level = []
rec_mdot_valve = [[] for _ in range(5)]
choke_events = []            # (t, valve_idx) onset events
choke_active = [False] * 5   # current choke state per valve (for edge detection)
choke_steps  = [0] * 5       # number of recorded steps each valve was choked
choke_first  = [None] * 5    # first time each valve choked

# map orifice branch index -> valve number
orifice_branches = {b["valve"]: bi for bi, b in enumerate(branches) if b["kind"] == "orifice"}

for step in range(nsteps + 1):
    t = step * DT
    # terminal progress counter: updates the same line with current simulated time
    if PRINT_EVERY and (step % max(1, PRINT_EVERY) == 0):
        print(f"\rSim time: {t:8.4f} s", end='', flush=True)

    # monitor sustained rocket fill condition: 15% volume for 0.5 s
    if step == 0:
        _sustain_time = 0.0

    # ---- evaluate all node states ----
    states = [node_state(mN2O[k], mair[k], U[k], nodes[k]["V"]) for k in range(N)]

    # ---- evaluate all branch flows ----
    dmN2O = np.zeros(N)
    dmair = np.zeros(N)
    dU    = np.zeros(N)
    valve_mdot = [0.0] * 5

    for b in branches:
        i, j = b["i"], b["j"]
        si, sj = states[i], states[j]
        if b["kind"] == "friction":
            mdot = friction_flow(b, si, sj)
            choked = False
        else:
            vidx = b["valve"]
            mdot, choked = orifice_flow(b, si, sj, valve_is_open(vidx, t))
            valve_mdot[vidx] = mdot
            if choked:
                choke_steps[vidx] += 1
                if choke_first[vidx] is None:
                    choke_first[vidx] = t
                    print(f"  [CHOKED]  valve{vidx+1} first choked at t = {t:8.4f} s "
                          f"(P_up={si['P']/1e5:6.2f} bar -> P_dn={sj['P']/1e5:6.2f} bar)")
                if not choke_active[vidx]:
                    choke_events.append((t, vidx))
                    choke_active[vidx] = True
            else:
                choke_active[vidx] = False

        if mdot == 0.0:
            continue

        # upstream node (source of mass & enthalpy)
        up = si if mdot > 0 else sj
        m_up = abs(mdot)
        # outflow limiter: don't drain more than a fraction of upstream species per step
        cap = MAX_DRAIN * max(up["mtot"], M_FLOOR) / DT
        if m_up > cap:
            m_up = cap
        # split by upstream mass fractions (homogeneous assumption)
        fN = up["mN2O"] / max(up["mtot"], 1e-30)
        fA = 1.0 - fN
        mN = m_up * fN
        mA = m_up * fA
        h_up = up["h"]
        sgn = 1.0 if mdot > 0 else -1.0

        # apply to i (positive mdot leaves i) and j
        dmN2O[i] -= sgn * mN
        dmN2O[j] += sgn * mN
        dmair[i] -= sgn * mA
        dmair[j] += sgn * mA
        dU[i]    -= sgn * m_up * h_up
        dU[j]    += sgn * m_up * h_up

    # ---- record ----
    if step % RECORD_EVERY == 0 or step == nsteps:
        rec_t.append(t)
        rec_P.append([s["P"] for s in states])
        rec_T.append([s["T"] for s in states])
        rec_level.append(states[i_rocket]["V_liq"] / nodes[i_rocket]["A_cross"])
        for vi in range(5):
            rec_mdot_valve[vi].append(valve_mdot[vi])

    # check finish condition: sustained liquid fraction >= target for 0.5 s
    if STOP_ON_ROCKET_FILL:
        rocket_liq_frac = states[i_rocket]["V_liq"] / nodes[i_rocket]["V"]
        if rocket_liq_frac >= ROCKET_FILL_TARGET:
            _sustain_time += DT
        else:
            _sustain_time = 0.0
        if _sustain_time >= 0.5:
            # Close valves 1-4 after fill target sustained for 0.5s and continue for 10 more seconds
            print(f"Rocket liquid fraction >={ROCKET_FILL_TARGET*100:.1f}% for {_sustain_time:.3f} s at t={t:.4f}s")
            print(f"Closing valves 1-4, continuing simulation for 10 more seconds...")
            # Dynamically add close events for valves 1-4 at current time
            for vi in range(4):
                VALVE_SCHEDULE[vi].append((t, 0))
            # Continue until 10 seconds after this point
            fill_reach_time = t
            _sustain_time = 0.0  # Reset to prevent multiple triggers
    else:
        _sustain_time = 0.0
    
    # Stop 10 seconds after rocket fill target is reached
    if STOP_ON_ROCKET_FILL and hasattr(locals().get('fill_reach_time', None), '__float__'):
        if t >= fill_reach_time + 10.0:
            print(f"Stopping: 10 seconds elapsed after fill target reached at t={t:.4f}s")
            break

    if step == nsteps:
        break

    # ---- explicit Euler update ----
    mN2O += DT * dmN2O
    mair += DT * dmair
    U    += DT * dU
    np.clip(mN2O, M_FLOOR, None, out=mN2O)
    np.clip(mair, M_FLOOR, None, out=mair)

    # stability guard
    if not np.all(np.isfinite(U)) or np.any(np.isfinite(U) == False):
        print(f"\n*** NUMERICAL INSTABILITY at t={t:.5f}s (step {step}). "
              f"Reduce DT and re-run. ***")
        break

print()
print("Integration finished.")

# =====================================================================================
#  9.  RESULTS  &  PLOTS
# =====================================================================================

rec_t = np.array(rec_t)
rec_P = np.array(rec_P)          # [nt, N]
rec_T = np.array(rec_T)
rec_level = np.array(rec_level)

name = [nd["name"] for nd in nodes]
plot_nodes = ["reserve"] + [f"pipe{k+1}" for k in range(4)] + ["rocket"]
plot_idx = [name.index(n) for n in plot_nodes]

# ---- choke summary ----
print("\n================ CHOKED-FLOW SUMMARY ================")
if not any(choke_first):
    print("No choked flow detected at any valve.")
else:
    for vi in range(5):
        if choke_first[vi] is not None:
            frac = 100.0 * choke_steps[vi] / max(nsteps, 1)
            print(f"  valve{vi+1}: first choked at t={choke_first[vi]:.4f} s, "
                  f"choked ~{frac:.1f}% of the run")
        else:
            print(f"  valve{vi+1}: never choked")
print("====================================================\n")

# ---- final-state table ----
print("Final node states (t = %.3f s):" % rec_t[-1])
print(f"{'node':>9} {'P [bar]':>10} {'T [K]':>9} {'phase':>11}")
final_states = [node_state(mN2O[k], mair[k], U[k], nodes[k]["V"]) for k in range(N)]
for k in plot_idx:
    s = final_states[k]
    print(f"{name[k]:>9} {s['P']/1e5:10.3f} {s['T']:9.2f} {s['phase']:>11}")
print(f"\nRocket-tank liquid level: {rec_level[-1]*1000:.1f} mm "
      f"({rec_level[-1]/ROCKET['H']*100:.1f} % of height)")
print(f"N2O delivered to rocket tank: {mN2O[i_rocket]:.4f} kg")

# ---- Figure 1: pressures ----
fig1, ax1 = plt.subplots(figsize=(10, 6))
for k in plot_idx:
    ax1.plot(rec_t, rec_P[:, k] / 1e5, label=name[k])
ax1.set_xlabel("time [s]"); ax1.set_ylabel("pressure [bar]")
ax1.set_title("Pressure in pipes and rocket tank")
ax1.grid(True, alpha=0.3); ax1.legend(ncol=2, fontsize=9)
fig1.tight_layout(); fig1.savefig("fill_pressures.png", dpi=130)

# ---- Figure 2: temperatures ----
fig2, ax2 = plt.subplots(figsize=(10, 6))
for k in plot_idx:
    ax2.plot(rec_t, rec_T[:, k], label=name[k])
ax2.set_xlabel("time [s]"); ax2.set_ylabel("temperature [K]")
ax2.set_title("Temperature in pipes and rocket tank")
ax2.grid(True, alpha=0.3); ax2.legend(ncol=2, fontsize=9)
fig2.tight_layout(); fig2.savefig("fill_temperatures.png", dpi=130)

# ---- Figure 3: rocket-tank liquid level ----
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(rec_t, rec_level * 1000.0, color="tab:blue")
ax3.set_xlabel("time [s]"); ax3.set_ylabel("liquid level [mm]")
ax3.set_title("Rising N2O liquid level in the rocket tank")
ax3.grid(True, alpha=0.3)
fig3.tight_layout(); fig3.savefig("fill_level.png", dpi=130)

# ---- Figure 4: valve mass flows ----
rec_mdot_valve = [np.asarray(mdot_series, dtype=float) for mdot_series in rec_mdot_valve]
fig4, axs4 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
for vi, ax in enumerate(axs4):
    mdot_arr = rec_mdot_valve[vi] * 1000.0
    if mdot_arr.size == 0:
        continue
    ax.plot(rec_t[:mdot_arr.size], mdot_arr, color=f"C{vi}")
    ax.set_ylabel("mass flow [g/s]")
    ax.set_title(f"Valve {vi+1} mass flow")
    ax.grid(True, alpha=0.3)
axs4[-1].set_xlabel("time [s]")
fig4.tight_layout(rect=[0, 0, 1, 0.96])
fig4.savefig("fill_valveflow.png", dpi=130)

# ---- Figure 5: valve positions (0=closed, 1=open) ----
rec_valve_pos = np.array([[1.0 if valve_is_open(vi, t) else 0.0 for t in rec_t] for vi in range(5)])
fig5, ax5 = plt.subplots(figsize=(10, 3))
for vi in range(5):
    ax5.step(rec_t, rec_valve_pos[vi, :], where='post', label=f"valve{vi+1}")
ax5.set_xlabel("time [s]"); ax5.set_ylabel("position")
ax5.set_title("Valve open/closed schedule (1=open, 0=closed)")
ax5.set_ylim(-0.1, 1.1)
ax5.grid(True, alpha=0.3); ax5.legend(ncol=5, fontsize=9)
fig5.tight_layout(); fig5.savefig("fill_valvepos.png", dpi=130)

plt.show()
print("\nSaved: fill_pressures.png, fill_temperatures.png, fill_level.png, fill_valveflow.png, fill_valvepos.png")
