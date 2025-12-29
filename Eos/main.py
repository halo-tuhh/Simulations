# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 19:20:42 2025

@author: finnb
"""

"Parameters"
# Propellants
air_temp_celsius = 20 # deg C, surrounding air temp
fuel_name = 'ethanol' # for fluid property lookup

# Tanks
tank_ox_diam_out = 0.11 # m, oxidizer tank outer diameter
tank_ox_thick = 0.003 # m, oxidizer tank wall thickness
tank_ox_len = 1.5 # m, oxidizer tank length, total inner length
tank_ox_ullage = 0.2 # fraction, liquid level fraction of tank
tank_fuel_diam_out = 0.05 # m, fuel tank outer diameter
tank_fuel_thick = 0.002 # m, fuel tank wall thickness
tank_fuel_len = 1 # m, fuel tank length, inner length below piston

# Valves
valve_ox_cv = 1.0 # flow coefficient of oxidizer run valve
valve_fuel_cv = 1.0 # flow coefficient of fuel run valve 

# Injectors
inj_ox_number = 12 # number of individual oxidizer orifices
inj_ox_diam_mm = 2.5 # mm, diameter of one ox orifice
inj_ox_cd = 0.8 # discharge coefficient oxidizer
inj_fuel_number = 12 # number of individual fuel orifices
inj_fuel_diam_mm = 1 # mm, diameter of one fuel orifice
inj_fuel_cd = 0.63 # discharge coefficient fuel

# Chamber
chamber_diam = 0.05 # m, chamber inner diameter
chamber_length = 0.1 # m, chamber length
chamber_throat = 0.003 # m, nozzle throat diameter
chamber_exit = 0.006 # m, nozzle exit diameter
chamber_Cstar_overwrite = 0.8 # combustion efficiency overwrite

# Rocket
rocket_mass_dry = 15 # kg, expected dry mass


"Script"
# imports
import math
import numpy as np
np.seterr(all='ignore')
import scipy
import matplotlib.pyplot as plt
import sys
import xml.etree.ElementTree as ET
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI as propsi
import rocketcea
from rocketcea.cea_obj import CEA_Obj 

# geometry
tank_ox_diam = tank_ox_diam_out - 2*tank_ox_thick # m, oxidizer tank inner diameter
tank_fuel_diam = tank_fuel_diam_out - 2*tank_fuel_thick # m, fuel tank inner diameter
tank_fuel_vol = (tank_fuel_diam/2)**2*np.pi * tank_fuel_len # m^3, fuel tank volume
tank_ox_vol = (tank_ox_diam/2)**2*np.pi * tank_ox_len - tank_fuel_vol# m^3, oxidizer tank volume
tank_ox_mass = 2700 * tank_ox_diam*tank_ox_thick*tank_ox_len # kg, oxidizer tank mass
tank_ox_heat = 910 # J/(kg K), specific heat of aluminium

inj_ox_diam = inj_ox_diam_mm/1000
inj_fuel_diam = inj_fuel_diam_mm/1000
inj_ox_area = (inj_ox_diam/2)**2 * np.pi * inj_ox_number # m^2, cross-section area of all oxidizer injector orifices
inj_fuel_area = (inj_fuel_diam/2)**2 * np.pi * inj_fuel_number # m^2, cross-section area of all fuel injector orifices

# initial conditions
gravity = 9.81 # m/s^2
air_R = 287.07 # surrounding air gas constant 
air_pressure_zero = 101325.0  # Pa, standard pressure at MSL in Pascal
air_temp = 273.15 + air_temp_celsius # K


"Functions"

# Tank thermodynamics
def tank_init_eq(T0, Vtank, Vvap_fraction):
    #m = y[0]
    #U = y[1]
    
    rho_liq = propsi ("D", "T", T0, "Q", 0, "N2O")
    rho_vap = propsi ("D", "T", T0, "Q", 1, "N2O")
    m_liq = rho_liq * Vtank * (1-Vvap_fraction)
    m_vap = rho_vap * Vtank * Vvap_fraction
    
    m = m_vap + m_liq   
    x = m_vap / m  
    T = T0
    u = propsi ("U", "T", T0, "Q", x, "N2O")
    T_liq = T
    T_vap = T      
    
    P = propsi ("P", "T", T0, "Q", x, "N2O")
    
    #y = np.array([m, U, T_liq, T_vap])
    y = np.array([m, u, T])
    info = np.array([P, x, u])
    
    return y, info




def tank_ode_eq(t, y):
    m, u = y
    ydot = np.array([0, 0])
    
    if m < 0.01:
        return ydot
    
    P_ch = 1e5
    
    def volume_constr(T_func): 
        rho_liq = propsi ("D", "T", T_func, "Q", 0, "N2O")
        rho_vap = propsi ("D", "T", T_func, "Q", 1, "N2O")
        u_liq = propsi ("U", "T", T_func, "Q", 0, "N2O")
        u_vap = propsi ("U", "T", T_func, "Q", 1, "N2O")
        x = (u-u_liq) / (u_vap-u_liq)
        return m*((1-x)/rho_liq + x/rho_vap) - tank_ox_vol
    T = scipy.optimize.brentq(volume_constr, 100, 309.5, maxiter=80)
    #print(T)
    if T < 200 :
        return ydot
            
    P = propsi ("P", "T", T, "D", m/tank_ox_vol, "N2O")
    dm = - injector_flow_ox_hem(T, P, P_ch)
    h = propsi ("H", "T|liquid", T, "P", P, "N2O")
    du = dm / m * h + 1000*(290-T)

    ydot = np.array([dm, du])
    return ydot



def tank_ode_eq2(t, y):
    m, u, Twall = y
    ydot = np.array([0, 0, 0])
    
    if m < 0.1:
        return ydot
    
    P_ch = 1e5
    
    def volume_constr(T_func): 
        rho_liq = propsi ("D", "T", T_func, "Q", 0, "N2O")
        rho_vap = propsi ("D", "T", T_func, "Q", 1, "N2O")
        u_liq = propsi ("U", "T", T_func, "Q", 0, "N2O")
        u_vap = propsi ("U", "T", T_func, "Q", 1, "N2O")
        x = (u-u_liq) / (u_vap-u_liq)
        return m*((1-x)/rho_liq + x/rho_vap) - tank_ox_vol
    T = scipy.optimize.brentq(volume_constr, 200, 309.5, maxiter=80)
    if T < 200 :
        return ydot
    
    u_liq = propsi ("U", "T", T, "Q", 0, "N2O")
    u_vap = propsi ("U", "T", T, "Q", 1, "N2O")
    x = (u-u_liq) / (u_vap-u_liq)
            
    P = propsi ("P", "T", T, "D", m/tank_ox_vol, "N2O")
    dm = - injector_flow_ox_hem(T, P, P_ch)
    h = propsi ("H", "T|liquid", T, "P", P, "N2O")
    
    
    def heat_flow(T_diff, coeff_c, coeff_n):
        L = tank_ox_diam_out
        cp = propsi ("Cpmass", "T", T, "Q", x, "N2O")
        rho = propsi ("D", "T", T, "Q", x, "N2O")
        #mu = propsi ("viscosity", "T", T, "Q", x, "N2O")
        mu = 15e-6
        #k = propsi ("conductivity", "T", T, "Q", x, "N2O")
        k = 50e-3
        beta = propsi ("isobaric_expansion_coefficient", "T", T, "Q", x, "N2O")
        Ra = cp * rho**2 * gravity*beta*abs(T_diff)*L**3 / (mu * k)
        dQ = coeff_c * Ra**coeff_n * k/L
        return dQ
    
    dQ_air = heat_flow(Twall-air_temp, 0.59, 0.25)
    dQ_tank = heat_flow(Twall-T, 0.21, 0.4)   

    du = dm / m * h # + dQ_tank / m
    
    dTwall = -0.01
    # dTwall = ( - dQ_tank - dQ_air) / ( tank_ox_mass * tank_ox_heat)

    ydot = np.array([dm, du, dTwall])
    return ydot




def tank_pt_eq(y):
    m, u, Twall = y
    
    def volume_constr(T_func): 
        rho_liq = propsi ("D", "T", T_func, "Q", 0, "N2O")
        rho_vap = propsi ("D", "T", T_func, "Q", 1, "N2O")
        u_liq = propsi ("U", "T", T_func, "Q", 0, "N2O")
        u_vap = propsi ("U", "T", T_func, "Q", 1, "N2O")
        x = (u-u_liq) / (u_vap-u_liq)
        return m*((1-x)/rho_liq + x/rho_vap) - tank_ox_vol
    T = scipy.optimize.brentq(volume_constr, 100, 309.5, maxiter=80)
    
    P = propsi ("P", "T", T, "D", m/tank_ox_vol, "N2O")

    return P, T

# Injector mass flow rates
def injector_flow_ox_hem(T1, P1, P2):
    # Two-phase Homogeneous Equilibrium Model
    mdot = 0 # kg/s, mass flow rate
    h1 = propsi ("H", "T|liquid", T1, "P", P1, "N2O")
    s1 = propsi ("S", "T|liquid", T1, "P", P1, "N2O")
    #h2 = propsi ("H", "P", max(P2, 1e4), "S", s1, "N2O")
    def flow(P2_func): 
        density = propsi ("D", "P", max(P2_func, 1e5), "S", s1, "N2O")
        h2 = propsi ("H", "P", max(P2_func, 1e5), "S", s1, "N2O")
        if h2 <= h1: 
            mdot = inj_ox_cd * inj_ox_area * density * math.sqrt(2*(h1 - h2))
        else: 
            mdot = 0
        return mdot 
    P2_crit = scipy.optimize.fmin(lambda P: -flow(P), P1, maxiter=10, maxfun=20, disp=0)
    if P2 < P2_crit: 
        mdot = flow(P2_crit)    
    else: 
        mdot = flow(P2) 
    return  mdot 

def injector_flow_fuel_spi(T1, P1, P2):
    mdot = 0 # kg/s, mass flow rate
    density = propsi ("D", "T", T1, "P", P1, fuel_name)
    mdot = inj_fuel_cd * inj_fuel_area * math.sqrt( 2*density*(P1-P2) )
    return  mdot 


"Testing ground"
# Injectors
# T1 = 273.15 
# test_Pv = propsi ("P", "T", T1, "Q", 0, "N2O")
# test_ox_mdot_hem = injector_flow_ox_hem(T1, test_Pv+1E3, 2E6)
# test_fuel_mdot = injector_flow_fuel_spi(T1, test_Pv+1E3, 2E6)
# P2min = 0
# P2max = 4E6
# N = 20
# P2s = np.linspace(P2min, P2max, num=N)
# mdots = np.zeros(N)
# n = 0 
# while n < N :
#     mdots[n] = injector_flow_ox_hem(T1, test_Pv+1E3, P2s[n])
#     n += 1
# plt.figure(1)
# plt.plot(P2s, mdots)

# Tanks
T_0 = 290
y0, info = tank_init_eq(T_0, tank_ox_vol, tank_ox_ullage)
ydot = tank_ode_eq2(0, y0)

t_end = 10
time_span = [0, t_end]
step = 0.1
sol_t = np.arange(0, t_end, step)
sol_P = np.zeros(len(sol_t))
sol_T = np.zeros(len(sol_t))
sol = scipy.integrate.solve_ivp(tank_ode_eq2, time_span, y0, method='RK45', max_step=0.5, dense_output=True, rtol=1e-6, atol=1e-5)
sol_y = sol.sol(sol_t)

plt.figure(1)
plt.plot(sol_t, sol_y[0].T)
for i in range(0, len(sol_t), 1):
    sol_P[i], sol_T[i] = tank_pt_eq(sol_y.T[i])
plt.figure(2)
plt.plot(sol_t, sol_P)
plt.figure(3)
plt.plot(sol_t, sol_T)
plt.plot(sol_t, sol_y[2].T)
    
