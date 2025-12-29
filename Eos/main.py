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
tank_ox_len = 1 # m, oxidizer tank length, total inner length
tank_ox_ullage = 0.1 # fraction, liquid level fraction of tank
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
    #y = np.array([m, U, T_liq, T_vap])
    y = np.array([m, u])
    
    P = propsi ("P", "T", T0, "Q", x, "N2O")
    info = np.array([P, x, u])
    
    return y, info


def tank_ode_eq(y):
    m = y[0]
    u = y[1]
    
    P_ch = 20e5
    
    def volume_constr(T_func): 
        rho_liq = propsi ("D", "T", T_func, "Q", 0, "N2O")
        rho_vap = propsi ("D", "T", T_func, "Q", 1, "N2O")
        u_liq = propsi ("U", "T", T_func, "Q", 0, "N2O")
        u_vap = propsi ("U", "T", T_func, "Q", 1, "N2O")
        x = (u-u_liq) / (u_vap-u_liq)
        return m*((1-x)/rho_liq + x/rho_vap) - tank_ox_vol
    T = scipy.optimize.root(volume_constr, 270).x[0]
            
    P = propsi ("P", "T", T, "D", m/tank_ox_vol, "N2O")
    dm = - injector_flow_ox_hem(T, P, P_ch)
    h = propsi ("H", "T|liquid", T, "P", P, "N2O")
    du = dm / m * h

    ydot = np.array([dm, du])
    return ydot, T, P

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
y, info = tank_init_eq(290, tank_ox_vol, tank_ox_ullage)
ydot, T, P = tank_ode_eq(y)
