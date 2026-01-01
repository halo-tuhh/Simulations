# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 19:20:42 2025

@author: finnb
"""

"Parameters"
# Propellants
air_temp_celsius = 20 # deg C, surrounding air temp
fuel_name = 'Ethanol' # for fluid property lookup

# Tanks
tank_ox_diam_out = 0.11 # m, oxidizer tank outer diameter
tank_ox_thick = 0.003 # m, oxidizer tank wall thickness
tank_ox_len = 1.2 # m, oxidizer tank length, total inner length
tank_ox_ullage = 0.15 # fraction, liquid level fraction of tank
tank_fuel_diam_out = 0.05 # m, fuel tank outer diameter
tank_fuel_thick = 0.002 # m, fuel tank wall thickness
tank_fuel_len = 1.2 # m, fuel tank length, inner length below piston
tank_fuel_piston_loss = 0.0 # Pa, fuel tank piston pressure loss

# Valves
valve_ox_cv = 1.0 # flow coefficient of oxidizer run valve
valve_fuel_cv = 1.0 # flow coefficient of fuel run valve 

# Injectors
inj_ox_number = 8 # number of individual oxidizer orifices
inj_ox_diam_mm = 2.5 # mm, diameter of one ox orifice
inj_ox_cd = 0.8 # discharge coefficient oxidizer
inj_fuel_number = 12 # number of individual fuel orifices
inj_fuel_diam_mm = 0.7 # mm, diameter of one fuel orifice
inj_fuel_cd = 0.63 # discharge coefficient fuel

# Chamber
chamber_diam = 0.05 # m, chamber inner diameter
chamber_length = 0.1 # m, chamber length
chamber_throat = 0.04 # m, nozzle throat diameter
chamber_exit = 0.06 # m, nozzle exit diameter
chamber_Cstar_overwrite = 0.8 # combustion efficiency overwrite

# Rocket
rocket_mass_dry = 15 # kg, expected dry mass


"Setup"
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
from rocketcea.cea_obj_w_units import CEA_Obj 
C = CEA_Obj( oxName='N2O', fuelName=fuel_name, isp_units='m/s', cstar_units='m/s', pressure_units='Pa', temperature_units='K', 
            sonic_velocity_units='m/s', enthalpy_units='J/kg', density_units='kg/m^3', specific_heat_units='J/kg-K', viscosity_units='poise', 
            thermal_cond_units='W/cm-degC', fac_CR=None, make_debug_prints=False, useFastLookup=0, makeOutput=0,)

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

chamber_throat_area = chamber_throat**2 / 4 * np.pi # m^2, nozzle throat area
chamber_exit_area = chamber_exit**2 / 4 * np.pi # m^2, nozzle exit area
chamber_expansion = chamber_exit_area / chamber_throat_area # nozzle expansion ratio

# initial conditions
gravity = 9.81 # m/s^2
air_R = 287.07 # surrounding air gas constant 
air_pressure_zero = 101325.0  # Pa, standard pressure at MSL in Pascal
air_temp = 273.15 + air_temp_celsius # K


"Functions"

### Injector mass flow rates
def injector_flow_ox_hem(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### Two-phase Homogeneous Equilibrium Model
    dm = 0 # kg/s, mass flow rate
    ### Ox condition
    h1 = propsi ("H", "T|liquid", T1, "P", P1, "N2O")
    s1 = propsi ("S", "T|liquid", T1, "P", P1, "N2O")
    rho1 = propsi ("D", "T|liquid", T1, "P", P1, "N2O")
    ### Choked flow ( After Waxman 2014 - An Investigation of Injectors...)
    def flow(P2_func): 
        rho2 = propsi ("D", "P", max(P2_func, 1e5), "S", s1, "N2O")
        h2 = propsi ("H", "P", max(P2_func, 1e5), "S", s1, "N2O")
        if h2 <= h1: 
            dm = inj_ox_cd * inj_ox_area * rho2 * math.sqrt(2*(h1 - h2))
        else: 
            dm = 0
        return dm 
    P2_crit = scipy.optimize.fmin(lambda P: -flow(P), 1e5, maxiter=10, maxfun=20, disp=0) # find analytical function for critical pressure
    if P2 < P2_crit: # downstream (chamber) pressure
        dm = flow(P2_crit)    
    else: 
        dm = flow(P2) 

    dV = dm / rho1
    return  - dm, -dV # kg/s mass flow rate, m^3/s volume removal rate


def injector_flow_ox_gas(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### One-phase ideal gas model
    dm = 0 # kg/s, mass flow rate   
    ### Ox condition
    rho = propsi ("D", "T|gas", T1, "P", P1, "N2O")  
    cp = propsi ("Cpmass", "T|gas", T1, "P", P1, "N2O")
    cv = propsi ("Cvmass", "T|gas", T1, "P", P1, "N2O")
    gamma = cp/cv
    ### Adiabatic flow with choke condition
    P1_crit = P2 * ( 2 / (gamma+1) ) ** ( (gamma-1) / gamma )   
    if P1 > P2 and P1 > P1_crit : 
        dm = inj_ox_cd*inj_ox_area*np.sqrt( gamma * rho * P1 * (2/(gamma+1))**( (gamma+1)/(gamma-1) ) )
    elif P1 > P2 : 
        dm = inj_ox_cd*inj_ox_area*rho*np.sqrt( 2*cp*T1*( (P2/P1)**(2/gamma) - (P2/P1)**((gamma+1)/gamma) ) )  
    else:
        dm = 0.0
    dV = dm / rho
    return  -dm, -dV # kg/s mass flow rate, m^3/s volume removal rate


def injector_flow_fuel_spi(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### One-phase ideal liquid model    
    dm = 0 # kg/s, mass flow rate
    P1 = P1 - tank_fuel_piston_loss
    rho = propsi ("D", "T", T1, "P", P1, fuel_name)
    if P1 > P2:
        dm = inj_fuel_cd * inj_fuel_area * math.sqrt( 2*rho*(P1-P2) )
    else:
        dm = 0.0
    dV = dm / rho 
    return  - dm, -dV # kg/s mass flow rate, m^3/s volume removal rate


### Chamber functions
def chamber_combustion(dm_ox, dm_fuel): # kg/s, mass flow rates. Must be positive
    if dm_fuel > 0.01:
        dm_ratio = dm_ox / dm_fuel
        Pc = 1e5 # Pa, chamber pressure. Initial guess with arbitrary atmospheric
        for i in range(0, 10, 1): # run couple of times to converge Pc
            isp_vac, c_star, T_c = C.get_IvacCstrTc(Pc=Pc, MR=dm_ratio, eps=chamber_expansion, frozen=0, frozenAtThroat=0)       
            Pc = c_star * (dm_ox + dm_fuel) / chamber_throat_area * chamber_Cstar_overwrite
    else:
        Pc = 0.0; c_star = 0.0; isp_vac = 0.0
    return Pc, c_star, isp_vac # Pa chamber pressure, m/s combustion efficiency, m/s specific impulse, 

def chamber_nozzle(dm, isp): # kg mass flow rate, m/s specific impulse
    Ft = dm * isp
    return Ft # N, thrust force


### Tank initialization
def tank_init_eq(T0, Vtank, Vvap_fraction): # K initial temperature, m^3 tank volume, vapor volume fraction (dip tube length dependent)
    # m, T = y
    
    rho_liq = propsi ("D", "T", T0, "Q", 0, "N2O")
    rho_vap = propsi ("D", "T", T0, "Q", 1, "N2O")
    m_liq = rho_liq * Vtank * (1-Vvap_fraction)
    m_vap = rho_vap * Vtank * Vvap_fraction
    
    m = m_vap + m_liq # kg, oxidizer mass
    x = m_vap / m  # ratio, vapor mass / total mass
    
    P = propsi ("P", "T", T0, "Q", x, "N2O") # Pa, tank pressure
    
    rho_fuel = propsi ("D", "T", T0, "P", air_pressure_zero, fuel_name)
    m_fuel = rho_fuel * tank_fuel_vol
    
    y = np.array([m, m_fuel, T0])
    
    return y, P, x



"Main"

### Main engine run function with equilibrium tank
def engine_tank_eq(y, ydot, P_ch, step):
    m, mf, T = y
    
    ### Integration
    y = y + step * ydot # explicit Euler
    ydot = np.array([0, 0, 0])  # initialize 
    
    ### Abort condition
    if m < 0.1 or T < 200: # empty tank or too cold for coolprop
        return y, ydot, 0.0, 0.0, 0.0
    
    ### Ox tank 
    rho = m/tank_ox_vol # ox density
    P = propsi ("P", "T", T, "D", rho, "N2O")  # tank pressure    
    if T > 309: # critical point
        rho_liq = propsi ("D", "T", T, "P", P, "N2O")
        rho_vap = rho_liq
        x = 1
        h_vaporization = 0
    else:
        rho_liq = propsi ("D", "T", T, "Q", 0, "N2O")
        rho_vap = propsi ("D", "T", T, "Q", 1, "N2O")
        x = ( rho_vap*rho_liq - rho_vap*rho ) / ( rho*(rho_liq-rho_vap) )
        h_liq = propsi ("H", "T", T, "Q", 0, "N2O")
        h_vap = propsi ("H", "T", T, "Q", 1, "N2O")
        h_vaporization = h_vap - h_liq       
    if x < 0: x = 0 
    if x > 1: x = 1
    m_vap = m * x
    m_liq = m * (1-x)
    
    ### Fuel injector
    if mf > 0.01: 
        dmf, dVf = injector_flow_fuel_spi(T, P, P_ch)
    else:
        dmf = 0.0; dVf = 0.0      
         
    ### Ox injector       
    if m > 0.01 and m_liq > 0.01:
        dm, dV = injector_flow_ox_hem(T, P, P_ch)
        cp = propsi ("Cpmass", "T", T, "Q", 0, "N2O")
        dT = ( h_vaporization * (dV+dVf)*rho_vap/m ) / cp # ox and fuel outflow result in expansion -> vaporization -> temperature drop   
    elif m > 0.01:
        dm, dV = injector_flow_ox_gas(T, P, P_ch)
        cp = propsi ("Cpmass", "T|gas", T, "P", P, "N2O")
        dT =  ( P*dV/m ) / cp   
    else:
        dm = 0.0; dT = 0.0
        
    ### Chamber 
    P_ch, c_star, isp_vac = chamber_combustion(-dm, -dmf)
    #P_ch = (1-time_const) * P_ch + time_const * P_ch_instant # 1st order lowpass to smooth
    F_thrust = chamber_nozzle( -(dm+dmf), isp_vac)
        
    ### Return
    ydot = np.array([dm, dmf, dT])
    
    return y, ydot, P, P_ch, F_thrust, isp_vac



"Run simulation"
### Setup
T_0 = 290
t_end = 8
step = 0.02

sol_t = np.arange(0, t_end, step)
sol_y = np.zeros((len(sol_t), 3))
sol_ydot = np.zeros((len(sol_t), 3))
sol_P = np.zeros(len(sol_t))
sol_Pc = np.zeros(len(sol_t))
sol_Ft = np.zeros(len(sol_t))
sol_isp = np.zeros(len(sol_t))

y0, P, x = tank_init_eq(T_0, tank_ox_vol, tank_ox_ullage)
sol_y[0] = y0
sol_ydot[0] = np.array([0, 0, 0])
sol_Pc[0] = air_pressure_zero

### Main loop
for t in range(1, len(sol_t), 1):

    y, ydot, P_tank, P_ch, F_thrust, isp_vac = engine_tank_eq(sol_y[t-1], sol_ydot[t-1], sol_Pc[t-1], step)
   
    sol_y[t] = y
    sol_ydot[t] = ydot
    sol_P[t] = P_tank
    sol_Pc[t] = P_ch
    sol_Ft[t] = F_thrust
    sol_isp[t] = isp_vac


"Plot"

plt.figure(1)
plt.plot(sol_t, sol_y[:,0])
plt.plot(sol_t, sol_y[:,1])
# plt.figure(2)
# plt.plot(sol_t, sol_y[:,2])
plt.figure(3)
plt.plot(sol_t, -sol_ydot[:,0])
plt.plot(sol_t, -sol_ydot[:,1])
plt.figure(4)
plt.plot(sol_t, sol_P)
plt.plot(sol_t, sol_Pc)
plt.figure(5)
plt.plot(sol_t, sol_Ft)
#plt.figure(5)
plt.plot(sol_t, sol_isp)
