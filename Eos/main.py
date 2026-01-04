# -*- coding: utf-8 -*-
"""
Nitrous blowdown engine
@author: Finn Breuer
Created on Sun Dec  7 19:20:42 2025
"""

"Parameters"
# Operations
ops_temp_celsius = 20 # deg C, initial tank temperature after filling
ops_fuel_name = 'Ethanol' # for fluid property lookup
ops_pressurant_name = 'N2' # for fluid property lookup
ops_pressurizing_enable = 0 # boolean, enable inert gas pressurization on the pad
ops_heating_enable = 0 # boolean, enable tank heating on the pad
ops_pressure_target = 5e6 # Pa, firing pressure. Requires either presurization or heating enabled

# Tanks
tank_ox_diam_out = 0.11 # m, oxidizer tank outer diameter
tank_ox_thick = 0.003 # m, oxidizer tank wall thickness
tank_ox_len = 1.0 # m, oxidizer tank length, total inner length
tank_ox_ullage = 0.15 # fraction, liquid level fraction of tank
tank_fuel_diam_out = 0.05 # m, fuel tank outer diameter
tank_fuel_thick = 0.002 # m, fuel tank wall thickness
tank_fuel_len = 1.0 # m, fuel tank length, inner length below piston
tank_fuel_piston_loss = 0.0 # Pa, fuel tank piston pressure loss

# Valves
valve_ox_cv = 0.8 # flow coefficient of oxidizer run valve
valve_fuel_cv = 0.8 # flow coefficient of fuel run valve 

# Injectors
inj_ox_number = 12 # number of individual oxidizer orifices
inj_ox_diam_mm = 2.4 # mm, diameter of one ox orifice
inj_ox_cd = 0.8 # discharge coefficient oxidizer
inj_fuel_number = 16 # number of individual fuel orifices
inj_fuel_diam_mm = 0.75 # mm, diameter of one fuel orifice
inj_fuel_cd = 0.63 # discharge coefficient fuel

# Chamber
chamber_diam = 0.05 # m, chamber inner diameter
chamber_length = 0.1 # m, chamber length
chamber_throat = 0.035 # m, nozzle throat diameter
chamber_exit = 0.055 # m, nozzle exit diameter
chamber_cstar_efficiency = 0.75 # factor, combustion efficiency
chamber_nozzle_efficiency = 0.95 # factor, expansion efficiency

# Rocket
rocket_mass_dry = 15 # kg, expected dry mass


"Requirements"
import math
import numpy as np
np.seterr(all='ignore')
import scipy
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI as propsi
import rocketcea
from rocketcea.cea_obj_w_units import CEA_Obj 
C = CEA_Obj( oxName='N2O', fuelName=ops_fuel_name, isp_units='m/s', cstar_units='m/s', pressure_units='Pa', temperature_units='K', 
            sonic_velocity_units='m/s', enthalpy_units='J/kg', density_units='kg/m^3', specific_heat_units='J/kg-K', viscosity_units='poise', 
            thermal_cond_units='W/cm-degC', fac_CR=None, make_debug_prints=False, useFastLookup=0, makeOutput=0,)


"Setup script"
# geometry
tank_ox_diam = tank_ox_diam_out - 2*tank_ox_thick # m, oxidizer tank inner diameter
tank_fuel_diam = tank_fuel_diam_out - 2*tank_fuel_thick # m, fuel tank inner diameter
tank_fuel_vol = (tank_fuel_diam/2)**2*np.pi * tank_fuel_len # m^3, fuel tank volume
tank_ox_vol = (tank_ox_diam/2)**2*np.pi * tank_ox_len - tank_fuel_vol# m^3, oxidizer tank volume

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
air_temp = 273.15 + ops_temp_celsius # K
ops_pressurant_mass = 0.0 # kg, mass of inert press gas. Set to zero in case pressurization is disabled


"Injectos"

def injector_ox_hem(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### Two-phase Homogeneous Equilibrium Model
    dm = 0 # kg/s, mass flow rate
    ### Ox 
    if P1 > propsi ("P", "T", T1, "Q", 0, "N2O") + 1e3 :
        h1 = propsi ("H", "T", T1, "P", P1, "N2O")
        s1 = propsi ("S", "T", T1, "P", P1, "N2O")
        rho1 = propsi ("D", "T", T1, "P", P1, "N2O")
    else: 
        h1 = propsi ("H", "T", T1, "Q", 0, "N2O")
        s1 = propsi ("S", "T", T1, "Q", 0, "N2O")
        rho1 = propsi ("D", "T", T1, "Q", 0, "N2O")      
    ### Choked flow ( After Waxman 2014 - An Investigation of Injectors...)
    def flow(P2_func): 
        rho2 = propsi ("D", "P", max(P2_func, 1e5), "S", s1, "N2O")
        h2 = propsi ("H", "P", max(P2_func, 1e5), "S", s1, "N2O")
        if h2 < h1: 
            dm = - inj_ox_cd*valve_ox_cv*inj_ox_area * rho2 * math.sqrt(2*(h1 - h2))
        else: 
            dm = 0
        return dm 
    P2_crit = scipy.optimize.fminbound(flow, 1e5, P1, xtol=1e3, maxfun=10, disp=0) # find critical pressure of choked flow
    if P2 < P2_crit: # P2 is downstream (chamber) pressure
       dm = flow(P2_crit)    
    else: 
        dm = flow(P2) 
    dV = dm / rho1
    return  dm, dV # kg/s mass flow rate, m^3/s volume removal rate


def injector_ox_hem_gas(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
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
        dm = - inj_ox_cd*valve_ox_cv*inj_ox_area*np.sqrt( gamma * rho * P1 * (2/(gamma+1))**( (gamma+1)/(gamma-1) ) )
    elif P1 > P2 : 
        dm = - inj_ox_cd*valve_ox_cv*inj_ox_area*rho*np.sqrt( 2*cp*T1*( (P2/P1)**(2/gamma) - (P2/P1)**((gamma+1)/gamma) ) )  
    else:
        dm = 0.0
    dV = dm / rho
    return  dm, dV # kg/s mass flow rate, m^3/s volume removal rate


def injector_fuel_spi(T1, P1, P2): # K tank temperature, Pa tank pressure, Pa chamber pressure
    ### One-phase ideal liquid model    
    dm = 0 # kg/s, mass flow rate
    P1 = P1 - tank_fuel_piston_loss
    rho = propsi ("D", "T", T1, "P", P1, ops_fuel_name)
    if P1 > P2:
        dm = - inj_fuel_cd*valve_fuel_cv * inj_fuel_area * math.sqrt( 2*rho*(P1-P2) )
    else:
        dm = 0.0
    dV = dm / rho 
    return  dm, dV # kg/s mass flow rate, m^3/s volume removal rate


"Combustion cahmber and nozzle"

def chamber(dm_ox, dm_fuel): # kg/s, mass flow rates
    dm_ox = abs(dm_ox)
    dm_fuel = abs(dm_fuel)
    dm = dm_ox + dm_fuel
    if dm_fuel > 0.01:
        dm_ratio = dm_ox / dm_fuel
        Pc = 1e5 # Pa, chamber pressure. Initial guess with arbitrary atmospheric
        for i in range(0, 10, 1): # run couple of times to converge Pc
            c_star_opt = C.get_Cstar(Pc=Pc, MR=dm_ratio)       
            c_star = c_star_opt * chamber_cstar_efficiency
            Pc = c_star * dm / chamber_throat_area
        expansion = air_pressure_zero / Pc * C.get_PcOvPe(Pc=Pc, MR=dm_ratio, eps=chamber_expansion, frozen=0, frozenAtThroat=0)
        isp_amb = C.estimate_Ambient_Isp(Pc=Pc, MR=dm_ratio, eps=chamber_expansion, Pamb=air_pressure_zero)[0]
    else:
        Pc = 0.0; c_star = 0.0; isp_amb = 0.0
        
    F = dm * isp_amb # N, thrust force
        
    return F, Pc, c_star, isp_amb, expansion # N thrust force, Pa chamber pressure, m/s combustion efficiency, m/s specific impulse, 


"Tank fill and press"

def tank_init_eq(T0, Vtank, Vvap_fraction): # K initial temperature, m^3 tank volume, vapor volume fraction (dip tube length dependent)
    # finds initial state of ox tank  
    rho_liq = propsi ("D", "T", T0, "Q", 0, "N2O")
    rho_vap = propsi ("D", "T", T0, "Q", 1, "N2O")
    m_liq = rho_liq * Vtank * (1-Vvap_fraction)
    m_vap = rho_vap * Vtank * Vvap_fraction    
    m = m_vap + m_liq # kg, oxidizer mass
    x = m_vap / m  # ratio, vapor mass / total mass   
    P = propsi ("P", "T", T0, "Q", x, "N2O") # Pa, tank pressure
    rho_fuel = propsi ("D", "T", T0, "P", air_pressure_zero, ops_fuel_name)
    m_fuel = rho_fuel * tank_fuel_vol   
    y = np.array([m, m_fuel, T0])  
    return y, P, x


def tank_init_heating(y0, P_target): # initial state, Pa firing pressure target
    # heating to firing pressure after closing vent valve 
    m, mf, T = y0 # ox mass, fuel mass, tank temperature  
    rho = m/tank_ox_vol   
    T_heated = propsi ("T", "P", P_target, "D", rho, "N2O")   
    y = np.array([m, mf, T_heated])  
    return y


def tank_init_pressurizing(y0, P_target): # initial state, Pa firing pressure target
    # inert gas pressurization to firing pressure  after closing vent valve
    # assumes press on pad only -> constant press mass
    m, mf, T = y0 # ox mass, fuel mass, tank temperature  
    
    rho = m/tank_ox_vol # ox density
    P = propsi ("P", "T", T, "D", rho, "N2O")  # oxidizer tank pressure   
    P_press = P_target - P # Pa, inert gas partial pressure    
    if T > 309: # above critical point of N2O
       m_press = 0 # do not press in this region
       return m_press
    rho_liq = propsi ("D", "T", T, "Q", 0, "N2O")
    rho_vap = propsi ("D", "T", T, "Q", 1, "N2O")
    x = ( rho_vap*rho_liq - rho_vap*rho ) / ( rho*(rho_liq-rho_vap) )
    if x < 0: x = 0 
    if x > 1: x = 1
    m_vap = m * x # kg, ox vapor mass

    if P_press > 1e3: # only do if significant difference
        V_press = m_vap / rho_vap # only in ox gaseous phase
        rho_press = propsi ("D", "T", T, "P", P_press, ops_pressurant_name) # kg/m^3, inert gas initial density    
        m_press = rho_press * V_press
    else:
        m_press = 0
        
    return m_press



"Main"

### Main engine run function with equilibrium tank
def engine_tank_eq(y, ydot, P_ch, step):
    m, mf, T = y # ox mass, fuel mass, tank temperature
    
    ### Integration
    y = y + step * ydot # explicit Euler
    ydot = np.array([0, 0, 0])  # initialize 
    
    ### End condition
    if m < 0.01 or T < 100: # empty tank or too cold for coolprop
        return y, ydot, 0.0, 0.0, 0.0, 0.0
    
    ### Ox tank 
    rho = m/tank_ox_vol # ox density
    P = propsi ("P", "T", T, "D", rho, "N2O")  # oxidizer tank pressure    
    if T > 309: # above critical point of N2O
        rho_liq = propsi ("D", "T", T, "P", P, "N2O")
        rho_vap = rho_liq
        x = 1 # ratio, vapor quality: gas mass / total mass
        h_vaporization = 0
    else: # in liquid/vapor regime
        rho_liq = propsi ("D", "T", T, "Q", 0, "N2O")
        rho_vap = propsi ("D", "T", T, "Q", 1, "N2O")
        x = ( rho_vap*rho_liq - rho_vap*rho ) / ( rho*(rho_liq-rho_vap) )
        h_liq = propsi ("H", "T", T, "Q", 0, "N2O")
        h_vap = propsi ("H", "T", T, "Q", 1, "N2O")
        h_vaporization = h_vap - h_liq
    if x < 0: x = 0 
    if x > 1: x = 1
    m_liq = m * (1-x) # kg, ox liquid mass
    m_vap = m * x # kg, ox vapor mass
    
    ### Inert gas pressurization
    if ops_pressurizing_enable:
        V_pressurant = m_vap / rho_vap # only in ox gaseous phase
        rho_pressurant = ops_pressurant_mass / V_pressurant # assumes press on pad only -> constant press mass
        P_pressurant = propsi ("P", "T", T, "D", rho_pressurant, ops_pressurant_name) # Pa, inert gas partial pressure 
        P = P + P_pressurant # approximate total pressure
    
    ### Fuel injector
    if mf > 0.01: 
        dmf, dVf = injector_fuel_spi(T, P, P_ch) # kg/s, fuel mass flow rate
    else:
        dmf = 0.0; dVf = 0.0      
         
    ### Ox injector       
    if m > 0.01 and m_liq > 0.01:
        dm, dV = injector_ox_hem(T, P, P_ch) # kg/s, ox mass flow rate for two-phase HEM
        cp = propsi ("Cpmass", "T", T, "Q", 0, "N2O")
        dT = ( h_vaporization * (dV+dVf)*rho_vap/m ) / cp # adiabatic expansion -> vaporization -> temperature drop   
    elif m > 0.01:
        dm, dV = injector_ox_hem_gas(T, P, P_ch) # kg/s, ox mass flow rate for one-phase vapor only
        cp = propsi ("Cpmass", "T|gas", T, "P", P, "N2O")
        dT =  ( P*(dV+dVf)/m ) / cp # adiabatic expansion -> temperature drop  
    else:
        dm = 0.0; dT = 0.0
        
    ### Chamber combustion and nozzle
    F_thrust, P_ch, c_star, isp, expansion = chamber(dm, dmf) # N, Pa, m/s, m/s
    isp_real = F_thrust / abs(dm+dmf)    
    
    ### Return
    ydot = np.array([dm, dmf, dT]) # rates of: ox mass, fuel mass, tank temperature
    
    return y, ydot, P, P_ch, F_thrust, isp_real, expansion


"Run simulation"
### Setup
T_0 = 283
t_end = 15
step = 0.02

ops_pressurizing_enable = 0 # boolean, enable inert gas pressurization on the pad
ops_heating_enable = 0 # boolean, enable tank heating on the pad
ops_pressure_target = 5e6 # Pa, firing pressure. Requires either presurization or heating enabled

y0, P, x = tank_init_eq(T_0, tank_ox_vol, tank_ox_ullage)
if ops_heating_enable:
    y0 = tank_init_heating(y0, ops_pressure_target)
if ops_pressurizing_enable:
    ops_pressurant_mass = tank_init_pressurizing(y0, ops_pressure_target)
      
sol_t = np.arange(0, t_end, step)
sol_y = np.array([y0])
sol_ydot = np.array([[0, 0, 0]])
sol_P = np.array([0])
sol_Pc = np.array([air_pressure_zero])
sol_Ft = np.array([0])
sol_isp = np.array([0])   
sol_exp = np.array([0])   
        
### Main loop
for t in range(1, len(sol_t), 1):

    y, ydot, P_tank, P_ch, F_thrust, isp, expans = engine_tank_eq(sol_y[t-1], sol_ydot[t-1], sol_Pc[t-1], step)
    
    if y[0] <= 0.01 or y[1] <= 0.01 or y[2] < 100: # empty tank or too cold for coolprop
        sol_t = sol_t[:t]
        break
    
    sol_y = np.append(sol_y, [y], axis=0)
    sol_ydot = np.append(sol_ydot, [ydot], axis=0)
    sol_P = np.append(sol_P, P_tank)
    sol_Pc = np.append(sol_Pc, P_ch)
    sol_Ft = np.append(sol_Ft, F_thrust)
    sol_isp = np.append(sol_isp, isp)
    sol_exp = np.append(sol_exp, expans)

### Solution numbers
print("Burn time:", round(sol_t[len(sol_t)-1], 2), "s")
print("Isp max:", round(max(sol_isp)/9.81, 1), "s") 
print("Isp avg:", round(sum(sol_isp)/9.81/len(sol_t), 1), "s")     
print("Thrust max:", round(max(sol_Ft)), "N")
print("Thrust avg:", round(sum(sol_Ft)/len(sol_t)), "N")
print("Impulse:", round(sum(sol_Ft)/len(sol_t)*sol_t[len(sol_t)-1]/1000, 3), "kNs")
print("TWR max:", round(max(sol_Ft)/9.81/(rocket_mass_dry+sol_y[0,0]+sol_y[0,1]), 2))
print("P_chamber max:", round(max(sol_Pc)/1e5, 2), "bar")
print("P_tank max:", round(max(sol_P)/1e5, 2), "bar")
print("Ox mass left:", round(sol_y[len(sol_t)-1, 0], 4), "kg")
print("Fuel mass left:", round(sol_y[len(sol_t)-1, 1], 4), "kg")




"Plot"

plt.figure(1)
plt.plot(sol_t, sol_y[:,0], label='N2O')
plt.plot(sol_t, sol_y[:,1], label='Fuel')
plt.title('Propellant mass')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Mass $m$ [kg]')
plt.grid(True)
plt.legend()
plt.figure(2)
plt.plot(sol_t, sol_y[:,2], label='Tank temperature')
plt.title('Temperature')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Temperature $T$ [kg]')
plt.grid(True)
plt.legend()
plt.figure(3)
plt.plot(sol_t, -sol_ydot[:,0], label='N2O')
plt.plot(sol_t, -sol_ydot[:,1], label='Fuel')
plt.title('Propellant flow')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Mass rate $\\dot m$ [kg/s]')
plt.grid(True)
plt.legend()
plt.figure(4)
plt.plot(sol_t, sol_P, label='Tank')
plt.plot(sol_t, sol_Pc, label='Chamber')
plt.plot(sol_t, sol_P-sol_Pc, label='$\\Delta$')
plt.title('Pressure')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Pressure $P$ [Pa]')
plt.grid(True)
plt.legend()
plt.figure(5)
plt.plot(sol_t, sol_Ft, label='Thrust')
plt.plot(sol_t, sol_isp, label='$I_{sp}$')
plt.title('Chamber')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Force $F_t$ [N], velocity $I_{sp}$ [m/s]')
plt.grid(True)
plt.legend()
plt.figure(6)
plt.plot(sol_t, sol_exp, label='Expansion ratio')
plt.title('Nozzle')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Ratio')
plt.grid(True)
plt.legend()



"Test"
### Injector HEM
# P1 = 5e6
# T1 = 290
# P2_arr = np.arange(1e5, 5e6, 1e4)
# dm_arr = np.zeros(len(P2_arr))
# crit_arr = np.zeros(len(P2_arr))
# #dm_arr, dV = injector_ox_hem(T1, P1, 1e5)
# for i in range(1, len(P2_arr), 1):
#     dm_arr[i], dV = injector_ox_hem(T1, P1, P2_arr[i])
#     #print(P2crit)    
# plt.figure(1)
# plt.plot(P2_arr, dm_arr)