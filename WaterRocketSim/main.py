from tkinter import *
from tkinter.ttk import *
from tkinter.font import *

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import math
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

def A(z):
    global rtank, rnozzle, ltaper, height_water_tank
    #upper taper: TO DO

    #straight part
    if(z>ltaper):
        return rtank**2 * math.pi
    #lower taper
    return ((rtank-rnozzle)/ltaper*z+rnozzle)**2 * math.pi

def B(h):
    global Aout
    delta_z = h/1000.0
    res = 0.0
    for i in range(1000):
        res += Aout/A((i+0.5)*delta_z) * delta_z
    return res

def C(h):
    global Aout
    return 0.5 * ((Aout/A(h))**2 - 1.0)

def volhelper(vol):
    global rtank, rnozzle, ltaper, height_water_tank
    height_water_tank = 0.0
    delta_z = ltaper/1000.0
    res = 0.0
    #lower taper
    for i in range(1000):
        res += A((i+0.5)*delta_z) * delta_z
        if(res >= vol):
            return (i+1)*delta_z
    #upper taper: TO DO

    vol -= res
    height_water_tank = vol/(math.pi*rtank**2) + ltaper
    return height_water_tank

def calculate():
    global EntryCntValve, EntryGravity, EntryMassW, EntryMassAdd, EntryMassTank, EntryPressure, EntryNozEff, EntryRadNozzle, EntryRhoWater, EntryVolAir, EntryRadTank, EntryLenTaper, EntryDeltaT, EntryKvValve, EntryCelsius, selectVar, rtank, rnozzle, ltaper, Aout, checkVar
    reset()
    #input
    rtank = float(EntryRadTank.get())*1e-3 #m
    rnozzle = float(EntryRadNozzle.get())*1e-3 #m
    nozeff = float(EntryNozEff.get()) #1
    Aout = math.pi*(rnozzle)**2
    ltaper = float(EntryLenTaper.get())*1e-3 #m
    rho_water = float(EntryRhoWater.get()) #kg/m^3
    rho_air = float(EntryRhoAir.get()) #kg/m^3
    mass_water = float(EntryMassW.get()) #kg
    H = volhelper(mass_water/rho_water) #m
    mass_dry = float(EntryMassTank.get())+float(EntryMassAdd.get()) #kg
    mass_all = mass_dry + mass_water #kg
    delta_t = float(EntryDeltaT.get()) #s
    g = float(EntryGravity.get()) #m/s^2
    pres_upper = float(EntryPressure.get())*1e+5 #Pa
    vol_upper = float(EntryVolAir.get())*1e-3 #m^3
    vol_lower = 1e-5 #m^3
    pres_lower = 0.0 #Pa
    kv_vav = float(EntryKvValve.get()) #m^3/h
    cnt_vav = int(EntryCntValve.get()) #1
    temperature = float(EntryCelsius.get())+273.15 #K
    static = bool(checkVar.get())

    t = 0.0 #s
    a = -g #m/s^2
    v = 0.0 #m/s
    s = 0.0 #m

    uout = 0.0 #m/s
    duoutdt = 0.0 #m/s^2

    print(mass_water,mass_all,H*1e3)

    while (H>= 0.0):
        duoutdt = -(C(H)* uout**2 + pres_lower/rho_water + (a+g)*H)/B(H)
        uout += duoutdt * delta_t
        H += (Aout*uout*delta_t)/A(H)
        if(H < 0.0):
            break
        Fthr = rho_water * Aout * (nozeff * uout)**2 #N
        Fint = -rho_water*Aout*(H*duoutdt+Aout/A(H) * uout**2) #N
        Fdrag = 0.0 #N
        if(not static):
            a = (Fthr+Fint+Fdrag)/mass_all - g
            v += a * delta_t
            s += v * delta_t
        Fges = Fthr + Fint + Fdrag - mass_all * g
        #*1e-5 for conversion Pa -> bar and *1e+5 bar -> Pa
        if(pres_upper < pres_lower/2.0):
            Q = 257*kv_vav*pres_upper*1e-5*1/math.sqrt(rho_air*temperature)
        else:
            Q = 514*kv_vav*math.sqrt(((pres_upper-pres_lower)*1e-5*pres_upper*1e-5)/(rho_air*temperature))
        Q /= 3600.0 #Nm^3/h -> Nm^3/
        Q *= cnt_vav #multiple valves feature
        pres_lower = ((pres_lower*1e-5*(vol_lower)**1.4)/((vol_lower-Aout*uout*delta_t)**1.4))*1e+5
        vol_lower -= Aout*uout*delta_t #-= because uout is negative
        pres_lower = ((pres_lower*1e-5*(vol_lower)**1+Q*delta_t)/(vol_lower)**1)*1e+5
        pres_upper = ((pres_upper*1e-5*(vol_upper)**1-Q*delta_t)/(vol_upper)**1)*1e+5
        
        mass_all += rho_water*uout*Aout*delta_t #+= because uout is negative
        t = t + delta_t

        add(t,Fthr,Fint,Fdrag,Fges,mass_all,pres_upper*1e-5,uout,Q*1e3,pres_lower*1e-5,vol_lower*1e3)
        addRocket(t,a,v,s)

    a = -g
    while(s>= 0.0 and not static):
        v += a * delta_t
        s += v * delta_t
        t += delta_t
        addRocket(t,a,v,s)
        
    changePlot(selectVar.get())
    #print("Done")

def add(t,fthrust,fint,fdrag,fges,mass,pressure,uout,Q_vav,presairwater,volairwater):
    global times, fthrusts, fints, fdrags, fgess, masses, pressures, uouts, volflows, pressuresairwater, volsairwater
    times.append(t)
    fthrusts.append(fthrust)
    fints.append(fint)
    fdrags.append(fdrag)
    fgess.append(-fges) # for better comparison to recorded data
    masses.append(mass)
    pressures.append(pressure)
    uouts.append(uout)
    volflows.append(Q_vav)
    volsairwater.append(volairwater)
    pressuresairwater.append(presairwater)

def addRocket(t,a,v,s):
    global timesr, a_s, v_s, s_s
    timesr.append(t)
    a_s.append(a)
    v_s.append(v)
    s_s.append(s)

def reset():
    global times, fthrusts, fints, fdrags, fgess, masses, pressures, uouts, volflows, pressuresairwater, timesr, a_s, v_s, s_s, volsairwater
    times = []
    timesr = []
    fthrusts = []
    fints = []
    fdrags = []
    fgess = []
    masses = []
    pressures = []
    volsairwater = []
    uouts = []
    volflows = []
    pressuresairwater = []
    a_s = []
    v_s = []
    s_s = []

def changePlot(att):
    global plot, figure, LabelUnitSelect, times, fthrusts, fints, fdrags, fgess, masses, pressures, uouts, volflows, pressuresairwater, timesr, a_s, v_s, s_s
    plot.clear()
    if(att == 'Thrust'):
        plot.plot(times, fthrusts , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [N]'
    elif(att == 'Drag'):
        plot.plot(times, fdrags , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [N]'
    elif(att == 'Int. Force'):
        plot.plot(times, fints , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [N]'
    elif(att == 'Force'):
        plot.plot(times, fgess , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [N]'
    elif(att == 'Pressure Upper'):
        plot.plot(times, pressures , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [bar]'
    elif(att == 'Pressure Lower'):
        plot.plot(times, pressuresairwater , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [bar]'
    elif(att == 'Exit Velocity'):
        plot.plot(times, uouts , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [m/s]'
    elif(att == 'Mass'):
        plot.plot(times, masses , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [kg]'
    elif(att == 'Volumeflow Valve'):
        plot.plot(times, volflows , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [Nl/s]'
    elif(att == 'Rocket-acc'):
        plot.plot(timesr, a_s , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [m/s^2]'
    elif(att == 'Rocket-vel'):
        plot.plot(timesr, v_s , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [m/s]'
    elif(att == 'Rocket-pos'):
        plot.plot(timesr, s_s , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [m]'
    elif(att == 'Vol Air Lower'):
        plot.plot(times, volsairwater , color="blue", linestyle="solid")
        LabelUnitSelect['text'] = 'Unit: [l]'
    figure.canvas.draw()

def close():
    global root
    root.destroy()

def setxlimit():
    global EntryLimitPlot,plot,figure
    limits = EntryLimitPlot.get().split(':')
    if(len(limits) < 2):
        return
    plot.set_xlim(float(limits[0]),float(limits[1]))
    figure.canvas.draw()

def setylimit():
    global EntryLimitPlot,plot,figure
    limits = EntryLimitPlot.get().split(':')
    if(len(limits) < 2):
        return
    plot.set_ylim(float(limits[0]),float(limits[1]))
    figure.canvas.draw()

root = Tk()
root.title("WaterRocketSim")

default_font = tkinter.font.nametofont("TkDefaultFont")
default_font.configure(size=12)
root.option_add("*Font", default_font)

#General
Label(master=root,text="General:").grid(row=0,column=0,columnspan=2,padx=5,pady=5)

#Air Density
Label(master=root,text="Density Air [kg/m^3]:").grid(row=1,column=0,sticky=E,padx=5,pady=5)
EntryRhoAir = Entry(master=root)
EntryRhoAir.insert(0,"1.2922")
EntryRhoAir.grid(row=1,column=1,sticky=E,padx=5,pady=5)

#Water Density
Label(master=root,text="Density Water [kg/m^3]:").grid(row=2,column=0,sticky=E,padx=5,pady=5)
EntryRhoWater = Entry(master=root)
EntryRhoWater.insert(0,"997")
EntryRhoWater.grid(row=2,column=1,sticky=E,padx=5,pady=5)

#Gravity
Label(master=root,text="Gravity [m/s^2]:").grid(row=3,column=0,sticky=E,padx=5,pady=5)
EntryGravity = Entry(master=root)
EntryGravity.insert(0,"9.81")
EntryGravity.grid(row=3,column=1,sticky=E,padx=5,pady=5)

#Temperatur
Label(master=root,text="Temperatur [°C]:").grid(row=4,column=0,sticky=E,padx=5,pady=5)
EntryCelsius = Entry(master=root)
EntryCelsius.insert(0,"20.0")
EntryCelsius.grid(row=4,column=1,sticky=E,padx=5,pady=5)

#Delta Time
Label(master=root,text="Time Delta [s]:").grid(row=5,column=0,sticky=E,padx=5,pady=5)
EntryDeltaT = Entry(master=root)
EntryDeltaT.insert(0,"0.0001")
EntryDeltaT.grid(row=5,column=1,sticky=E,padx=5,pady=5)

#Mass
Label(master=root,text="Mass:").grid(row=6,column=0,columnspan=2,padx=5,pady=5)

#Tank Mass
Label(master=root,text="Mass Tank [kg]:").grid(row=7,column=0,sticky=E,padx=5,pady=5)
EntryMassTank = Entry(master=root)
EntryMassTank.insert(0,"1.0")
EntryMassTank.grid(row=7,column=1,sticky=E,padx=5,pady=5)

#Add Mass
Label(master=root,text="Mass Add. [kg]:").grid(row=8,column=0,sticky=E,padx=5,pady=5)
EntryMassAdd = Entry(master=root)
EntryMassAdd.insert(0,"2.5")
EntryMassAdd.grid(row=8,column=1,sticky=E,padx=5,pady=5)

#Radius Tank
Label(master=root,text="Radius Tank [mm]:").grid(row=9,column=0,sticky=E,padx=5,pady=5)
EntryRadTank = Entry(master=root)
EntryRadTank.insert(0,"34")
EntryRadTank.grid(row=9,column=1,sticky=E,padx=5,pady=5)

#Checkbox teststand
checkVar = IntVar()
CheckButtonTest = Checkbutton(master=root,text="Static Fire",variable=checkVar,onvalue=1,offvalue=0)
CheckButtonTest.grid(row=10,column=0,columnspan=2,padx=5,pady=5)

#Button Calculate
buttonGo = Button(master=root, text='Calculate!', command=calculate)
buttonGo.grid(row=11,column=0,columnspan=4,padx=5,pady=5)

#Row Spaceing
root.grid_rowconfigure(12, minsize=20)

#OptionMenu Settings
selectVar = StringVar(root)
choices = {'Thrust','Mass','Drag','Pressure Upper','Pressure Lower','Exit Velocity','Int. Force','Force','Volumeflow Valve','Rocket-acc','Rocket-vel','Rocket-pos','Vol Air Lower'}
selectCell = OptionMenu(root,selectVar,'Thrust',*sorted(choices),command=changePlot)
selectCell.grid(row=11,column=5,padx=5,pady=5)

#Unit viewer
LabelUnitSelect = Label(master=root,text="Unit: [N]")
LabelUnitSelect.grid(row=11,column=6,padx=5,pady=5)

#Plot Scaling
EntryLimitPlot = Entry(master=root,width="10")
EntryLimitPlot.insert(0,"-10:10")
EntryLimitPlot.grid(row=11,column=7,sticky=E,padx=5,pady=5)

buttonXLimit = Button(master=root, text='X Lim!', command=setxlimit)
buttonXLimit.grid(row=11,column=8,padx=5,pady=5)

buttonYLimit = Button(master=root, text='Y Lim!', command=setylimit)
buttonYLimit.grid(row=11,column=9,padx=5,pady=5)

#Upper Tank
Label(master=root,text="Upper Tank:").grid(row=0,column=2,columnspan=2,padx=5,pady=5)

#Upper Tank Pressure
Label(master=root,text="Pressure [bar]:").grid(row=1,column=2,sticky=E,padx=5,pady=5)
EntryPressure = Entry(master=root)
EntryPressure.insert(0,"8")
EntryPressure.grid(row=1,column=3,sticky=E,padx=5,pady=5)

#Upper Tank Volume
Label(master=root,text="Volume [l]:").grid(row=2,column=2,sticky=E,padx=5,pady=5)
EntryVolAir = Entry(master=root)
EntryVolAir.insert(0,"2.5")
EntryVolAir.grid(row=2,column=3,sticky=E,padx=5,pady=5)

#Lower Tank
Label(master=root,text="Lower Tank:").grid(row=3,column=2,columnspan=2,padx=5,pady=5)

#Water Mass
Label(master=root,text="Mass Water [kg]:").grid(row=4,column=2,sticky=E,padx=5,pady=5)
EntryMassW = Entry(master=root)
EntryMassW.insert(0,"2.0")
EntryMassW.grid(row=4,column=3,sticky=E,padx=5,pady=5)

#Taperlength Lower Tank
Label(master=root,text="Taperlength [mm]:").grid(row=5,column=2,sticky=E,padx=5,pady=5)
EntryLenTaper = Entry(master=root)
EntryLenTaper.insert(0,"50")
EntryLenTaper.grid(row=5,column=3,sticky=E,padx=5,pady=5)

#Nozzle + Valve
Label(master=root,text="Nozzle + Valve:").grid(row=6,column=2,columnspan=2,padx=5,pady=5)

#Nozzle Radius
Label(master=root,text="Nozzle Radius [mm]:").grid(row=7,column=2,sticky=E,padx=5,pady=5)
EntryRadNozzle = Entry(master=root)
EntryRadNozzle.insert(0,"3.75")
EntryRadNozzle.grid(row=7,column=3,sticky=E,padx=5,pady=5)

#Nozzle Effiency
Label(master=root,text="Nozzle Thrust Eff. [1]:").grid(row=8,column=2,sticky=E,padx=5,pady=5)
EntryNozEff = Entry(master=root)
EntryNozEff.insert(0,"0.85")
EntryNozEff.grid(row=8,column=3,sticky=E,padx=5,pady=5)

#Valve KV
Label(master=root,text="Valve KV [m^3/h]:").grid(row=9,column=2,sticky=E,padx=5,pady=5)
EntryKvValve = Entry(master=root)
EntryKvValve.insert(0,"0.31")
EntryKvValve.grid(row=9,column=3,sticky=E,padx=5,pady=5)

#Valve Count
Label(master=root,text="Valve Count [1]:").grid(row=10,column=2,sticky=E,padx=5,pady=5)
EntryCntValve = Entry(master=root)
EntryCntValve.insert(0,"1")
EntryCntValve.grid(row=10,column=3,sticky=E,padx=5,pady=5)

#Column Spaceing
root.grid_columnconfigure(4, minsize=50)

#Plot
figure = Figure(figsize=(5, 4), dpi=200)
plot = figure.add_subplot(1, 1, 1)

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().grid(row=0,column=5,rowspan=11,columnspan=5,padx=5,pady=5)

#Vars Plot
times = []
timesr = []
fthrusts = []
fints = []
fdrags = []
fgess = []
masses = []
volsairwater = []
pressures = []
uouts = []
volflows = []
pressuresairwater = []
a_s = []
v_s = []
s_s = []

#global vars
H = 0.0
rnozzle = 0.0
rtank = 0.0
ltaper = 0.0

#Start Plot
root.protocol("WM_DELETE_WINDOW", close)
root.mainloop()
