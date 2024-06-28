import math
import csv

H = 0.1
Aout = math.pi*(3.75e-3)**2
uout = 0.0
doutdt = 0.0
pw = 997.0
mtot = 0.5 #water added later
v = 0.0
a = 0.0
t = 0.0
delta_t = 1e-4
g = 9.81

PU = 8e+5
VU = 2.1e-3

def A(z):
    if(z>0.1):
        return 0.01 * math.pi
    return ((0.1-3.75e-3)/0.1*z+3.75e-3)**2 * math.pi

def B(h):
    delta_z = h/1000.0
    res = 0.0
    for i in range(1000):
        res += Aout/A((i+0.5)*delta_z) * delta_z
    return res

def C(h):
    return 0.5 * ((Aout/A(h))**2 - 1.0)

def volhelper(h):
    delta_z = h/1000.0
    res = 0.0
    for i in range(1000):
        res += A((i+0.5)*delta_z) * delta_z
    return res

mtot += volhelper(H)*pw
print(mtot)
Fthrmax = 0.0
vmax = 0.0
amax = 0.0
VH = 1e-6
PH = 0 #Pa relative
Q = 0.0
kv = 0.16
pn = 1.2922

i = 0

with open('SimulationTools/WaterRocketSim/csvFiles/water_rocket_simulation.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Time (s)", "Fthr (N)", "Fint (N)", "mtot (kg)", "uout (m/s)", "H (m)", "a (m/s^2)", "v (m/s)", "Q (m^3/s)", "PU (Pa)", "PH (Pa)"])
    
    while (H>= 0.0 and i <= 20):
        doutdt = -(C(H)* uout**2 + PH/pw + (a+g)*H)/B(H)
        uout += doutdt * delta_t
        H += (Aout*uout*delta_t)/A(H)
        print(doutdt,uout)
        if(H < 0):
            break
        Fthr = pw * Aout * uout**2
        Fint = -pw*Aout*(H*doutdt+Aout/A(H) * uout**2)
        Fdrag = 0.0
        a = (Fthr+Fint+Fdrag)/mtot - g
        v += a* delta_t

        #ToDo PH Update
        if(PU < PH/2.0):
            Q = 257*kv*PU*1e-5*1/math.sqrt(pn*293.15)
        else:
            Q = 514*kv*math.sqrt(((PU-PH)*1e-5*PU*1e-5)/(pn*293.15))
        Q /= 3600.0
        PH = ((PH*VH*1e-5+Q*delta_t)/(VH-Aout*uout*delta_t))*1e+5
        VH -= Aout*uout*delta_t
        PU = ((PU*VU*1e-5-Q*delta_t)/(VU))*1e+5
        #print(Q,PH,VH,PU)
        
        mtot += pw*uout*Aout*delta_t
        t += delta_t
        #print("%.4f %.2f %.1f %.4f %.1f %.4f %.1f %.4f %.1f %.1f %.1f"%(t,Fthr,Fint,mtot,uout,H,a,v,Q,PU*1e-5,PH*1e-5))
        
        # Write the data to the CSV file
        writer.writerow([f"{t:.4f}", f"{Fthr:.4f}", f"{Fint:.4f}", f"{mtot:.4f}", f"{uout:.4f}", f"{H:.4f}", f"{a:.4f}", f"{v:.4f}", f"{Q:.4f}", f"{PU:.4f}", f"{PH:.4f}"])

        if(v > vmax):
            vmax = v
        if(a > amax):
            amax = a
        if(Fthr > Fthrmax):
            Fthrmax = Fthr
        i+=1

print(vmax,amax,Fthrmax)