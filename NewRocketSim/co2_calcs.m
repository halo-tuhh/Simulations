mw = 10;
cw = 4920;

mco2 = 1.5;
cp = 800;
qsub = 591e3;
Rco2 = 188.92;

p_vapor = 60e5;

Tw = 273 + 40;
Tco2 = 273 - 78.5;

T1 = (mw*cw*Tw + mco2*cp*Tco2 - qsub*mco2) / (mw*cw + mco2*cp);
T1_Celsius = T1 - 273

Vco2 = mco2*Rco2*T1 / p_vapor;
Vco2_liter = Vco2 * 1000