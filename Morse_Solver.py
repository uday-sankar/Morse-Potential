import numpy as np
import matplotlib.pyplot as plt
import Support_functions_Morse as EF
def Morse_V(r):
	V=D_e*(1-2.718**(-a*(r-r_e)))**2
	return V
m,h = 1,1
Xmin = 0.5
Xmax = 6
D_e = 10
a = 0.8
r_e = 1
R = np.linspace(Xmin, Xmax, 10**3)
for p in range(len(R)):
	if Morse_V(R[p]) < D_e:
		Xmin = R[p] - 1
		Pos = p
		break
R=np.linspace(Xmin, Xmax, 10**4)
EF.Constant_feeder(h, m, D_e, Morse_V)

Eigen_E=EF.Eigen_Range_finder(R, 0, D_e*.9, 10, 0.1)
fig=plt.figure()
ax=plt.axes(xlabel="r", ylabel="Psi", ylim=(0, Eigen_E[-1]*1.1))
R=np.linspace(Xmin, Xmax, 10**5)
EF.Plot_Eq(R, Eigen_E, ax)
EF.Analatic_mult(R, np.arange(len(Eigen_E)), D_e, a, r_e, ax)
ax.plot(R, Morse_V(R))
plt.legend()
plt.show()
