from scipy.special import eval_genlaguerre as lag
from scipy.special import gamma as gm
import numpy as np
from math import exp
from math import factorial as fac
import matplotlib.pyplot as plt
De=10
a=.8
Re=1

e=exp(1)
R=np.linspace(-1,5,100)

X=R*a
Xe=Re*a
m,h=1,1

lam=((2*m*De)**.5)/(a*h)
n=5#n<=[lamda-1/2]
eps=-(lam-n-.5)**2
alp=(2*lam-2*n-1)#alpha

En=eps*(a*h)**2/(2*m)+10
Z=2*lam*e**(-(X-Xe))
Nn=(fac(n)*(alp)/(gm(2*lam-n)))**.5

L=lag(n,alp,Z)#lagragian

Psi=Nn*Z**(alp/2)*e**(-.5*Z)*L

plt.plot(X,Psi)
plt.show()