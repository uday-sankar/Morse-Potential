import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.constants import h, c
from morse import Morse, FAC
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

COLOUR1 = (0.6196, 0.0039, 0.2588, 1.0)

# Atom masses and equilibrium bond length for (1H)(35Cl).
mA, mB = 1., 35.
X_re = 1.27455e-10
X_Te = 0
X_we, X_wexe = 2990.945, 52.818595

X = Morse(mA, mB, X_we, X_wexe, X_re, X_Te)
X.make_rgrid()
X.V = X.Vmorse(X.r)

fig, ax = plt.subplots()
X.plot_V(ax, color='k')

X.draw_Elines(range(X.vmax), ax)
X.draw_Elines(X.get_vmax(), ax, linestyles='--', linewidths=1)
X.plot_psi([2, 14], ax, scaling=2, color=COLOUR1)
X.label_levels([2, 14], ax)

ax.set_xlabel(r'$r\;/\mathrm{\\A}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('morse-psi.png')
plt.show()
