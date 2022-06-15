from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from alive_progress import alive_bar

def equations(a, X, k, h, Omega_r, Omega_m, Omega_l):
    Phi,Theta0,Theta1,delta,v = X
    H0 = (100)/(300000)*h
    num = H0*np.sqrt(Omega_r + Omega_m*a + Omega_l*(a**4))

    DeltaPhi = 1/(num**2)*(0.5*(H0**2)*(Omega_m*delta + (4*Omega_r*Theta0)/a) - (1/3)*a*(k**2)*Phi) - (Phi/a)
    DeltaTheta0 = -DeltaPhi - ((k*Theta1)/num)
    DeltaTheta1 = (k/(3*num))*(Theta0-Phi)
    Deltadelta = ((k*v)/(num)) - 3*DeltaPhi
    Deltav = (k/(num))*Phi - (v/a)
    return [DeltaPhi,DeltaTheta0,DeltaTheta1,Deltadelta,Deltav]

def inits(k,h,a0,Omega_r,Omega_m,Omega_l):
    H0 = (100)/(300000)*h
    num0 = H0*np.sqrt(Omega_r + Omega_m*a0 + Omega_l*(a0**4))
    return [1,0.5,-k*a0/(6*num0),1.5,k*a0/(2*num0)]

#global parameters
a0 = 1e-7
afin = 10**(-2)
alist = np.geomspace(a0, afin, 1000)

"""
Below is the Phi plot, comment out if not used.
"""


#local parameters
k_array = np.array([0.001, 0.01, 0.1, 1.0])
N_k = len(k_array)
solutions_array = np.zeros((N_k, 5, 1000))
for k_idx,k in enumerate(k_array):
    #LambdaCDM
    Omega_m = 0.32
    Omega_r = 9.4*10**(-5)
    Omega_l = 0.68
    aeq = Omega_r/Omega_m
    h = 0.7
    H0 = (100)/(300000)*h
    keq = np.sqrt(2 * Omega_m / aeq) * H0
    h = 0.7

    initialvalues = inits(k,h,a0,Omega_r,Omega_m,Omega_l)
    sol = solve_ivp(equations, [a0, afin], initialvalues, args=(k,h, Omega_r, Omega_m, Omega_l), dense_output=True)
    z = sol.sol(alist)
    solutions_array[k_idx] = z

for idx in range(N_k):
  plt.plot(alist, solutions_array[idx, 0], label=f"k = {k_array[idx]}" + r" Mpc$^{-1}$") #solutions_array[idx, 0]: 0 as Phi is first
plt.axvline(aeq,c='black',label="a$_{eq}$")
plt.legend()
plt.xscale('log')
plt.xlabel("a")
plt.ylabel("Φ")
plt.show()


"""
Transfer function
"""
#LambdaCDM
TransferLambdaCDM = []
k_array = np.geomspace(0.001,1,10)
N_k = len(k_array)
solutions_array = np.zeros((N_k, 5, 1000))

with alive_bar(N_k,title='Calculating ΛCDM',length=20,bar='filling',spinner='dots_waves2') as bar: #fun progress bar, of course not needed for the program to function
    for k_idx,k in enumerate(k_array):
        Omega_m = 0.32
        Omega_r = 9.4*10**(-5)
        Omega_l = 0.68
        aeq = Omega_r/Omega_m
        h = 0.7
        H0 = (100)/(300000)*h
        keq = np.sqrt(2 * Omega_m / aeq) * H0

        initialvalues = inits(k,h,a0,Omega_r,Omega_m,Omega_l)
        sol = solve_ivp(equations, [a0, afin], initialvalues, args=(k,h, Omega_r, Omega_m, Omega_l), dense_output=True)
        z = sol.sol(alist)
        solutions_array[k_idx] = z
        bar()

for idx in range(N_k):
    TransferLambdaCDM.append(solutions_array[idx, 0][-1]*(10/9))

plt.plot(k_array/keq,TransferLambdaCDM,label='ΛCDM')
plt.xscale('log')
plt.yscale('log')

#sCDM
TransfersCDM = []
solutions_array = np.zeros((N_k, 5, 1000))

with alive_bar(N_k,title='Calculating sCDM',length=20,bar='filling',spinner='dots_waves2') as bar: #fun progress bar, of course not needed for the program to function
    for k_idx,k in enumerate(k_array):

        h = 0.5
        omega_r = 9.4e-5 / 0.32
        omega_m = 1.0
        omega_l = 0.0
        aeq = Omega_r/Omega_m
        H0 = (100)/(300000)*h
        keq = np.sqrt(2 * Omega_m / aeq) * H0

        initialvalues = inits(k,h,a0,Omega_r,Omega_m,Omega_l)
        sol = solve_ivp(equations, [a0, afin], initialvalues, args=(k,h, Omega_r, Omega_m, Omega_l), dense_output=True)
        z = sol.sol(alist)
        solutions_array[k_idx] = z
        bar()

for idx in range(N_k):
    TransfersCDM.append(solutions_array[idx, 0][-1]*(10/9))

plt.plot(k_array/keq,TransfersCDM,label="sCDM")
plt.xscale('log')
plt.yscale('log')

x = k_array/keq
T = np.log(1+0.171*x)/(0.171*x)*(1+0.284*x+(1.18*x)**2 + (0.399*x)**3 + (0.490*x)**4)**(-0.25)
plt.plot(x,T,'--',label="BBKS")
plt.legend()
plt.xlabel("x")
plt.ylabel("T")
plt.show()
