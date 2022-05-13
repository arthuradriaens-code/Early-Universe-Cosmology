from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

#global parameters:
Omega_m = 0.32
Omega_r = 9.4*10**(-5)
Omega_l = 0.7
h = 0.7

def num(a):
    return np.sqrt(Omega_r*(a**(-4)) + Omega_m*(a**(-3)) + Omega_l)

def DeltaPhi(delta,Theta0,a,delta_a,Phi,x):
    return (0.5*(Omega_m*delta + 4*Omega_r*Theta0)/(a*num(a)**2) - (x**2)*Phi/(3*a**3*(num(a)**2)) - Phi/a)*delta_a

def DeltaTheta0(Theta1,deltaPhi,a,delta_a,x):
    return (x*Theta1/(a**2*num(a)))*delta_a - deltaPhi

def DeltaTheta1(Theta0,Phi,a,delta_a,x):
    return (x*(Theta0-Phi)/(3*a**2*num(a)))*delta_a

def Deltadelta(DeltaPhi,v,delta_a,a,x):
    return -3*DeltaPhi - (x*v*delta_a)/(a**2*num(a)) #notice i transformed iv\rightarrow v

def Deltav(Phi,a,v,delta_a,x):
    return (-(x*Phi)/(a**2*num(a)) - v/a)*delta_a #same here


#variable parameters
N= 10000
alist = np.logspace(-9,-7,N)
Transfer = []

for x in np.linspace(1,5000000,20):
    Theta0list = np.zeros(N)
    Theta1list = np.zeros(N)
    Philist = np.zeros(N)
    vlist = np.zeros(N)
    deltalist = np.zeros(N)

    #initial conditions
    vlist[0] = -x/(2*10**(-7)*num(10**(-7)))
    Philist[0] = 1.0
    Theta0list[0] = 0.5
    deltalist[0] = 1.5
    Theta1list[0] = vlist[0]/3
    for i in range(len(alist)-1):
        Delta_a = alist[i+1]-alist[i]
        DeltaPhi_ = DeltaPhi(deltalist[i],Theta0list[i],alist[i],Delta_a,Philist[i],x)
        DeltaTheta0_ = DeltaTheta0(Theta1list[i],DeltaPhi_,alist[i],Delta_a,x)
        DeltaTheta1_ = DeltaTheta1(Theta0list[i],Philist[i],alist[i],Delta_a,x)
        Deltadelta_ = Deltadelta(DeltaPhi_,vlist[i],Delta_a,alist[i],x)
        Deltav_ = Deltav(Philist[i],alist[i],vlist[i],Delta_a,x)
        
        vlist[i+1] = vlist[i] + Deltav_
        Philist[i+1] = Philist[i] + DeltaPhi_
        Theta0list[i+1] = Theta0list[i] + DeltaTheta0_
        deltalist[i+1] = deltalist[i] + Deltadelta_
        Theta1list[i+1] = Theta1list[i] + DeltaTheta1_

    plt.plot(alist,Philist,label="y={}".format(x*0.00456/h))
    plt.xscale('log')
    print(x)
    Transfer.append(Philist[5000]*10)
plt.legend()
plt.xlabel("a")
plt.ylabel("Φ")
plt.show()

"""
This is for if we get the transfer function
"""
x = np.logspace(-5,2,10000)

T = np.log(1+0.171*x)/(0.171*x)*(1+0.284*x+(1.18*x)**2 + (0.399*x)**3 + (0.490*x)**4)**(-0.25)

plt.plot(x,T,'--',label="BBKS")
plt.plot(np.linspace(1,10000000,20)*0.00000456/h,Transfer,label="ΛCDM")

Omega_l = 0
h = 0.5

def num(a):
    return np.sqrt(Omega_r*(a**(-4)) + Omega_m*(a**(-3)) + Omega_l)


#variable parameters
N= 10000
alist = np.logspace(-9,-7,N)
Transfer = []

for x in np.linspace(1,5000000,20):
    Theta0list = np.zeros(N)
    Theta1list = np.zeros(N)
    Philist = np.zeros(N)
    vlist = np.zeros(N)
    deltalist = np.zeros(N)

    #initial conditions
    vlist[0] = -x/(2*10**(-7)*num(10**(-7)))
    Philist[0] = 1.0
    Theta0list[0] = 0.5
    deltalist[0] = 1.5
    Theta1list[0] = vlist[0]/3
    for i in range(len(alist)-1):
        Delta_a = alist[i+1]-alist[i]
        DeltaPhi_ = DeltaPhi(deltalist[i],Theta0list[i],alist[i],Delta_a,Philist[i],x)
        DeltaTheta0_ = DeltaTheta0(Theta1list[i],DeltaPhi_,alist[i],Delta_a,x)
        DeltaTheta1_ = DeltaTheta1(Theta0list[i],Philist[i],alist[i],Delta_a,x)
        Deltadelta_ = Deltadelta(DeltaPhi_,vlist[i],Delta_a,alist[i],x)
        Deltav_ = Deltav(Philist[i],alist[i],vlist[i],Delta_a,x)
        
        vlist[i+1] = vlist[i] + Deltav_
        Philist[i+1] = Philist[i] + DeltaPhi_
        Theta0list[i+1] = Theta0list[i] + DeltaTheta0_
        deltalist[i+1] = deltalist[i] + Deltadelta_
        Theta1list[i+1] = Theta1list[i] + DeltaTheta1_
    print(x)
    Transfer.append(Philist[5000]*10)

plt.plot(np.linspace(1,10000000,20)*0.00000456/h,Transfer,label="sCDM")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel("y")
plt.ylabel("T")
plt.show()