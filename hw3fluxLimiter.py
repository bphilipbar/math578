#!/usr/bin/env python2.7

from numpy import *
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# Numerical Parameters
Nx = 50
Nt = 125
tf = 0.316
xf = 1.0
dx = xf/(Nx-1)
dt = tf/(Nt-1)

#Solution Array and Error Array
u = zeros((Nt,Nx+1))  # final x index is not physical but is used for periodic BC
I = zeros(Nt-1)

#initial condition
x = linspace(0,xf,Nx)
u[0,0:Nx] = sin(2*pi*x)+0.5*sin(pi*x)
u[0,Nx] = u[0,1]


#print u[0]
#quit()




# Time Stepping
for n in range(1,Nt):
  for m in range(1,Nx):  # loop over x values

    u2plus = u[n-1,m+1]**2 / 2.0
    u2 = u[n-1,m]**2 / 2.0
    u2minus = u[n-1,m-1]**2 / 2.0

    if u2-u2minus > 0:
      Iminus = m-1
    else:
      Iminus = m+1

    if u2plus-u2 > 0:
      Iplus = m-1
    else:
      Iplus = m+1

    thetaMinus = (u[n-1,Iminus]-u[n-1,Iminus-1]) / (u[n-1,m]-u[n-1,m-1])
    thetaPlus = (u[n-1,Iplus+1]-u[n-1,Iplus]) / (u[n-1,m+1]-u[n-1,m])

    if thetaMinus >= 1:
      phiMinus = 1
    elif thetaMinus > 0:
      phiMinus = thetaMinus
    else:
      phiMinus = 0

    if thetaPlus >= 1:
      phiPlus = 1
    elif thetaPlus > 0:
      phiPlus = thetaPlus
    else:
      phiPlus = 0

    Fminus = 0.5*(u2+u2minus) - 0.5 * sign(u2-u2minus) * (1-phiMinus*(1-abs(u2-u2minus)*dt/(dx))) * (u2-u2minus)
    Fplus = 0.5*(u2plus+u2) - 0.5 * sign(u2plus-u2) * (1-phiPlus*(1-abs(u2plus-u2)*dt/(dx))) * (u2plus-u2)

    u[n,m] = u[n-1,m] - (dt / dx) * (Fplus - Fminus)
  u[n,0] = u[n,Nx-1]
  u[n,Nx] = u[n,1]
  I[n-1] = dx * (sum(u[n]) - sum(u[n-1]))



#Plots

x_arr = linspace(0,xf,Nx)
t_arr = linspace(0,tf,Nt)
X,T = meshgrid(x_arr,t_arr)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, u[:,0:Nx], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')
plt.title('U')

plt.figure(2)
plt.plot(t_arr[0:Nt-1], I)
plt.xlabel('t')
plt.ylabel('$\Delta I$')

ti = int(round(Nt*0.7/2.0))
plt.figure(3)
plt.plot(x_arr, u[ti,0:Nx])
plt.xlabel('x')
plt.ylabel('u')
plt.title('before critical time')

ti = int(round(Nt*1.3/2.0))
plt.figure(4)
plt.plot(x_arr, u[ti,0:Nx])
plt.xlabel('x')
plt.ylabel('u')
plt.title('after critical time')


plt.show()


