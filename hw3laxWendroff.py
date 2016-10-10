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
r = dt/dx

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
    u[n,m] = u[n-1,m] - (r / 2.0) * (u[n-1,m+1]**2.0 - u[n-1,m-1]**2.0)/2.0 + (r**2 / 2.0) * (   ((u[n-1,m+1]**2.0 - u[n-1,m]**2)/2.0)**2.0  -  ((u[n-1,m]**2.0 - u[n-1,m-1]**2)/2.0)**2.0  )
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


