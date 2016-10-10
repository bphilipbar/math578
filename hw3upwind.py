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

left = 0.0
right = 0.0

#Solution Array
u = zeros((Nt,Nx))
I = zeros(Nt-1) #added for part b)

#initial condition
x = linspace(0,xf,Nx)
u[0] = sin(2*pi*x)+0.5*sin(pi*x)

#print u[0]
#quit()

# Time Stepping
for n in range(1,Nt):
  a=0
  b=0
  for m in range(0,Nx):
    if u[n-1,m] > 0:
      if m == 0: # if we are at the left end, flag and skip for now until u[n,Nx-1] is defined
        a=1
        continue
      u[n,m] = u[n-1,m] - r * u[n-1,m]*(u[n-1,m] - u[n-1,m-1])
    else:
      if m == Nx-1:
        b=1
        continue
      u[n,m] = u[n-1,m] - r * u[n-1,m]*(u[n-1,m+1] - u[n-1,m])
  if a == 1: # we are now setting the periodic boundary condition
    u[n,0] = u[n,Nx-1]
  else:
    u[n,Nx-1] = u[n,0]
  I[n-1] = dx * (sum(u[n]) - sum(u[n-1]))



#Plots

x_arr = linspace(0,xf,Nx)
t_arr = linspace(0,tf,Nt)
X,T = meshgrid(x_arr,t_arr)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, u, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')
plt.title('$Delta I$')

fig = plt.figure(2) #error
plt.plot(t_arr[0:Nt-1],I)
plt.xlabel('t')
plt.ylabel('$\Delta I$')

ti = int(round(Nt*.7/2.0))
fig = plt.figure(3)
plt.plot(x_arr,u[ti])
plt.xlabel('u')
plt.ylabel('before critical time')

ti = int(round(Nt*1.3/2.0))
fig = plt.figure(4)
plt.plot(x_arr,u[ti])
plt.xlabel('u')
plt.ylabel('after critical time')

plt.show()


