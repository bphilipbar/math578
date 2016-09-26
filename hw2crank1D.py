#!/usr/bin/env python2.7

from numpy import *
from scipy.optimize import newton_krylov   #our fancy non-linear system solver
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# Numerical Parameters
Nx = 50
Nt = 51
tf = 0.2
xf = 1.0
dx = xf/(Nx-1)
dt = tf/(Nt-1)
r = dt/(dx**2)

left = 0.0
right = 0.0

#Solution Array
u = zeros((Nt,Nx))

#initial condition
for i in range(Nx):
   x = i*dx
   if x >= 0.5 and x<= 0.7:
      u[0,i] = 1.0
   else:
      u[0,i] = 0.0


#u[0,Nx-1]=3
#print u[0,1:2]
#print u[0,0:Nx]
#print u[0,Nx-1]
#quit()


def residual(input):
  return hstack((input[0]-left,input[1:Nx-1] - r/2 * (1.0/2 + input[1:Nx-1]**2) * (input[0:Nx-2]+input[2:Nx]-2*input[1:Nx-1]) - RHS,input[Nx-1]-right))

# Time Stepping
for n in range(1,Nt):
   RHS = u[n-1,1:Nx-1] + r/2 * (1.0/2 + u[n-1,1:Nx-1]**2) * (u[n-1,0:Nx-2]+u[n-1,2:Nx]-2*u[n-1,1:Nx-1])
   u[n] = newton_krylov(residual,zeros(Nx))

## for Nx=10
#print u[:,4]
#print u[:,5]
#print u[:,6]
#print u[:,7]

#Plots
x_arr = linspace(0,xf,Nx)
t_arr = linspace(0,tf,Nt)
X,T = meshgrid(x_arr,t_arr)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, u, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')
plt.show()


