#!/usr/bin/env python2.7

from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


# Numerical Parameters
Nx = 11
Nt = 39
tf = 0.2
xf = 1.0
dx = xf/(Nx-1)
dt = tf/(Nt-1)
r = dt/(dx**2)


#Solution Array
# We are not saving the boundary values, hence Nx-2 values
u = zeros((Nt,Nx-2))

#initial condition
x = linspace(dx,xf-dx,Nx-2)
u[0,:] = sin(2 * pi * x)

A = -r*diag(ones(Nx-3,),-1) +(2+2*r)*diag(ones(Nx-2,)) -r*diag(ones(Nx-3,),1)
mat = r*diag(ones(Nx-3,),-1) +(2-2*r)*diag(ones(Nx-2,)) +r*diag(ones(Nx-3,),1)
vec = zeros(Nx-2)



# Time Step
for n in range(1,Nt):
    vec[0] = r*1.0*sin(40.0*dt*n) + r*1.0*sin(40.0*dt*(n-1)) #If you look at the math, the boundary conditions at BOTH the current and previous time steps appear. Due to dt being small, the results do not seem to noticeably to depend on this.
    b = vec + mat.dot(u[n-1,:])
    u[n,:] = linalg.solve(A,b)



#Plots
x_arr = linspace(0,xf,Nx)
t_arr = linspace(0,tf,Nt)
u_arr = zeros((Nt,Nx))
u_arr[:,1:Nx-1] = u  #note that final value when indexing (Nx-1) does not occur
u_arr[:,0] = 1.0*sin(40.0*t_arr)



# change x_arr and u_arr to x and u as desired (depending on if BC should be plotted)
X,T = meshgrid(x_arr,t_arr)



fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, u_arr, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')




plt.show()
