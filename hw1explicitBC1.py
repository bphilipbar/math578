#!/usr/bin/env python2.7

from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


# Numerical Parameters
Nx = 10  # should be even number else our actual solution divides by 0
Nt = 4
tf = 0.2
xf = 1.0
dx = xf/(Nx-1)
dt = tf/(Nt-1)
r = dt/(dx**2)

left = 0.0
right = 0.0

#Solution Array
# We are not saving the boundary values, hence Nx-2 values
u = zeros((Nt,Nx-2))
uu = zeros((Nt,Nx-2))

#initial condition
x = linspace(dx,xf-dx,Nx-2)
u[0,:] = sin(2 * pi * x)
uu[0,:] = sin(2 * pi * x)



A = diag(ones(Nx-3,),-1) -2*diag(ones(Nx-2,))  + diag(ones(Nx-3,),1)
#print A
b = zeros(Nx-2)
b[0] = left
b[-1] = right
#print b

# Time Step
for n in range(1,Nt):
    u[n,:] =  u[n-1,:] + r * A.dot( u[n-1,:]) + r * b
    uu[n,:] = sin(2*pi*x)*exp(-4*(pi**2)*dt*n)



#Plots
x_arr = linspace(0,xf,Nx)
t_arr = linspace(0,tf,Nt)
u_arr = zeros((Nt,Nx))
u_arr[:,1:Nx-1] = u  #note that final value when indexing (Nx-1) does not occur
u_arr[:,0] = left
u_arr[:,Nx-1] = right
uu_arr = zeros((Nt,Nx))
uu_arr[:,1:Nx-1] = uu  #note that final value when indexing (Nx-1) does not occur
uu_arr[:,0] = left
uu_arr[:,Nx-1] = right


# change x_arr and u_arr to x and u as desired (depending on if BC should be plotted)
X,T = meshgrid(x,t_arr)



fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, u, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')



fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, uu, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')
plt.title('actual solution')



plt.figure(3)
plt.plot(x,u[-1,:])
plt.xlabel('$x$')
plt.ylabel('$u(x,tn)$')
plt.title('final time slice')


fig = plt.figure(4)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, (u-uu)/uu, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('t')
plt.title('relative error')





plt.show()
