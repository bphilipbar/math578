#!/usr/bin/env python2.7

from numpy import *
from scipy.optimize import newton_krylov   #our fancy non-linear system solver
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# Numerical Parameters
Nx = 30  # NOT including the "fake" value on left
Ny = 30  # NOT including the "fake" value on bottom
Nt = 300
tf = 0.28
xf = 1.0
yf = 1.0
dx = xf/(Nx-1)
dy = yf/(Ny-1)
dt = tf/(Nt-1)
rx = dt/(dx**2)
ry = dt/(dy**2)

top = 0.0
right = 0.0

#Solution Array
u = zeros((Nt,Nx+1,Ny+1))  # where the +1 is by creating a "fake" location on left and bottom to handle 1st-derivative BC

#initial condition
for i in range(Nx):
  x = i*dx
  for j in range(Ny):
    y = j*dy
    r = (x-1.0/2)**2+(y-1.0/2)**2
    if 1.0/16 <= r and r <= 1.0/8:
      u[0,i+1,j+1] = 5.0
    else:
      u[0,i+1,j+1] = 0.0


#print 1.0/2
#quit()


def residual(input):
  output = zeros((Nx+1,Ny+1))
  # interior
  output[1:Nx,1:Ny] = input[1:Nx,1:Ny] - rx/2*(input[0:Nx-1,1:Ny]+input[2:Nx+1,1:Ny]-2*input[1:Nx,1:Ny]) - ry*(input[1:Nx,0:Ny-1]+input[1:Nx,2:Ny+1]-2*input[1:Nx,1:Ny]) - RHS
  # top
  output[:,Ny] = input[:,Ny] - top
  # right (do NOT add the top right point so that it doesn't appear twice; does NOT matter)
  output[Nx,0:Ny] = input[Nx,0:Ny] - right
  # bottom
  output[1:Nx,0] = input[1:Nx,0] - u[n,1:Nx,1]
  # left
  output[0,1:Ny] = input[0,1:Ny] - u[n,1,1:Ny]
  # bottom left (does NOT matter, but having all inputs appear seems wise)
  output[0,0] = input[0,0] - 0
  return output
  

# Time Stepping
for n in range(1,Nt):
   RHS = u[n-1,1:Nx,1:Ny] + rx/2*(u[n-1,0:Nx-1,1:Ny]+u[n-1,2:Nx+1,1:Ny]-2*u[n-1,1:Nx,1:Ny]) + ry*(u[n-1,1:Nx,0:Ny-1]+u[n-1,1:Nx,2:Ny+1]-2*u[n-1,1:Nx,1:Ny])
   u[n] = newton_krylov(residual,zeros((Nx+1,Ny+1)))



#Plots

x_arr = linspace(0,xf,Nx)
y_arr = linspace(0,yf,Ny)
X,Y = meshgrid(x_arr,y_arr)

ti=0;
#ti=round(Nt/30)-1;
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, transpose(u[ti,1:Nx+1,1:Ny+1]), rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('y')

ti=Nt-1;
fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, transpose(u[ti,1:Nx+1,1:Ny+1]), rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('y')

plt.show()


