#!/usr/bin/env python2.7

# the equation is linear, but I use a nonlinear solver so that the code is more general!

# for the Neumann BCs, we imagine that there are "fake" ("ghost") indices over the edge, but, when we use the CENTERED 1st derivative formula (because the backward and forward derivatives have error of order (Delta x)^0 or (Delta y)^0), we see that those "fake" indices never appear or affect anything, so this code does not need to include them



from numpy import *
from scipy.optimize import newton_krylov   #our fancy non-linear system solver
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# Numerical Parameters
Nx = 30
Ny = 30
Nt = 1000
tf = 1.01
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
u = zeros((Nt,Nx,Ny))

#initial condition
for i in range(Nx):
  x = i*dx
  for j in range(Ny):
    y = j*dy
    r = (x-1.0/2)**2+(y-1.0/2)**2
    if 1.0/16 <= r and r <= 1.0/8:
      u[0,i,j] = 5.0
    else:
      u[0,i,j] = 0.0


#print 1.0/2
#quit()


def residual(input):

  output = zeros((Nx,Ny))

  # interior
  output[1:Nx-1,1:Ny-1] = input[1:Nx-1,1:Ny-1] - rx/2*(input[0:Nx-2,1:Ny-1]+input[2:Nx,1:Ny-1]-2*input[1:Nx-1,1:Ny-1]) - ry*(input[1:Nx-1,0:Ny-2]+input[1:Nx-1,2:Ny]-2*input[1:Nx-1,1:Ny-1]) - RHS

  # left (equation changes for left BC)
  output[0,1:Ny-1] = input[0,1:Ny-1] - rx*(input[1,1:Ny-1]-input[0,1:Ny-1]) - ry*(input[0,0:Ny-2]+input[0,2:Ny]-2*input[0,1:Ny-1]) - RHSleft

  # bottom (equation changes for bottom BC)
  output[1:Nx-1,0] = input[1:Nx-1,0] - rx/2*(input[0:Nx-2,0]+input[2:Nx,0]-2*input[1:Nx-1,0]) - ry*2*(input[1:Nx-1,1]-input[1:Nx-1,0]) - RHSbottom

  # bottom left (equation changes for left and bottom BCs)
  output[0,0] = input[0,0] - rx*(input[1,0]-input[0,0]) - ry*2*(input[0,1]-input[0,0]) - RHSbottomleft

  # sets top BC
  output[:,Ny-1] = input[:,Ny-1] - top

  # sets right BC (do not add the top right point so that it doesn't appear twice; does NOT matter)
  output[Nx-1,0:Ny-1] = input[Nx-1,0:Ny-1] - right

  return output

  

# Time Stepping
for n in range(1,Nt):
   RHS = u[n-1,1:Nx-1,1:Ny-1] + rx/2*(u[n-1,0:Nx-2,1:Ny-1]+u[n-1,2:Nx,1:Ny-1]-2*u[n-1,1:Nx-1,1:Ny-1]) + ry*(u[n-1,1:Nx-1,0:Ny-2]+u[n-1,1:Nx-1,2:Ny]-2*u[n-1,1:Nx-1,1:Ny-1])
   RHSleft = u[n-1,0,1:Ny-1] + rx*(u[n-1,1,1:Ny-1]-u[n-1,0,1:Ny-1]) + ry*(u[n-1,0,0:Ny-2]+u[n-1,0,2:Ny]-2*u[n-1,0,1:Ny-1])
   RHSbottom = u[n-1,1:Nx-1,0] + rx/2*(u[n-1,0:Nx-2,0]+u[n-1,2:Nx,0]-2*u[n-1,1:Nx-1,0]) + ry*2*(u[n-1,1:Nx-1,1]-u[n-1,1:Nx-1,0])
   RHSbottomleft = u[n-1,0,0] + rx*(u[n-1,1,0]-u[n-1,0,0]) + ry*2*(u[n-1,0,1]-u[n-1,0,0])
   u[n] = newton_krylov(residual,zeros((Nx,Ny)))



#Plots

x_arr = linspace(0,xf,Nx)
y_arr = linspace(0,yf,Ny)
X,Y = meshgrid(x_arr,y_arr)

ti=0;
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, transpose(u[ti,0:Nx,0:Ny]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('y')
plt.title('initial time')

ti=round(Nt/100)-1;
fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, transpose(u[ti,0:Nx,0:Ny]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('y')
plt.title('1/100 of final time')

ti=Nt-1;
fig = plt.figure(3)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, transpose(u[ti,0:Nx,0:Ny]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.0, antialiased=False)
fig.colorbar(surf)
plt.xlabel('x')
plt.ylabel('y')
plt.title('final time')

plt.show()


