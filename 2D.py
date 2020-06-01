import matplotlib.pyplot as plt
import numpy as np
import math
import random

def make_simple_covariance(C):
    """Make a simple covariance by normalizing a correlation matrix.
    this builds a pseudo covariance that can be used for Gaussian denoising
    """
    n = C.shape[0] 

    W = np.zeros_like(C) #matrix similaire a C remplie de 0
    for k in range(n):
        W[k,k] = 1./np.sqrt(sum(C[k,:])) #1/racine carree de la sum de chaque ligne
        pass
    B = np.dot( W, np.dot( C, W ) )
    return B

#construction des Matrices X et Y
x, y = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10)) 
#Calcul de d
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
#utilisation du Gausian Kernel for 2D Pbs
g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

print("2D Gaussian-like array:\n")
#print(g)

B = make_simple_covariance(g)
#print("Affiche B:\n",B)

# generating Gaussian noise of mean 0 and standard deviation 0.2
rnd = np.array( [random.gauss(0., 0.2) for _ in x] )
#print(rnd)

# making a Sine wave
sinx = np.array([ math.sin( 2.*math.pi*(i-0.5) ) for i in x[0,:]] )
noisy = sinx + rnd

# Denoising
denoised = np.dot(B, noisy)

fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.plot(x[0,:], noisy, "r--", lw = 2.0, label="Noisy")
ax.plot(x[0,:], sinx, "g-", lw = 4.0, label="Truth")
ax.plot(x[0,:], denoised, "k-", lw = 2.0, label="denoised")
ax.legend( loc=3 )
#plt.ylim( 0., 1.2 )
ax.set_title('Random and correlated')


ax = axs[1]
cplot = ax.contourf(x, y, g)
#ax.axis('equal')
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('SE correlation')
fig.colorbar(cplot, shrink=0.5, aspect=5)


plt.show()
plt.close()

exit()
