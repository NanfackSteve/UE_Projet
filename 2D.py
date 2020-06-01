import matplotlib.pyplot as plt
import numpy as np
import random
import math

#-------------------------- FONCTIONS ------------------------------------------------

#--------------------- MATRICE DE CORRELATION ----------------------------------------

def se_corr(x, y, lxy):
    """Square exponential correlation kernel or Gaussian kernel in 2D
    x and y   -- vector of coordinates
    lx and ly -- decorrelation lenght scale
    In this example, the decorrelation lenght scale is constant.
    However, in the general case, the decorrelation lenght scale is not constant. It changes from one point to another.
    """
    n = len(x)
    m = len(y)
    C = np.zeros((n,m,n,m,)) #matrice de dimension 4
    two_lxy2 = 2.0*lxy*lxy

    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    distx = x[i] - x[k]
                    disty = y[j] - y[l]
                    C[i,j,k,l] = np.exp( -(distx**2 + disty**2)/two_lxy2 )

    return C

#-------------------- NORMALISATION DE LA MATRICE -----------------------------------

#encore a modifier
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

#------------- INITIALISATION DES VARIABLES --------------------------

#construction des Matrices X et Y
x, y = np.linspace(0,1,10), np.linspace(0,1,10) 
dCoordx = x[1]-x[0]
dCoordy = y[1]-y[0] 
lxy = 10.0*dCoordx*dCoordy

C = se_corr(x, y, lxy) #obtention de la matrix de correlation
print(C)
