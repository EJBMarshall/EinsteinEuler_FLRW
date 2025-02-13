#!/usr/bin/env python
# encoding: utf-8

"""
Gowdy_EulerFC.py

Code to evolve the Einstein-Euler Equations
in Gowdy Symmetry in the expanding direction.

The Kurganov-Tadmor scheme is used to evolve the fluid.
Created by Elliot Marshall 2024-12-21
"""


### Import Python Libraries
import sys
import numpy as np
import h5py
import argparse
# from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from numba import njit
import time


t0 = time.time()


#######################################
#Parser Settings
#######################################

"""The code takes two compulsory arguments:
    -N - The number of grid points
    -K - The value of the sound speed K
     
    Additionally, there are two optional argument:
    -d - Plots the solution using matplotlib
    -f - Saves the file using HDF 
    (Note you must include the .hdf5 extension when specifying the file name) """

# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This program numerically solves an IVP for the
Einstein-Euler Equations in Gowdy Symmetry.""")

# Parse files
parser.add_argument('-d','-display', default = False, 
    action='store_true', help=\
"""A flag to indicate if visual display is required.""")
parser.add_argument('-K', type=float,help=\
"""The value of the parameter K.""")
parser.add_argument('-N', type=int,help=\
"""The number of grid points.""")
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced.""")
args = parser.parse_args()

#Output Settings
display_output = args.d
store_output = args.f is not None
if store_output and args.f is None:
    print("Gowdy_FC.py: error: argument -f/-file is required")
    sys.exit(1)

#Check Inputs
if args.K is None:
    print("Gowdy_FC.py: error: argument -K is required.")
    sys.exit(1)
if args.N is None:
    print("Gowdy_FC.py: error: argument -N is required.")
    sys.exit(1)

    

##############################################################
# Finite Difference Derivative Operators
##############################################################  
@njit(cache=True) 
def Dx(f,r):
    """ This function calculates the first derivative
        using a 2nd order central finite difference stencil"""
    h = r[1]
    n = np.shape(f)[0]
    df = np.zeros(np.shape(f))

    #2nd Order Central Finite Difference
    df[1:n-1] = (-f[0:n-2]+f[2:n])/(2*h)

    #Periodic Boundary Condition
    df[0] = (-f[n-1]+f[1])/(2*h)
    df[n-1] = (-f[n-2]+f[0])/(2*h)

    #4th Order Central Finite Difference
    # df[2:n-2] = (f[0:n-4]-8*f[1:n-3]+8*f[3:n-1]-f[4:n])/(12*h)

    # #Periodic Boundary Condition
    # df[1] = (f[n-1]-8*f[0]+8*f[2]-f[3])/(12*h)
    # df[0] = (f[n-2]-8*f[n-1]+8*f[1]-f[2])/(12*h)
    # df[n-2] = (f[n-4]-8*f[n-3]+8*f[n-1]-f[0])/(12*h)
    # df[n-1] = (f[n-3]-8*f[n-2]+8*f[0]-f[1])/(12*h)


    return df


@njit(cache=True) 
def DxF(f,r):
    """ This function calculates the first derivative
        using a 1st order forward difference. 
        This is only used if you want to use 
        the form of the Minmod Limiter given in Kurganov & Tadmor 2000."""
    h = r[1]
    n = np.shape(f)[0]
    df = np.zeros(np.shape(f))

    # #1st Order
    df[0:n-1] = (f[1:n]-f[0:n-1])/h
    df[n-1] = (f[0]-f[n-1])/h

    return df

# def Dx2(f,r):
#     """ This function calculates the second derivative
#         using a 4th order central finite difference stencil"""
#     h = r[1]
#     n = np.shape(f)[0]
#     df = np.zeros(np.shape(f))

    
#     #4th Order Central Finite Difference
#     df[2:n-2] = (-1/12*f[0:n-4]+4/3*f[1:n-3]-5/2*f[2:n-2]+4/3*f[3:n-1]-1/12*f[4:n])/(h**2)

#     #Periodic Boundary Condition
#     df[1] = (-1/12*f[-1]+4/3*f[0]-5/2*f[1]+4/3*f[2]-1/12*f[3])/(h**2)
#     df[0] = (-1/12*f[-2]+4/3*f[-1]-5/2*f[0]+4/3*f[1]-1/12*f[2])/(h**2)
#     df[n-2] = (-1/12*f[n-4]+4/3*f[n-3]-5/2*f[n-2]+4/3*f[n-1]-1/12*f[0])/(h**2)
#     df[n-1] = (-1/12*f[n-3]+4/3*f[n-2]-5/2*f[n-1]+4/3*f[0]-1/12*f[1])/(h**2)

#     return df

@njit(cache=True) 
def Dx4(f,r):
    """ Computes periodic 4th derivative for artificial viscosity"""
    
    n = np.shape(f)[0]
    df = np.zeros(np.shape(f))

    h = r[1]
    df[2:n-2] = (1*f[0:n-4]-4*f[1:n-3]+6*f[2:n-2]-4*f[3:n-1]+1*f[4:n])/h**4

    df[1] = (1*f[n-1]-4*f[0]+6*f[1]-4*f[2]+1*f[3])/h**4
    df[0] = (1*f[n-2]-4*f[n-1]+6*f[0]-4*f[1]+1*f[2])/h**4
    df[n-2] = (1*f[n-4]-4*f[n-3]+6*f[n-2]-4*f[n-1]+1*f[0])/h**4
    df[n-1] = (1*f[n-3]-4*f[n-2]+6*f[n-1]-4*f[0]+1*f[1])/h**4

    return df

@njit(cache=True) 
def Dx3(f,r):
    """ This function calculates the third derivative
        using a 4th order central finite difference stencil"""
    h = r[1]
    n = np.shape(f)[0]
    df = np.zeros(np.shape(f))

    
    #4th Order Central Finite Difference
    df[3:n-3] = (1/8*f[0:n-6]-1*f[1:n-5]+13/8*f[2:n-4]\
                 -13/8*f[4:n-2]+1*f[5:n-1]-1/8*f[6:n])/(h**3)

    #Periodic Boundary Condition
    df[2] = (1/8*f[-1]-1*f[0]+13/8*f[1]\
                 -13/8*f[3]+1*f[4]-1/8*f[5])/(h**3)
    df[1] = (1/8*f[-2]-1*f[-1]+13/8*f[0]\
                 -13/8*f[2]+1*f[3]-1/8*f[4])/(h**3)
    df[0] = (1/8*f[-3]-1*f[-2]+13/8*f[-1]\
                 -13/8*f[1]+1*f[2]-1/8*f[3])/(h**3)
    
    df[n-3] = (1/8*f[n-6]-1*f[n-5]+13/8*f[n-4]\
                 -13/8*f[n-2]+1*f[n-1]-1/8*f[0])/(h**3)
    df[n-2] = (1/8*f[n-5]-1*f[n-4]+13/8*f[n-3]\
                 -13/8*f[n-1]+1*f[0]-1/8*f[1])/(h**3)
    df[n-1] = (1/8*f[n-4]-1*f[n-3]+13/8*f[n-2]\
                 -13/8*f[0]+1*f[1]-1/8*f[2])/(h**3)
    
    return df


##############################################################
# Flux Limiter Style Reconstruction Functions
##############################################################  

### These functions are used to compute a slope-limited TVD reconstruction 
### of the primitive variables at the cell edges. In particular, I do *not* use the form 
### of the minmod limiter given in Kurganov-Tadmor 2000. We use flux-limiter type functions instead. 
###
### The two forms of the limiter are equivalent, see:
### LeVeque "Numerical Methods for Conservation Laws", 1992, pg 185-186.

@njit(cache=True)
def calc_r(f):
    """This calculates the ratio of consecutive gradients
    see LeVeque 1992 Chapter 16.2, eqn (16.15)"""

    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    ### NB: Add small number to denominator to 
    ### stop NaN issues when neighbouring grid points are close to equal

    df[1:n-1] = (f[1:n-1]-f[0:n-2])/(f[2:n]-f[1:n-1]+1e-16)
    df[0] = (f[0]-f[n-1])/(f[1]-f[0]+1e-16)
    df[n-1] = (f[n-1]-f[n-2])/(f[0]-f[n-1]+1e-16) ### Periodic BCs


    return df


### Various choices of limiter functions:

@njit(cache=True)
def Minmod(f):
    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    df = np.fmax(0,np.fmin(1,f))
    # df = np.fmax(0,np.fmin(1*f,np.fmin((1+f)/2,1))) #Generalised Minmod

    return df


@njit(cache=True)
def superbee(f):
    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    df = np.fmax(0,np.fmax(np.fmin(2*f,1),np.fmin(f,2)))

    return df

@njit(cache=True)
def MC(f):
    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    df = np.fmax(0,np.fmin(np.fmin(2*f,0.5*(1+f)),2))

    return df

@njit(cache=True)
def VA(f):
    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    df = (f**2 +f)/(f**2+1)

    return df

@njit(cache=True)
def ospre(f):
    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    df = (1.5*(f**2+f))/(f**2+f+1)

    return df

@njit(cache=True)
def vanleer(f):
    df = np.zeros(np.shape(f))
    n = np.shape(f)[0]

    df = (f+np.abs(f))/(1+np.abs(f))

    return df

@njit(cache=True)
def downwind_slopes(u,r):

    """This function reconstructs a function u at cell edge using a given flux limiter 
    applied to the ratio of consecutive gradients r """

    u_Rup = np.zeros_like(u)
    u_Rdown = np.zeros_like(u)
    u_Lup = np.zeros_like(u)
    u_Ldown = np.zeros_like(u)
    n = np.shape(u)[0]

    limiter = Minmod
    
    u_Lup[0:n-1] = u[0:n-1] + 0.5*limiter(r)[0:n-1]*(u[1:n]-u[0:n-1])
    u_Lup[n-1] = u[n-1] + 0.5*limiter(r)[n-1]*(u[0]-u[n-1])

    u_Rup[0:n-2] = u[1:n-1] - 0.5*limiter(r)[1:n-1]*(u[2:n]-u[1:n-1])
    u_Rup[n-2] = u[n-1] - 0.5*limiter(r)[n-1]*(u[0]-u[n-1])
    u_Rup[n-1] = u[0] - 0.5*limiter(r)[0]*(u[1]-u[0])

    u_Ldown = np.roll(u_Lup,1) ### NB: These are just defined to make constructing 
    u_Rdown = np.roll(u_Rup,1) ### the flux at each cell edge easier

    return u_Lup,u_Rup,u_Ldown,u_Rdown



##############################################################
# Kurganov-Tadmor Style Minmod Reconstruction
##############################################################  

"""These functions compute the reconstructions at cell edges using 
the form of the minmod limiter given in Kurganov-Tadmor 2000.

These were only used for testing purposes as I think 
the flux limiter method is nicer to work with."""

# @njit(cache=True)
# def Minmod(f,x):
#     F_up = DxF(f,x)
#     F_down = np.roll(F_up,1)
#     # n = np.shape(f)[0]

#     # df = np.fmax(0,np.fmin(1,f))
#     df = 0.5*(np.sign(F_up)+np.sign(F_down))*np.fmin(np.abs(F_up),np.abs(F_down))

#     return df

# @njit(cache=True)
# def downwind_slopes(u,x):
#     u_Rup = np.zeros_like(u)
#     u_Rdown = np.zeros_like(u)
#     u_Lup = np.zeros_like(u)
#     u_Ldown = np.zeros_like(u)
#     n = np.shape(u)[0]

#     h = x[1]-x[0]

#     limiter = Minmod
    
#     u_Lup[0:n-1] = u[0:n-1] + 0.5*h*limiter(u,r)[0:n-1]
#     u_Lup[n-1] = u[n-1] + 0.5*h*limiter(u,r)[n-1]

#     u_Rup[0:n-2] = u[1:n-1] - 0.5*h*limiter(u,r)[1:n-1]
#     u_Rup[n-2] = u[n-1] - 0.5*h*limiter(u,r)[n-1]
#     u_Rup[n-1] = u[0] - 0.5*h*limiter(u,r)[0]


#     u_Ldown = np.roll(u_Lup,1)
#     u_Rdown = np.roll(u_Rup,1)


#     # return np.array([u_Lup,u_Rup,u_Ldown,u_Rdown])
#     return u_Lup,u_Rup,u_Ldown,u_Rdown




############################################################
#System of PDEs to Solve
###########################################################
def Gowdy(f,r,t,K):

    """This function computes the evolution equations.
    f - The functions we are evolving
    r - This is the grid variable (not the ratio of slopes!)
    t - Time
    K - Sound Speed"""

    global CS ### Max size of the characteristic speed - used for adjusting the time steps

    A, A0_tilde, A1_tilde, U, U0_tilde, U1_tilde, alpha, tau_tilde, S_tilde = f

    h = r[1]

    #####################################
    # Define Useful Variables
    #####################################

    tau = tau_tilde*np.exp(alpha)
    S = S_tilde*np.exp(alpha)
    A0 = A0_tilde*np.exp(alpha)
    A1 = A1_tilde*np.exp(alpha)
    U0 = U0_tilde*np.exp(alpha) + 0.5
    U1 = U1_tilde*np.exp(alpha)

    ######### Recover Primitive Variables ##########

    Q = S**2/((1+K)**2*tau**2)
    Gamma2 = (1 -2*K*(1+K)*Q + np.sqrt(1-4*K*Q))/(2*(1-(1+K)**2*Q))
        
    mu = tau/((K+1)*Gamma2-K)
    v = S/((K+1)*Gamma2*mu)

    
    ####################################################################
    # Compute Reconstruction of Primitive Variables using Flux Limiter
    ###################################################################
    v_Lup,v_Rup,v_Ldown,v_Rdown = downwind_slopes(v,calc_r(v))

    Gamma2_Lup,Gamma2_Rup,Gamma2_Ldown,Gamma2_Rdown = downwind_slopes(Gamma2,calc_r(Gamma2))

    mu_Lup,mu_Rup,mu_Ldown,mu_Rdown = downwind_slopes(mu,calc_r(mu))

    alpha_Lup,alpha_Rup,alpha_Ldown,alpha_Rdown = downwind_slopes(alpha,calc_r(alpha))

    ### If you want to evolve metric variables using KT scheme uncomment:

    # A0_Lup,A0_Rup,A0_Ldown,A0_Rdown = downwind_slopes(A0,calc_r(A0))

    # A1_Lup,A1_Rup,A1_Ldown,A1_Rdown = downwind_slopes(A1,calc_r(A1))

    # U0_Lup,U0_Rup,U0_Ldown,U0_Rdown = downwind_slopes(U0,calc_r(U0))

    # U1_Lup,U1_Rup,U1_Ldown,U1_Rdown = downwind_slopes(U1,calc_r(U1))


    ####################################################################
    # Compute Reconstruction using K-T Minmod (Only for testing)
    ###################################################################

    # v_Lup,v_Rup,v_Ldown,v_Rdown = downwind_slopes(v,r)

    # Gamma2_Lup,Gamma2_Rup,Gamma2_Ldown,Gamma2_Rdown = downwind_slopes(Gamma2,r)

    # mu_Lup,mu_Rup,mu_Ldown,mu_Rdown = downwind_slopes(mu,r)

    # alpha_Lup,alpha_Rup,alpha_Ldown,alpha_Rdown = downwind_slopes(alpha,r)

    # A0_Lup,A0_Rup,A0_Ldown,A0_Rdown = downwind_slopes(A0,r)

    # A1_Lup,A1_Rup,A1_Ldown,A1_Rdown = downwind_slopes(A1,r)

    # U0_Lup,U0_Rup,U0_Ldown,U0_Rdown = downwind_slopes(U0,r)

    # U1_Lup,U1_Rup,U1_Ldown,U1_Rdown = downwind_slopes(U1,r)


    ###############################################################
    # Define Flux Components
    ################################################################
    
    tauflux_Lup = ((1+K)*Gamma2_Lup*mu_Lup*v_Lup)
    tauflux_Rup = ((1+K)*Gamma2_Rup*mu_Rup*v_Rup)
    tauflux_Ldown = ((1+K)*Gamma2_Ldown*mu_Ldown*v_Ldown)
    tauflux_Rdown = ((1+K)*Gamma2_Rdown*mu_Rdown*v_Rdown)

    
    Sflux_Lup = ((K+1)*Gamma2_Lup*v_Lup*v_Lup*mu_Lup + K*mu_Lup)
    Sflux_Rup = ((K+1)*Gamma2_Rup*v_Rup*v_Rup*mu_Rup + K*mu_Rup)
    Sflux_Ldown = ((K+1)*Gamma2_Ldown*v_Ldown*v_Ldown*mu_Ldown + K*mu_Ldown)
    Sflux_Rdown = ((K+1)*Gamma2_Rdown*v_Rdown*v_Rdown*mu_Rdown + K*mu_Rdown)


    ################################################################
    # Compute Reconstruction of Conserved Variables
    ###############################################################
    tau_Lup = np.exp(-alpha_Lup)*((1+K)*Gamma2_Lup*mu_Lup - K*mu_Lup)
    tau_Rup = np.exp(-alpha_Rup)*((1+K)*Gamma2_Rup*mu_Rup - K*mu_Rup)
    tau_Ldown = np.exp(-alpha_Ldown)*((1+K)*Gamma2_Ldown*mu_Ldown - K*mu_Ldown)
    tau_Rdown = np.exp(-alpha_Rdown)*((1+K)*Gamma2_Rdown*mu_Rdown - K*mu_Rdown)

    S_Lup = np.exp(-alpha_Lup)*((1+K)*Gamma2_Lup*mu_Lup*v_Lup)
    S_Rup = np.exp(-alpha_Rup)*((1+K)*Gamma2_Rup*mu_Rup*v_Rup)
    S_Ldown = np.exp(-alpha_Ldown)*((1+K)*Gamma2_Ldown*mu_Ldown*v_Ldown)
    S_Rdown = np.exp(-alpha_Rdown)*((1+K)*Gamma2_Rdown*mu_Rdown*v_Rdown)

    ### If you want to evolve metric variables using KT scheme uncomment:

    # A0tilde_Lup = np.exp(-alpha_Lup)*A0_Lup
    # A0tilde_Rup = np.exp(-alpha_Rup)*A0_Rup
    # A0tilde_Ldown = np.exp(-alpha_Ldown)*A0_Ldown
    # A0tilde_Rdown = np.exp(-alpha_Rdown)*A0_Rdown

    # A1tilde_Lup = np.exp(-alpha_Lup)*A1_Lup
    # A1tilde_Rup = np.exp(-alpha_Rup)*A1_Rup
    # A1tilde_Ldown = np.exp(-alpha_Ldown)*A1_Ldown
    # A1tilde_Rdown = np.exp(-alpha_Rdown)*A1_Rdown

    # U0tilde_Lup = np.exp(-alpha_Lup)*U0_Lup
    # U0tilde_Rup = np.exp(-alpha_Rup)*U0_Rup
    # U0tilde_Ldown = np.exp(-alpha_Ldown)*U0_Ldown
    # U0tilde_Rdown = np.exp(-alpha_Rdown)*U0_Rdown

    # U1tilde_Lup = np.exp(-alpha_Lup)*U1_Lup
    # U1tilde_Rup = np.exp(-alpha_Rup)*U1_Rup
    # U1tilde_Ldown = np.exp(-alpha_Ldown)*U1_Ldown
    # U1tilde_Rdown = np.exp(-alpha_Rdown)*U1_Rdown

    ###################################################################
    # Calculate Characteristic Speeds
    ###################################################################
    #### Calculate Eigenvalues + Characteristic Speeds

    lam1 = np.exp(alpha)*((K-1)*v + (1-v**2)*np.sqrt(K))/(-1+K*v**2)
    lam2 = np.exp(alpha)*((K-1)*v - (1-v**2)*np.sqrt(K))/(-1+K*v**2)
    # lam3 = np.exp(alpha)

    CS = np.max(np.abs([lam1,lam2]))

    ### If you want to evolve metric variables using KT scheme we have another e-value:
    # CS = np.max(np.abs([lam1,lam2,lam3]))

    lam1_Lup = np.exp(alpha_Lup)*((K-1)*v_Lup + (1-v_Lup**2)*np.sqrt(K))/(-1+K*v_Lup**2)
    lam1_Rup = np.exp(alpha_Rup)*((K-1)*v_Rup + (1-v_Rup**2)*np.sqrt(K))/(-1+K*v_Rup**2)
    lam1_Ldown = np.exp(alpha_Ldown)*((K-1)*v_Ldown + (1-v_Ldown**2)*np.sqrt(K))/(-1+K*v_Ldown**2)
    lam1_Rdown = np.exp(alpha_Rdown)*((K-1)*v_Rdown + (1-v_Rdown**2)*np.sqrt(K))/(-1+K*v_Rdown**2)


    lam2_Lup = np.exp(alpha_Lup)*((K-1)*v_Lup - (1-v_Lup**2)*np.sqrt(K))/(-1+K*v_Lup**2)
    lam2_Rup = np.exp(alpha_Rup)*((K-1)*v_Rup - (1-v_Rup**2)*np.sqrt(K))/(-1+K*v_Rup**2)
    lam2_Ldown = np.exp(alpha_Ldown)*((K-1)*v_Ldown - (1-v_Ldown**2)*np.sqrt(K))/(-1+K*v_Ldown**2)
    lam2_Rdown = np.exp(alpha_Rdown)*((K-1)*v_Rdown - (1-v_Rdown**2)*np.sqrt(K))/(-1+K*v_Rdown**2)

    a_up = np.fmax.reduce(np.abs([lam1_Lup,lam1_Rup,lam2_Lup,lam2_Rup]))
    a_down = np.fmax.reduce(np.abs([lam1_Ldown,lam1_Rdown,lam2_Ldown,lam2_Rdown]))


    ### Use three e-values if evolving metric with KT scheme:

    # lam3_Lup = np.exp(alpha_Lup)
    # lam3_Rup = np.exp(alpha_Rup)
    # lam3_Ldown = np.exp(alpha_Ldown)
    # lam3_Rdown = np.exp(alpha_Rdown)

    # a_up = np.fmax.reduce(np.abs([lam1_Lup,lam1_Rup,lam2_Lup,lam2_Rup,lam3_Lup,lam3_Rup]))
    # a_down = np.fmax.reduce(np.abs([lam1_Ldown,lam1_Rdown,lam2_Ldown,lam2_Rdown,lam3_Ldown,lam3_Rdown]))

    

    ##################################################################
    # Calculate Left and Right Fluxes
    ##################################################################
    F_tau_down = 0.5*((tauflux_Rdown + tauflux_Ldown) - a_down*(tau_Rdown-tau_Ldown))
    F_tau_up = 0.5*((tauflux_Rup + tauflux_Lup) - a_up*(tau_Rup-tau_Lup))

    FS1_down = 0.5*((Sflux_Rdown + Sflux_Ldown) - a_down*(S_Rdown-S_Ldown)) 
    FS1_up = 0.5*((Sflux_Rup + Sflux_Lup) - a_up*(S_Rup-S_Lup))


    ### Uncomment to evolve metric using KT Scheme:

    # F_A0_down = 0.5*(-(A1_Rdown + A1_Ldown) - a_down*(A0tilde_Rdown-A0tilde_Ldown))
    # F_A0_up = 0.5*(-(A1_Rup + A1_Lup) - a_up*(A0tilde_Rup-A0tilde_Lup))

    # F_A1_down = 0.5*(-(A0_Rdown + A0_Ldown) - a_down*(A1tilde_Rdown-A1tilde_Ldown)) 
    # F_A1_up = 0.5*(-(A0_Rup + A0_Lup) - a_up*(A1tilde_Rup-A1tilde_Lup))

    # F_U0_down = 0.5*(-(U1_Rdown + U1_Ldown) - a_down*(U0tilde_Rdown-U0tilde_Ldown))
    # F_U0_up = 0.5*(-(U1_Rup + U1_Lup) - a_up*(U0tilde_Rup-U0tilde_Lup))

    # F_U1_down = 0.5*(-(U0_Rdown + U0_Ldown) - a_down*(U1tilde_Rdown-U1tilde_Ldown)) 
    # F_U1_up = 0.5*(-(U0_Rup + U0_Lup) - a_up*(U1tilde_Rup-U1tilde_Lup))


    ############################################
    # ODE Evolution Equations
    ###########################################
    dtA = A0 

    dtU = U0 

    dtalpha = 1 + (-1 + K)*mu 

    ############################################
    # Compute Source Terms for PDE Equations
    ###########################################

    A0tilde_Source = np.exp(-alpha)*(A0 + 4*A1*U1 - 4*A0*U0)

    U0tilde_Source = np.exp(-alpha)*(0.5*np.exp(-2*t+4*U)*(A0**2-A1**2) + 0.5 - U0)

    tautilde_Source = (np.exp(-2*t - alpha)*mu*(-4*np.exp(2*t)*K \
                        + (-1 + K)*(-(np.exp(4*U)*(A1**2 + A0**2)) - 4*np.exp(2*t)*(U1**2 + (-1 + U0)*U0))))/4.

    Stilde_Source = np.exp(-alpha)*((-1 + K)*mu*(np.exp(-2*t + 4*U)*A1*A0 + U1*(-2 + 4*U0)))/2.

    ##################################################################
    # Einstein Evolution Equations
    ##################################################################

    ### Finite Difference Method
    dtA0_tilde = -Dx(-A1,r) + A0tilde_Source 

    dtA1_tilde = -Dx(-A0,r) 

    dtU0_tilde = -Dx(-U1,r) + U0tilde_Source 

    dtU1_tilde = -Dx(-U0,r) 


    ### KT Method
    # dtA0_tilde = (-1/h)*(F_A0_up-F_A0_down) + A0tilde_Source

    # dtA1_tilde = (-1/h)*(F_A1_up-F_A1_down) 

    # dtU0_tilde = (-1/h)*(F_U0_up-F_U0_down) + U0tilde_Source 

    # dtU1_tilde = (-1/h)*(F_U1_up-F_U1_down) 

    
    ##################################################################
    # Euler Evolution Equations
    ##################################################################
    dttau_tilde = (-1/h)*(F_tau_up-F_tau_down) + tautilde_Source 

    dtS_tilde = (-1/h)*(FS1_up-FS1_down) + Stilde_Source 

    ######################################################################
    # Collect Evolved Quantities
    ######################################################################
    

    return np.array([dtA,dtA0_tilde,dtA1_tilde,dtU,dtU0_tilde,dtU1_tilde,dtalpha,dttau_tilde,dtS_tilde])

    
############################################################################
#Norm functions
############################################################################


def computel2norm(f,r):
    """Note this function outputs a vector whose 
    ith entry is the L2 norm at timestep i"""
    l2norm = np.zeros_like(f[:,1])
    for i in range(0,np.shape(f)[0]):
        l2norm[i] = np.sqrt(r[1]*np.sum(f[i,:]**2))
    return l2norm

def computeH3norm(f,r):
    """This function outputs a vector whose 
    ith entry is the L2 norm of third derivative at timestep i"""
    l2norm = np.zeros_like(f[:,1])
    for i in range(0,np.shape(f)[0]):
        l2norm[i] = np.sqrt(r[1]*np.sum(Dx3(f[i,:],r)**2))
    return l2norm


###########################################################################
# Runge-Kutta Scheme
##########################################################################
def rk(f,t0,tf,yinit,dt,x_step):
    """Strong Stability preserving 3rd order Runge-Kutta"""
    y = []
    t = []
    y.append(yinit[:])
    t.append(t0)
    y0 = yinit[:]
    h = dt
    i = 0
    while t0<tf:
            i += 1 

            ### RK2 (Heun's Method)
            # k1 = f(t0,y0)
            # k2 = f(t0+h,y0+h*k1)
            # y0 = y0 + h*(0.5*k1 + 0.5*k2) 
            
            ### SSP RK3 Method (Osher-Shu?)  
            # k1 = f(t0,y0)
            # k2 = f(t0+h,y0+h*k1)
            # k3 = f(t0+0.5*h,y0+h*(0.25*k1+0.25*k2))
            # y0 = y0 + h*(1/6*k1 + 1/6*k2 + 2/3*k3)

            ### 'Classic' RK4 Method
            k1 = f(t0,y0)
            k2 = f(t0+0.5*h,y0+0.5*h*k1)
            k3 = f(t0 +0.5*h,y0+0.5*h*k2)
            k4 = f(t0+h,y0+h*k3)
            y0 = y0 + h*(1/6*k1 + 1/3*k2 +1/3*k3 +1/6*k4)

            t0 = t0 + h
            h = 0.5*(x_step)/(CS) ### Adjust timestep based on characteristic speeds
            if h > 0.5*x_step:
                h = 0.5*x_step

            if i == 30: ### Timesteps to output
                i = 0
                y.append(y0)
                t.append(t0)
    return [np.array(t),np.array(y)]

    
############################################################################
#Creating grid and initial data
############################################################################

R = 2*np.pi #Length of grid
N = args.N #Number of Grid Points 
r = np.delete(np.linspace(0,R,num=N+1),-1) #Uniform Grid
h = r[1] #Grid Spacing
K = args.K #Value of K 

##########################
# Initial Data
##########################

#Parameters for A0, A etc.
a = 0.05 
b = 0.05
c = 0.05
d = 0.05

t_init = 0.0

######## Perturbed FLRW Initial Data #########

# nu = -c*(1+2*d)*np.cos(r) + 3/4*(1+K)*b*np.exp(-a)*np.cos(r) 

### Metric:
U = -c*np.cos(r)

alpha = 0*r + a

A = c*np.sin(r)

A0 = 0*r

A1 = c*np.exp(alpha)*np.cos(r)

U0 = 0.5 + d + 0*np.sin(r)

U1 = c*np.exp(alpha)*np.sin(r)

### Construct Conserved Metric Variables
A0_tilde = np.exp(-alpha)*A0

A1_tilde = np.exp(-alpha)*A1

U0_tilde = np.exp(-alpha)*(U0-0.5)

U1_tilde = np.exp(-alpha)*U1

### Fluid:
v = b*np.sin(r)

Gamma2 = 1/(1-v**2)

mu = 3/4*(1-v**2)

tau = (K+1)*mu*Gamma2 -K*mu

S = (K+1)*mu*Gamma2*v


### Construct Conserved Fluid Variables
tau_tilde = np.exp(-alpha)*tau

S_tilde = np.exp(-alpha)*S


### Initial Eigenvalues
lam1 = np.exp(alpha)*((K-1)*v + (1-v**2)*np.sqrt(K))/(-1+K*v**2)
lam2 = np.exp(alpha)*((K-1)*v - (1-v**2)*np.sqrt(K))/(-1+K*v**2)
lam3 = np.exp(alpha)

CS = np.max(np.abs([lam1,lam2,lam3]))

# print(-Dx(nu,r) + ((1 + K)*v*mu)/(np.exp(alpha)*(-1 + v**2)) + (np.exp(-2*t + 4*U)*Dx(A,r)*A0)/2. + 2*Dx(U,r)*U0)
# breaks

# plt.plot(((1 + K)*v*mu)/(np.exp(alpha)*(-1 + v**2)) + (np.exp(-2*t0 + 4*U)*Dx(A,r)*A0)/2. + 2*Dx(U,r)*U0)
# plt.show()
# breaks


#### Perfect Fluid FLRW Solution Initial Data

# A0 = 0*np.sin(r)
# A1 = 0*np.sin(r)
# A = 0*np.sin(r)
# U0 = 0*np.sin(r)+  0.5
# U1 = 0*np.sin(r) 
# U = 0*np.sin(r) + 0.5*t_init
# nu = 0*np.sin(r) + t_init + 0.25*(1+3*K)*t_init
# mu = 0*np.sin(r) + 3/4
# alpha = 0*np.sin(r) + 0.25*(1+3*K)*t_init
# S_tilde = 0*r
# tau_tilde = np.exp(-alpha)*((K+1)*mu -K*mu)

y0 = np.array([A,A0_tilde,A1_tilde,U,U0_tilde,U1_tilde,alpha,tau_tilde,S_tilde])


######## Evolution Equation ############
rhs = lambda t,y: Gowdy(y,r,t,K)
[t,y] = rk(rhs, 0, 8, y0, 0.5*(r[1])/(CS),r[1])

t1 = time.time()
print("Elapsed time is",t1-t0) #Prints simulation length of time 


####################################
#Create HDF File
####################################
if args.f is not None:
    
    ### Change location of saved file here:
    h5f = h5py.File(\
        "/Users/elliotmarshall/Desktop/Gowdy_ShockCapture/HDF_Files/"\
        +args.f, 'w')
    h5f.create_dataset('Solution Matrices', data=y,compression="gzip")#_hdf.T)
    h5f.create_dataset('Time Vector', data=t,compression="gzip")
    h5f.create_dataset('K', data=K)
    h5f.close()




############################
#Plot with MatPlotLib
############################
A = y[:,0]
A0_tilde = y[:,1]
A1_tilde = y[:,2]
U = y[:,3]
U0_tilde = y[:,4]
U1_tilde = y[:,5]
alpha = y[:,6]
tau_tilde = y[:,7]
S_tilde = y[:,8]

#### Recover Primitives
A0 = np.zeros_like(A)
A1 = np.zeros_like(A)
U0 = np.zeros_like(A)
U1 = np.zeros_like(A)
tau = np.zeros_like(A)
S = np.zeros_like(A)
mu = np.zeros_like(A)
Q = np.zeros_like(A)
Gamma2 = np.zeros_like(A)
rho = np.zeros_like(A)
v= np.zeros_like(A)
C_U1 = np.zeros_like(A)
C_A1 = np.zeros_like(A)
for i in range(np.shape(t)[0]):
    A0[i,:] = A0_tilde[i,:]*np.exp(alpha[i,:])

    A1[i,:] = A1_tilde[i,:]*np.exp(alpha[i,:])

    U0[i,:] = U0_tilde[i,:]*np.exp(alpha[i,:]) + 0.5

    U1[i,:] = U1_tilde[i,:]*np.exp(alpha[i,:])

    tau[i,:] = tau_tilde[i,:]*np.exp(alpha[i,:])

    S[i,:] = S_tilde[i,:]*np.exp(alpha[i,:])

    ######### Recover Primitive Variables ##########
    Q[i,:] = S[i,:]**2/((1+K)**2*tau[i,:]**2)
    Gamma2[i,:] = (1 -2*K*(1+K)*Q[i,:] + np.sqrt(1-4*K*Q[i,:]))/(2*(1-(1+K)**2*Q[i,:]))
    
    mu[i,:] = tau[i,:]/((K+1)*Gamma2[i,:]-K)
    v[i,:] = S[i,:]/((K+1)*Gamma2[i,:]*mu[i,:])

    C_U1[i,:] = np.exp(alpha[i,:])*Dx(U[i,:],r)-U1[i,:] ### A1, U1 Constraints
    C_A1[i,:] = np.exp(alpha[i,:])*Dx(A[i,:],r)-A1[i,:]



# plt.plot(t,alpha[:,0],'x')
# plt.plot(t,1/4*(1+3*K)*t)
# plt.show()

# totalH3 = computeH3norm(v,r)**2 + computeH3norm(mu,r)**2  
# totall2 = computel2norm(v,r)**2 + computel2norm(mu,r)**2 
# plt.plot(t,totalH3/totall2)
# plt.yscale("log")
# plt.show()

# plt.plot(np.log2(computel2norm(C_U1,r)),label='U1')
# plt.plot(np.log2(computel2norm(C_A1,r)),label='A1')
# plt.legend()
# plt.show()

# print(t[300])
# plt.rcParams.update({"text.usetex": True,
#                 "font.family": "serif",
#                 "font.serif": "Computer Modern",
#                 "savefig.bbox": "tight",
#                 "savefig.format": "pdf"})
# plt.rc('font', size=16)
# plt.plot(r,v[300,:])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$v$')
# plt.tight_layout()
# plt.show()


if display_output is True:
    plt.ion()
    # plt.rc('font', size=18)
    fig, ax = plt.subplots()
    
    for i in range(0,np.shape(t)[0],1):
            
            ax.plot(r,v[i,:])
                                        
            #ax.legend()
            ax.set_title(f'Time = {t[i]}')
            #ax.set_title(r'$\alpha$ exact vs numerical')
            plt.draw()
            plt.pause(0.01) 

            # if i == 300:
                
            #     # ax.set_ylabel(r'$\frac{\partial_{\theta}\rho}{\rho}$', rotation=0)
            #     # ax.set_xlabel(r'$\theta$')
                
            #     #ax.set_ylabel(r'$|R|$')
            #     plt.tight_layout()
            #     plt.ioff()
            #     plt.show()
            #     break

            plt.cla()
            # ax[0].cla()
            # ax[1].cla()
            # ax[1,0].cla()
            # ax[1,1].cla()





            
            



