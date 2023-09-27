# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 23:53:51 2023

@author: Racha
"""

import numpy as np
import matplotlib.pyplot as plt
q = -1.60217663e-19 # electron charge
eps_0 = 8.8541878128e-12 # Vacuum permittivity

a_w = 1.0
lambda_w = 0.05
k_w = 2*np.pi/lambda_w
gamma = 5
c = 299792458

beta_bar = 1 - (1 + a_w**2/2)/(2*gamma**2)


def position(t):
    t = np.asarray(t)  # Convert to numpy array if not already
    r0_x = a_w/(gamma*beta_bar*k_w) * np.sin(k_w*beta_bar*c*t)
    r0_z = beta_bar*c*t - a_w**2/(8*gamma**2 * beta_bar*k_w) * np.sin(2*k_w*beta_bar*c*t)
    # Scalar value (0-dimensional)
    if np.ndim(t) == 0:
        return np.array([r0_x, 0, r0_z]) if t >= 0 else np.array([0, 0, 0])
    
    # Ensure t has at least 2 dimensions
    elif np.ndim(t) == 1:
        t = t[:, np.newaxis]

    pos_x = r0_x
    pos_y = np.zeros_like(t)
    pos_z = r0_z
    
    pos = np.stack((pos_x, pos_y, pos_z), axis=-1)
    mask = t < 0
    pos[mask, 2] = 0
    pos[mask, 1] = 0
    pos[mask, 0] = 0

    # Squeeze out the unnecessary dimension if input was 1D
    return np.squeeze(pos)

def velocity(t):
    t = np.asarray(t)  # Convert to numpy array if not already
    beta_x = a_w/gamma * np.cos(k_w * beta_bar * c * t)
    beta_z = beta_bar - a_w**2/(4*gamma**2) * np.cos(2 * k_w * beta_bar * c * t)
    # Handle scalar value
    if np.ndim(t) == 0:
        return np.array([beta_x, 0, beta_z]) if t >= 0 else np.array([0, 0, 0])
    
    # For 1D and 2D arrays
    vel_x = beta_x
    vel_y = np.zeros_like(t)
    vel_z = beta_z
    
    vel = np.stack((vel_x, vel_y, vel_z), axis=-1)
    mask = t < 0
    vel[mask] = 0

    # Squeeze out the unnecessary dimension if input was 1D
    return np.squeeze(vel)

def acceleration(t):
    t = np.asarray(t)  # Convert to numpy array if not already
    betadot_x = - a_w*beta_bar*c*k_w/gamma * np.sin(k_w * beta_bar * c * t)
    betadot_z = a_w**2 * beta_bar*c*k_w/(2*gamma**2) * np.sin(2* k_w * beta_bar * c * t)
    # Handle scalar value
    if np.ndim(t) == 0:
        return np.array([betadot_x, 0, betadot_z]) if t >= 0 else np.array([0, 0, 0])
    
    # For 1D and 2D arrays
    acc_x = betadot_x
    acc_y = np.zeros_like(t)
    acc_z = betadot_z
    
    acc = np.stack((acc_x, acc_y, acc_z), axis=-1)
    mask = t < 0
    acc[mask] = 0

    # Squeeze out the unnecessary dimension if input was 1D
    return np.squeeze(acc)

def retarded_time(t, R_initial, x_position_func, max_iterations=100, tol=1e-20):
    t_ret_guess = t - R_initial/c
    for _ in range(max_iterations):
        R_ret = np.linalg.norm(r_ - x_position_func(t_ret_guess), axis=-1)
        t_new_guess = t - R_ret/c

        # Check for convergence
        if np.abs(t_new_guess - t_ret_guess).max() < tol:
            break
        t_ret_guess = t_new_guess

    return t_ret_guess

    
x = np.linspace(-1.5*lambda_w, 1.5*lambda_w, 500)
z = np.linspace(0*lambda_w, 6.0*lambda_w, 1000)
Z, X = np.meshgrid(z, x)  # Note the swapped order
Y = np.zeros_like(X)

r_ = np.dstack((X, Y, Z))

N_w = np.linspace(0, 6, 1200)
for index, value in enumerate(N_w):
    t_current = value * 2*np.pi/(beta_bar*c*k_w)
    R_ = r_ - position(t_current) # initial guess of vector(R) from source -> observer at the current time
    R = np.linalg.norm(R_, axis=-1) # |R_|
    # n_hat = R_/R[:,:,None]

    t_retarded = t_current - R/c

    t_r = retarded_time(t_current, R, position)
    Rr_ = r_ - position(t_r) # retarded position
    Rr = np.linalg.norm(Rr_, axis=-1) # |Rr_|
    n_hat = Rr_/Rr[:,:,None] # unit vector
    Inverse_Rr = 1.0/Rr # 1/R

    beta_ = velocity(t_r)
    betadot_ = acceleration(t_r)

    Inverse_one_s_nhat_dot_beta_3 = 1.0/(1.0 - np.einsum('ijk,ijk->ij', n_hat, beta_))**3

    nhat_s_beta = n_hat - beta_
    one_s_beta2 = 1.0 - (np.linalg.norm(beta_, axis=-1))**2
    firstterm = nhat_s_beta* one_s_beta2[:,:,None] *  Inverse_one_s_nhat_dot_beta_3[:,:,None] * Inverse_Rr[:,:,None] * Inverse_Rr[:,:,None]

    nhat_cross_nhat_s_beta_cross_betadot = np.cross(n_hat, np.cross(nhat_s_beta, betadot_))
    secondterm = nhat_cross_nhat_s_beta_cross_betadot * Inverse_one_s_nhat_dot_beta_3[:,:,None] * Inverse_Rr[:,:,None]  / c

    E_ = q*(firstterm + secondterm)

    E_x = E_[:,:,0]
    E_z = E_[:,:,2]

    E_mag = np.linalg.norm(E_, axis=-1)
    E_mag_log = np.log10(E_mag)
    max_norm = np.percentile(E_mag_log, 99.85) 
    min_norm = np.percentile(E_mag_log, 5.0) 

    E_hat = E_/E_mag[:,:, None]

    u = E_hat[:,:,0]  # X component
    v = E_hat[:,:,2]  # Z component

    stride = 10
    # Extracting the amplitude values for the subsampled data
    amplitude_subsampled = np.log10(E_mag[::stride, ::stride])
    min_val = -17.806363778167935
    max_val = -13.419803961370096
    amplitude_subsampled = np.clip(amplitude_subsampled, min_val, max_val)
    alpha_values = (amplitude_subsampled - min_val) / (max_val - min_val)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_facecolor('black')
    contour = plt.contourf(Z, X, np.log10(E_mag), cmap='magma', levels=512, vmax=-13.419803961370096, vmin=-17.806363778167935)
    for i in np.linspace(min(z), max(z), 7):  
        ax.vlines(i,min(x),max(x)  , colors='grey', linestyles='-', linewidths=0.2)
 
    for i in np.linspace(min(x), max(x), 4):
        ax.hlines(i,min(z),max(z)  , colors='grey', linestyles='-', linewidths=0.2)

    ax.set_xlim(min(z),max(z))
    ax.set_ylim(min(x),max(x))
    ax.axis('equal')
    ax.quiver(Z[::stride, ::stride],
          X[::stride, ::stride], 
          v[::stride, ::stride], 
          u[::stride, ::stride], 
          angles='xy', 
          scale_units='xy', 
          scale=320, 
          color='white',
          alpha=alpha_values,
          width=0.0015,
          headlength=4,  # Adjust as needed
          headwidth=2.5,   # Adjust as needed
          headaxislength=4)  # Adjust as needed
    ax.scatter(position(t_current)[2], position(t_current)[0], color='tab:green', s = 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"output_{index}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"output_{index}.png")
    plt.close() 