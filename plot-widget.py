import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre, factorial
import ipywidgets as widgets
from IPython.display import display

%matplotlib inline
%matplotlib widget

def plot_hydrogen_orbital_slice(n, l, m, axis='z', grid_size=100):
    """
    Interactive plot of a 2D slice of the probability density for a hydrogen atom's orbital.

    Parameters:
        n (int): Principal quantum number (n > 0)
        l (int): Orbital angular momentum quantum number (0 <= l < n)
        m (int): Magnetic quantum number (-l <= m <= l)
        axis (str): Axis along which to take the slice ('x', 'y', or 'z')
        grid_size (int): Size of the spatial grid
    """
    # Define range for coordinates
    x = np.linspace(-20, 20, grid_size)
    y = np.linspace(-20, 20, grid_size)
    z = np.linspace(-20, 20, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Convert Cartesian coordinates to spherical
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(Z / R)
    Phi = np.arctan2(Y, X)
    
    # Avoid division by zero at the origin
    R[R == 0] = np.finfo(float).eps
    
    # Radial part
    rho = 2 * R / n
    radial = np.exp(-rho/2) * rho**l
    L = genlaguerre(n-l-1, 2*l+1)
    radial *= L(rho)
    radial *= (2 / n)**3 * np.sqrt(factorial(n-l-1) / (2*n*factorial(n+l)))

    # Angular part
    Ylm = sph_harm(m, l, Phi, Theta)
    
    # Total wavefunction
    psi = radial * Ylm
    probability_density = np.abs(psi)**2

    def update_plot(slice_index):
        plt.clf()
        if axis == 'x':
            slice_data = probability_density[slice_index, :, :]
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 'y':
            slice_data = probability_density[:, slice_index, :]
            xlabel, ylabel = 'X', 'Z'
        else:
            slice_data = probability_density[:, :, slice_index]
            xlabel, ylabel = 'X', 'Y'
        
        plt.contourf(x, y, slice_data, levels=50, cmap='viridis')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()

    # Slider widget
    slice_slider = widgets.IntSlider(min=0, max=grid_size-1, step=1, value=grid_size//2, description='Slice Index')
    widgets.interact(update_plot, slice_index=slice_slider)

# Example usage: Plotting a slice of the 3d orbital along Z interactively
plot_hydrogen_orbital_slice(n=3, l=2, m=0, axis='z')
