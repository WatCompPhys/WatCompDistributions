import numpy as np

def cartesian_to_spherical(position):
    x, y, z = position
    r = np.linalg.norm(position)
    if r == 0:
        return 0.0, 0.0, 0.0
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi

def radial_function(r): # 1s Hydrogen Orbital
    return r * np.exp(-r)

def spherical_harmonic(theta, phi): # 1s Hydrogen Orbital
    return 1.0 / np.sqrt(4 * np.pi)

def probability_density(position):
    r, theta, phi = cartesian_to_spherical(position)
    
    if r == 0:
        return 0.0  
    
    psi = radial_function(r)
    y = spherical_harmonic(theta, phi)
    
    p = (1.0 / (r**2)) * (np.abs(psi)**2) * (np.abs(y)**2)
    return p

def mhmc_sampler(iterations: int, standard_deviation: float) -> np.ndarray:
    """
    Returns an array of randomly generated spatial positions using the Metropolis-Hastings Monte Carlo method for a 1s hydrogen orbital.
    """
    positions = []

    for i in range(iterations):
        if len(positions) == 0:
            current_position = np.random.uniform(-1.0, 1.0, size=3)
        else:
            previous_position = positions[-1]
            r_prime = np.random.normal(loc=previous_position, scale=standard_deviation, size=3) # Gaussian centered at previous position
            
            r_probability = probability_density(previous_position)
            r_prime_probability = probability_density(r_prime)
            
            if r_probability == 0:
                alpha = 1.0
            else:
                alpha = min(1.0, r_prime_probability / r_probability)
                
            u = np.random.uniform(0, 1)
            
            if u <= alpha:
                current_position = r_prime
            else:
                current_position = previous_position

        positions.append(current_position)

    positions = np.array(positions)

    return positions