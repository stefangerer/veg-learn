import numpy as np

def safe_divide(a, b):
    """Safely divide two arrays, avoiding division by zero and handling NaN values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = 0  # Replace infinity and NaNs with 0
    return result

def calculate_ndvi(nir_band, red_band):
    return safe_divide(nir_band - red_band, nir_band + red_band + 1e-10)

def calculate_savi(nir_band, red_band, L=0.5):
    return safe_divide((nir_band - red_band) * (1 + L), nir_band + red_band + L + 1e-10)

def calculate_psri(red_band,nir_band):
    return safe_divide(red_band - nir_band, red_band)

def calculate_gndvi(nir_band, green_band):
    return safe_divide((nir_band - green_band), (nir_band + green_band))

def calculate_ndre(nir_band, red_edge_band):
    return safe_divide((nir_band - red_edge_band), (nir_band + red_edge_band))

def calculate_vari(green_band, red_band, blue_band): 
    return safe_divide((green_band - red_band), (green_band + red_band - blue_band))

def calculate_sr(nir_band, red_band):
    return safe_divide(nir_band, red_band)

def calculate_mcari(vnir_band, red_band, green_band):
    return (vnir_band - red_band) - 0.2 * (vnir_band - green_band) * (vnir_band / red_band)

def calculate_ndgi(green_band, red_band): 
    return safe_divide((green_band - red_band), (green_band + red_band))

def calculate_pri(blue_band, green_band):
    return safe_divide(blue_band - green_band, blue_band + green_band)





