import numpy as np
import matplotlib.pyplot as plt

# Define the smooth bump function
def smooth_bump_function(x, d_min, d_max):
    if d_min < x < d_max:
        scaled = 2 * (x - (d_min + d_max) / 2) / (d_max - d_min)
        return np.exp(-1 / (1 - scaled**2))
    else:
        return 0
    
def smooth_bump_function2(x, d_min, d_max):
    if d_min < x < d_max:
        scaled = 2 * (x - (d_min + d_max) / 2) / (d_max - d_min)
        return np.exp(-1 / (1 - scaled**2)) / np.exp(-1)
    else:
        return 0
    
# Define the bump function using the user's proposed formula
def smooth_bump_function3(x, d_min, d_max):
    lambda_val = 2 * (x - (d_min + d_max) / 2) / (d_max - d_min)
    if d_min < x < d_max:
        return np.exp(-lambda_val**2 / (1 - lambda_val**2))
    else:
        return 0
    
# Define the bump function using the user's proposed formula (shifted)
def smooth_bump_function4(x, d_min, d_max, d):
    #offset = (d_max - d_min)/2 
    #offset = d - (d_max - d_min)/2 
    #d_min = d - offset
    #d_max = d + offset
    #d_min += offset
    #d_max += offset
    center = d_min + (d_max - d_min)/2 
    if d < center:
        offset = d - d_min
        #d_max = d_max - offset
        #d_max = d_max + offset
        d_max = d + offset
    elif d > center:
        offset = d_max - d
        #d_min = d_min + offset
        #d_min = d - offset
        d_min = d - offset
    lambda_val = 2 * (x - (d_min + d_max) / 2) / (d_max - d_min)
    if d_min < x < d_max:
        return np.exp(-lambda_val**2 / (1 - lambda_val**2))
    else:
        return 0
    
    
# Define the bump function using the user's proposed formula (shifted)
# with flat top

def smooth_bump_function5(x, d_min, d_max, d, p, bias = 'yes'):
    #offset = (d_max - d_min)/2 
    #offset = d - (d_max - d_min)/2 
    #d_min = d - offset
    #d_max = d + offset
    #d_min += offset
    #d_max += offset
    if bias == 'yes':
        center = d_min + (d_max - d_min)/2 
        if d < center:
            offset = d - d_min
            #d_max = d_max - offset
            #d_max = d_max + offset
            d_max = d + offset
        elif d > center:
            offset = d_max - d
            #d_min = d_min + offset
            #d_min = d - offset
            d_min = d - offset
    else:
        pass
            
        
    lambda_val = 2 * (x - (d_min + d_max) / 2) / (d_max - d_min)
    if d_min < x < d_max:
        #return np.exp(-lambda_val**2 / (1 - lambda_val**2))
        return np.exp(-((lambda_val**2 / (1 - lambda_val**2))**p))
    
    else:
        return 0
    
    
# with flat top (no adjust constraints)
def smooth_bump_function6(x, d_min, d_max, d, p):
    #offset = (d_max - d_min)/2 
    #offset = d - (d_max - d_min)/2 
    #d_min = d - offset
    #d_max = d + offset
    #d_min += offset
    #d_max += offset
    # center = d_min + (d_max - d_min)/2 
    # if d < center:
    #     offset = d - d_min
    #     #d_max = d_max - offset
    #     #d_max = d_max + offset
    #     d_max = d + offset
    # elif d > center:
    #     offset = d_max - d
    #     #d_min = d_min + offset
    #     #d_min = d - offset
    #     d_min = d - offset
    lambda_val = 2 * (x - (d_min + d_max) / 2) / (d_max - d_min)
    if d_min < x < d_max:
        #return np.exp(-lambda_val**2 / (1 - lambda_val**2))
        return np.exp(-((lambda_val**2 / (1 - lambda_val**2))**p))
    
    else:
        return 0

# Parameters
d_min   = 5
d       = 12 # desired center
d_max   = 15
#center = (d_max - d_min)/2 + d_min
#p = 6

# Generate x values and compute the function values
x_values = np.linspace(0, 20, 500)
#y_values = np.array([smooth_bump_function3(x, d_min, d_max) for x in x_values])
#y_values_shifted = np.array([smooth_bump_function6(x, d_min, d_max, d) for x in x_values])
#y_values_shifted = np.array([smooth_bump_function5(x, d_min, d_max, d, p) for x in x_values])

# Plot the smooth bump function
plt.figure(figsize=(8, 5))
#plt.plot(x_values, y_values, linestyle='--', label="Bump Function", color='purple')
for p in range(1,6):
    y_values_shifted = np.array([smooth_bump_function5(x, d_min, d_max, d, p, 'no') for x in x_values])
    plt.plot(x_values, y_values_shifted, linestyle='-',label=f'p={p}')
plt.axvline(d, color='green', linestyle=':', label="Desired")
plt.axvline(d_min, color='red', linestyle=':', label="Minimum")
plt.axvline(d_max, color='red', linestyle=':', label="Maximum")
#plt.axvline(center, color='black', linestyle=':', label="Mean")
plt.title("Bump Function")
plt.xlabel("Input ($\hat{d})$")
plt.ylabel("Output")
plt.xlim([d_min-1, d_max+1])
plt.legend()
plt.grid()
plt.show()
