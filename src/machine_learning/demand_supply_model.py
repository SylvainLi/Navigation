import math
import numpy as np

def supply_drop(beggining_function, ending_function, switch_state, f_0, t):
    arr = [f_0]
    for i in range(1, t):
        if i < switch_state:
            arr += beggining_function(i)
        else:
            arr += [ending_function(i)]
    return np.array(arr, int)

def square_function(a, alpha, beta, t):
    f = lambda x: a*(x-alpha)**2+beta
    return supply_drop(f, f, 0, f(0), t)

def square_drop(beggining_supply, minimal_supply, crisis_duration):
    assert(beggining_supply >= minimal_supply)
    assert(minimal_supply >= 0)
    assert(crisis_duration >= 1)

    a = (beggining_supply-minimal_supply)/((crisis_duration-1)/2)**2
    alpha = (crisis_duration-1)/2
    beta = minimal_supply
    return square_function(a, alpha, beta, crisis_duration)