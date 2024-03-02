import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numba import njit, prange
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

@njit
def calculate_epsilon(T):
    """
    args
        T: learning duration
    returns
        A list of T epsilon values
    """
    epsilon_values = []
    
    for i in range(T):
        theta = -((1/1000000) ** (1/T)) + 1
        epsilon = (1 - theta) ** i
        epsilon_values.append(epsilon)
    
    return epsilon_values

@njit
def demand(p1t,p2t):
    """
    args:
        p1t: price of agent 1
        p2t: price of agent 2
    returns:
        d: demand for given set of prices
    """
    if p1t < p2t:
        d = 1 - p1t
    elif p1t == p2t:
        d = 0.5 * (1 - p1t)
    else:
        d = 0
    return d

@njit
def profit(p1t, p2t):
    """
    args:
        p1t: price of agent 1
        p2t: price of agent 2
    returns:
        profit for agent
    """
    return p1t * demand(p1t, p2t)
 
@njit
def Q_func(p_curr_idx, s_curr_idx, i, j, t, alpha, delta, p_table, Q_table, price_grid, s_next) -> float: # p_table contains p and s (opponent price)
    """
    args:
        p_curr_idx: current price of player i
        s_curr_idx: current state of player i
        i: player 0
        j: player 1
        t: current period
        alpha: step-size parameter
        delta: discount factor
        p_table: 2x500.000 array storing prices for player 0 and 1
        Q_table: current Q_table for player i
        price_grid: price_grid
        s_next: next state for player i
    returns:
        updated value for Q_table 
    """
    prev_est = Q_table[p_curr_idx, s_curr_idx]
    s_next_index=np.where(price_grid == s_next)[0][0] 
    maxed_Q = max(Q_table[:, s_next_index])
    new_est = profit(p_table[i, t], p_table[j, t]) + delta * profit(p_table[i, t], s_next) + delta**2 * maxed_Q
    return (1 - alpha) * prev_est + alpha * new_est

    
@njit
def select_price(j, t, p_table, Q_table, price_grid, epsilon):
    """
    args:
        j: player 1
        t: current period
        p_table: 2x500.000 array storing prices for player 0 and 1
        Q_table: current Q_table
        price_grid: price_grid
        epsilon: decay parameter of learning module
    returns:
        random price or maximized price
    """
    # Exploration
    if epsilon >= np.random.uniform(0,1):
        return np.random.choice(price_grid)
    else:
    # Exploitation
        s_t_idx = np.where(price_grid == p_table[j, t-1])[0][0] # current state (opponent's price)
        maxedQ_idx = np.argmax(Q_table[:, s_t_idx])
        return price_grid[maxedQ_idx]

        

@njit
def Klein_simulation(alpha, delta, T, price_grid):
    """
    args:
        alpha: step-size parameter
        delta: discount factor
        T: learning duration
        price_grid: price_grid
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
        avg_profs0: player 0 list of average profit for each 1000 period
        avg_profs1: player 1 list of average profit for each 1000 period
    """
    #np.random.seed(3)
    # Initializing values
    epsilon = calculate_epsilon(T)
    i = 0
    j = 1
    t = 0
    # Initializing Q-functions
    p = len(price_grid)
    q1 = np.zeros((p, p)) 
    q2 = np.zeros((p, p)) 

    p_table = np.zeros((2,T))
    profits = np.zeros((2,T))
    avg_profs1 = []
    avg_profs2 = []
    # Setting prices for players in first 2 periods 
    p_table[i, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[j, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[i, t] = p_table[i, t-1]
    t += 1 # now t = 2

    for t in range(t, T):
        p_it_idx = np.where(price_grid == p_table[i, t-2])[0][0]
        s_t_idx =  np.where(price_grid == p_table[j, t-2])[0][0]
        s_next = select_price(i, t, p_table, q2, price_grid, epsilon[t])
        q1[p_it_idx, s_t_idx] = Q_func(p_it_idx, s_t_idx, i,j, t-2, alpha, delta, p_table, q1, price_grid, s_next)
             
        
        p_table[i, t] = select_price(j, t, p_table, q1, price_grid, epsilon[t])
        p_table[j, t] = p_table[j, t-1]

        # Store profits for both firms
        profits[i, t] = profit(p_table[i,t], p_table[j,t])
        profits[j, t] = profit(p_table[j,t], p_table[i,t])

        # compute avg profitability of last 1000 runs for both firms
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        
        # changing agents
        tmp = i
        i = j
        j = tmp
        tmp=q1
        q1=q2
        q2=tmp
        
    return p_table, avg_profs1, avg_profs2

@njit
def Klein_simulation_FD(alpha, delta, T, price_grid):
    """
    args:
        alpha: step-size parameter
        delta: discount factor
        T: learning duration
        price_grid: price_grid
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
        profits: 2x500.000 array, with all profts for player 0 and 1
        avg_profs0: player 0 list of average profit for each 1000 period
        avg_profs1: player 1 list of average profit for each 1000 period
    """
    # Initializing values
    epsilon = calculate_epsilon(T)
    i = 0
    j = 1
    t = 0
    # Initializing Q-functions
    p = len(price_grid)
    q1 = np.zeros((p, p)) 
    q2 = np.zeros((p, p)) 

    p_table = np.zeros((2,T))
    profits = np.zeros((2,T))
    avg_profs1 = []
    avg_profs2 = []
    #avg_price = []
    # Setting prices for players in first 2 periods 
    p_table[i, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[j, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[i, t] = p_table[i, t-1]
    t += 1 # now t = 2

    for t in range(t, T):
        p_it_idx = np.where(price_grid == p_table[i, t-2])[0][0]
        s_t_idx =  np.where(price_grid == p_table[j, t-2])[0][0]
        s_next = select_price(i, t, p_table, q2, price_grid, epsilon[t])
        q1[p_it_idx, s_t_idx] = Q_func(p_it_idx, s_t_idx, i,j, t-2, alpha, delta, p_table, q1, price_grid, s_next)
             
        if t==400000:
            p_table[i,t] = price_grid[s_t_idx - 1]
            #print(p_table[i,t])
        else:
            p_table[i, t] = select_price(j, t, p_table, q1, price_grid, epsilon[t])
        p_table[j, t] = p_table[j, t-1]
        
        #avg_price_t = (p_table[i,t] + p_table[j, t]) / 2
        #avg_price.append(avg_price_t)
        # Store profits for both firms
        profits[i, t] = profit(p_table[i,t], p_table[j,t])
        profits[j, t] = profit(p_table[j,t], p_table[i,t])

        # compute avg profitability of last 1000 runs for both firms
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        
        # changing agents
        tmp = i
        i = j
        j = tmp
        tmp=q1
        q1=q2
        q2=tmp
        
    return p_table, profits, avg_profs1, avg_profs2#, avg_price




def run_sim(runs, k, method='klein'):
    
    num_calcs=int(500000/1000-1)
    summed_avg_profitabilities = np.zeros(num_calcs)

    for n in range(0, runs):
        p_table, avg_profs1, avg_profs2 = Klein_simulation(0.3, 0.95, 500000, k)
        per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0)/2
        summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)

    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, runs)
    return avg_avg_profitabilities

def run_simFD(runs, k):
    
    num_calcs=int(500000/1000-1)
    summed_avg_profitabilities = np.zeros(num_calcs)
    A = np.zeros([0,500000])
    B = np.zeros([0,500000])
    C=np.zeros([0,249999])
    D=np.zeros([0,249999])
    counter = 0
    avg_2period_prof1 = []
    avg_2period_prof2 = []
    cap=1
    #for n in range(0, runs):
    while cap<=177:
        p_table,  profits, avg_profs1, avg_profs2 = Klein_simulation_FD(0.3, 0.95, 500000, k)
        var1 = np.var(p_table[0, 398999:399999])
        var2 = np.var(p_table[1, 398999:399999])
        var = np.mean([var1, var2])
        if var < 0.001:
            per_firm_profit = avg_profs1 #np.sum([avg_profs1, avg_profs2], axis=0)/2
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            A = np.vstack([A,p_table[0,:]])
            B = np.vstack([B,p_table[1,:]])
            counter += 1
            avg_2period_prof1 = []
            avg_2period_prof2 = []
            prof1 = profits[0,:]
            prof2 = profits[1,:]
            for i in range(1,len(prof1)-1, 2):
                avg_2period_prof1.append(prof1[i] + prof1[i+1])
                avg_2period_prof2.append(prof2[i] + prof2[i+1])
            #print(len(avg_2period_prof1))
            C = np.vstack([C,avg_2period_prof1])
            D = np.vstack([D,avg_2period_prof2])
            cap+=1
    #print(A)
    #print("Shape of A: ", A.shape)
    sum_avg_prices = np.sum([])
    new = np.zeros([2, 500000])  # Initialize new list with zeros
    for i in range(499998):
        for j in range(counter):
            new[0,i] += A[j, i]
            new[1,i] += B[j, i]
        new[0,i] /= counter  
        new[1,i]/= counter
    
    
    avg_profits=np.zeros([2,250000])

    for i in range(249999):
        for j in range(cap-1):
            avg_profits[0,i] += C[j, i]
            avg_profits[1,i] += D[j, i]
        avg_profits[0,i] /= cap  
        avg_profits[1,i]/= cap

    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, counter)
    #print("Shape of new: ", new.shape)
    #print(counter)
    return new, avg_avg_profitabilities, avg_2period_prof1, avg_2period_prof2, avg_profits