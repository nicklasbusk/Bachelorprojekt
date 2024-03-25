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
        epsilon_values: list of T epsilon values, epsilon is a value between 0 and 1 and decreases over time t
    """

    epsilon_values = []
    
    for i in range(T):
        theta = -((1/1000000) ** (1/T)) + 1 # theta being a decay paramter
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
def select_price_WoLF(epsilon, price_grid, current_state, policy):
    """
    args:
        epsilon: epsilon value to period t
        price_grid: grid of prices
        current_state: current state of player
        policy: policy of player
    returns:
        either a random price or price determined by policy
    """
    u = np.random.uniform(0,1)
    if epsilon > u:
        return np.random.choice(price_grid)
    else:
        cumsum = np.cumsum(policy[np.where(price_grid == current_state)[0][0], :])
        idx = np.searchsorted(cumsum, np.array([u]))[0]
        return price_grid[idx]
    

@njit
def q_func_wolf(q, alpha, gamma, p_table, price_grid, i, j,t):
    """
    args:
        q: Q-table of player
        alpha: step-size parameter
        gamma: discount factor
        p_table: 2x500.000 array containing all prices
        i: player i
        j: player j
        t: current period
    returns:
        updated Q-table
    """
    # Update Q-function
    current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
    next_state_idx = np.where(price_grid == p_table[j,t-1])[0][0]
    reward = profit(p_table[i, t-1], p_table[j, t-2]) + gamma * profit(p_table[i, t-1], p_table[j, t-1])
    #max_Q = max(q[current_state_idx, :])
    max_Q = max(q[next_state_idx, :])
    p_idx=np.where(price_grid == p_table[i,t-1])[0][0]
   
    #q[current_state_idx, p_idx] = q[current_state_idx, p_idx] + alpha * (reward + gamma**2 * max_Q - q[current_state_idx, p_idx])
    q[current_state_idx, p_idx] = q[current_state_idx, p_idx] + alpha * (reward + gamma**2 * max_Q - q[current_state_idx, p_idx])
       
    return q

@njit
def update_policy_WoLF(policy, price_grid, delta_l, delta_w, p_table, q, t, N, j, avg_policy, k):
    """
    args:
        policy: policy of player to be updated
        price_grid: grid of prices
        alpha: step-size parameter
        delta_l: learning rate when losing
        delta_w: learning ratwe when winning
        p_table: 2x500.000 array containing all prices
        q: Q-talbe of player
        t: current period
        N: counter matrix 
        j: player j
        avg_policy: averagee policy of player
        k: length of price g
    returns:
        either a random price or price determined by policy
    """
    current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
       
    
    # Update policy
    avg_policy[current_state_idx, :] = avg_policy[current_state_idx, :] + (1 / N[current_state_idx]) * (policy[current_state_idx, :] - avg_policy[current_state_idx, :])
    avg_policy[current_state_idx, :] /= avg_policy[current_state_idx, :].sum() # To ensure a legal probability distribution
            
    if np.sum(q[current_state_idx, :] * policy[current_state_idx, :]) > np.sum(q[current_state_idx, :] * avg_policy[current_state_idx, :]):
        delta_now = delta_w
    else:
        delta_now = delta_l

    
    delta_sa = np.zeros(k) - delta_now / (k - 1)
    p_max_idx = np.argmax(q[current_state_idx, :])
    delta_sa[p_max_idx] = - (delta_sa.sum() - delta_sa[p_max_idx])
    policy[current_state_idx, :] += delta_sa
    policy[current_state_idx, :] = np.minimum(1, np.maximum(policy[current_state_idx, :], 0))
    policy[current_state_idx, :] /= policy[current_state_idx, :].sum()
    
    return policy, N, avg_policy, q


@njit
def WoLF_PHC(alpha, delta_l, delta_w, gamma, price_grid, T):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        delta_l: learning rate when losing
        delta_w: learning rate when winning
        gamma: discount factor
        price_grid: grid of prices
        T: learning duration

    returns:
        avg_profs1: list of average profits of player 1
        avg_profs2: list of average profits of player 2
        p_table: 2xT array of prices
    """
    # Initializing values
    epsilon = calculate_epsilon(T)
    i = 0
    j = 1
    t = 0
    # Initializing Q-functions
    k = len(price_grid)
    q1 = np.zeros((k, k)) 
    q2 = np.zeros((k, k)) 
    # Initializing policies
    policy_1 = np.ones((k, k)) / k
    policy_2 = np.ones((k, k)) / k
    # Initializing average policies
    avg_policy1 = np.ones((k, k)) / k
    avg_policy2 = np.ones((k, k)) / k
    # Initializing N, a counter
    N1 = np.zeros(k)
    N2 = np.zeros(k)
    # Initializing profits
    p_table = np.zeros((2,T))
    profits = np.zeros((2,T))
    avg_profs1 = []
    avg_profs2 = []

    # Setting random price and state for t = 0
    p_table[i,t] = np.random.choice(price_grid)
    p_table[j,t] = np.random.choice(price_grid)
    
    t += 1
    # Setting random price and state for t = 1
    p_table[i,t] = np.random.choice(price_grid)
    p_table[j,t] = np.random.choice(price_grid)
    t += 1
    
    for t in range(t, T-1):
        
        #update Q
        q1=q_func_wolf(q1, alpha, gamma, p_table, price_grid, i, j, t)

        #update N
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        N1[current_state_idx] += 1
        
        #update policy 
        policy_1, N1, avg_policy1, q1 = update_policy_WoLF(policy_1, price_grid, delta_l, delta_w, p_table, q1, t, N1, j, avg_policy1, k)
        
        #update prices
        p_table[i,t] = select_price_WoLF(epsilon[t], price_grid, p_table[j,t-1], policy_1)
        p_table[j,t]= p_table[j,t-1]
        
        #update profits
        profits[i, t] = profit(p_table[i,t], p_table[j,t])
        profits[j, t] = profit(p_table[j,t], p_table[i,t])
        
        # Compute profits
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        
        
        #switching players and their variables
        i,j=j,i
        q1,q2=q2,q1
        policy_1,policy_2=policy_2,policy_1
        avg_policy1,avg_policy2=avg_policy2,avg_policy1
        N1,N2=N2,N1
         
    return avg_profs1, avg_profs2, p_table, policy_1, policy_2

            





def run_sim_wolf(n, k):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
    """
    num_calcs=int(500000/1000-1) # size of avg. profits 
    summed_avg_profitabilities = np.zeros(num_calcs)

    # simulating n runs of WoLF-PHC
    for n in range(0, n):
        avg_profs1, avg_profs2,_,_,_ = WoLF_PHC(0.3, 0.6, 0.2, 0.95, np.linspace(0,1,7), 500000)
        per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0)/2
        summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)

    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities






@njit
def WoLF_PHC_FD(alpha, delta_l, delta_w, gamma, price_grid, T):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        delta_l: learning rate when losing
        delta_w: learning rate when winning
        gamma: discount factor
        price_grid: grid of prices
        T: learning duration

    returns:
        avg_profs1: list of average profits of player 1
        avg_profs2: list of average profits of player 2
        p_table: 2xT array of prices
    """
    # Initializing values
    epsilon = calculate_epsilon(T)
    i = 0
    j = 1
    t = 0
    # Initializing Q-functions
    k = len(price_grid)
    q1 = np.zeros((k, k)) 
    q2 = np.zeros((k, k)) 
    # Initializing policies
    policy_1 = np.ones((k, k)) / k
    policy_2 = np.ones((k, k)) / k
    # Initializing average policies
    avg_policy1 = np.ones((k, k)) / k
    avg_policy2 = np.ones((k, k)) / k
    # Initializing N, a counter
    N1 = np.zeros(k)
    N2 = np.zeros(k)
    # Initializing profits
    p_table = np.zeros((2,T))
    profits = np.zeros((2,T))
    avg_profs1 = []
    avg_profs2 = []

    # Setting random price and state for t = 0
    p_table[i,t] = np.random.choice(price_grid)
    p_table[j,t] = np.random.choice(price_grid)
    
    t += 1
    # Setting random price and state for t = 1
    p_table[i,t] = np.random.choice(price_grid)
    p_table[j,t] = np.random.choice(price_grid)
    t += 1
    
    for t in range(t, T-1):
        
        #update Q
        q1=q_func_wolf(q1, alpha, gamma, p_table, price_grid, i, j, t)

        #update N
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        N1[current_state_idx] += 1
        
        #update policy 
        policy_1, N1, avg_policy1, q1 = update_policy_WoLF(policy_1, price_grid, delta_l, delta_w, p_table, q1, t, N1, j, avg_policy1, k)
        
        #update prices
        if t==499950:
            p_table[i,t] = price_grid[current_state_idx - 1]
        else:
            p_table[i, t] = select_price_WoLF(epsilon[t], price_grid, p_table[j,t-1], policy_1)
        p_table[j, t] = p_table[j, t-1]
        
        #update profits
        profits[i, t] = profit(p_table[i,t], p_table[j,t])
        profits[j, t] = profit(p_table[j,t], p_table[i,t])
        
        # Compute profits
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        
        
        #switching players and their variables
        i,j=j,i
        q1,q2=q2,q1
        policy_1,policy_2=policy_2,policy_1
        avg_policy1,avg_policy2=avg_policy2,avg_policy1
        N1,N2=N2,N1
         
    return profits, avg_profs1, avg_profs2, p_table




def run_sim_wolf_FD(n, k):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
    """
    num_calcs=int(500000/1000-1)
    summed_avg_profitabilities = np.zeros(num_calcs)
    # initialising ???
    A = np.zeros([0,500000])
    B = np.zeros([0,500000])
    C=np.zeros([0,249999])
    D=np.zeros([0,249999])
    counter = 0
    avg_2period_prof1 = []
    avg_2period_prof2 = []
    cap=1
    #for n in range(0, runs):
    while cap <=n:   #while cap<=177:
        profits, avg_profs1, avg_profs2, p_table = WoLF_PHC_FD(0.3, 0.6, 0.2, 0.95, np.linspace(0,1,k), 500000)
        #check to see if the variance of the last 1000 periods is low
        var1 = np.var(p_table[0, 498999:499999])
        var2 = np.var(p_table[1, 498999:499999])
        var = np.mean([var1, var2])
        #if variance is low, we use the average profitabilities, as pricecyles should not occur when for forced deviation
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
            C = np.vstack([C,avg_2period_prof1])
            D = np.vstack([D,avg_2period_prof2])
            cap+=1

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

    return new, avg_avg_profitabilities, avg_2period_prof1, avg_2period_prof2, avg_profits