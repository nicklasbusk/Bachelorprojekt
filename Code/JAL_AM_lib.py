import numpy as np
from model_lib import *

@njit
def update_AV(AV, s, Q, pi_other, action_space):
    """
    Args:
        AV: The action-value matrix to update
        s: The current state
        Q: The Q-value table
        pi_other: The estimated policy of the other agents
        action_space: The action space for the other agents
    """
    for a_i in range(len(action_space)):
        expected_value = 0.0  # Ensure expected_value is a scalar
        for a_j in range(len(action_space)):  # Iterate over actions of the other agent
            joint_action = (a_i, a_j)
            prob = pi_other[s, a_j]
            expected_value += Q[s, joint_action[0], joint_action[1]] * prob  # Handle joint_action indexing correctly
        AV[s, a_i] = expected_value
    return AV



@njit
def update_agent_model(agent_model, state, action, counter,k):
    """
    args:
        state: the current state of player j
        action: action of player j
        counter: array holding count of every action played in each state
    returns:
        counter: the updated counter
        agent_model: the updated agent model
    """
    counter[state, action] += 1
    # måske skal man opdatere alle agent models af gangen og ikke kun 1
    for i in range(k):
        agent_model[state,i] = counter[state, i] / np.sum(counter[state,:])
    #print(agent_model)
    return counter, agent_model

@njit
def select_price(s_t_idx, price_grid, epsilon, AV):
    if epsilon >= np.random.uniform(0, 1):
        return np.random.choice(price_grid)
    else:
        maxedQ_idx = np.argmax(AV[s_t_idx, :])
        return price_grid[maxedQ_idx]

@njit
def Q_func(p_curr_idx, s_curr_idx, i, j, t, alpha, gamma, p_table, Q_table, price_grid, s_next, pi_other, T, AV):
    prev_est = Q_table[s_curr_idx, p_curr_idx]
    s_next_index = np.where(price_grid == s_next)[0][0]
    Max_AV_idx = np.argmax(AV[s_next_index, :])
    Max_AV = AV[s_next_index, Max_AV_idx]
    reward = profit(p_table[i, t], p_table[j, t-2]) + gamma * profit(p_table[i, t], s_next) + gamma**2 * Max_AV
    return (1 - alpha) * prev_est + alpha * reward
@njit
def JAL_AM(alpha, gamma, T, price_grid):
    epsilon = calculate_epsilon(T)
    i, j = 0, 1
    t = 0
    k = len(price_grid)
    q1, q2 = np.zeros((k, k, k)), np.zeros((k, k, k))  # Adjust dimensions to handle joint actions
    Agent_model_1, Agent_model_2 = np.ones((k, k)) / k, np.ones((k, k)) / k
    AV_1, AV_2 = np.ones((k, k)), np.ones((k, k))
    N1, N2 = np.zeros((k, k)), np.zeros((k, k))
    p_table = np.zeros((2, T))
    profits = np.zeros((2, T))
    avg_profs1, avg_profs2 = [], []

    p_table[i, t] = np.random.choice(price_grid)
    p_table[j, t] = np.random.choice(price_grid)
    t += 1
    p_table[i, t] = np.random.choice(price_grid)
    p_table[j, t] = np.random.choice(price_grid)
    t += 1
    
    for t in range(t, T):
        p_table[i, t] = p_table[i, t-1]
        p_idx = np.where(price_grid == p_table[i, t])[0][0]
        s_next = p_table[j, t-1]
        current_state_idx = np.where(price_grid == p_table[j, t-2])[0][0]
        
        N2, Agent_model_2 = update_agent_model(Agent_model_2, p_idx, current_state_idx, N2, k)
        AV_1 = update_AV(AV_1, current_state_idx, q1, Agent_model_2, price_grid)
        q1[current_state_idx, p_idx] = Q_func(p_idx, current_state_idx, i, j, t, alpha, gamma, p_table, q1, price_grid, s_next, Agent_model_2, T, AV_1)
        
        s_next_idx = np.where(price_grid == p_table[j, t-1])[0][0]
        p_table[i, t] = select_price(s_next_idx, price_grid, epsilon[t], AV_1)
        p_table[j, t] = p_table[j, t-1]
        
        profits[i, t] = profit(p_table[i, t], p_table[j, t])
        profits[j, t] = profit(p_table[j, t], p_table[i, t])
        
        if t % 1000 == 0:
            avg_profs1.append(np.sum(profits[i, (t-1000):t]) / 1000)
            avg_profs2.append(np.sum(profits[j, (t-1000):t]) / 1000)
        
        i, j = j, i
        q1, q2 = q2, q1
        Agent_model_1, Agent_model_2 = Agent_model_2, Agent_model_1
        N1, N2 = N2, N1
        AV_1, AV_2 = AV_2, AV_1
    
    return p_table, avg_profs1, avg_profs2

"""
@njit
def update_agent_model(agent_model, state, action, counter,k):
    """
"""
    args:
        state: the current state of player j
        action: action of player j
        counter: array holding count of every action played in each state
    returns:
        counter: the updated counter
        agent_model: the updated agent model
    """
"""
    counter[state, action] += 1
    # måske skal man opdatere alle agent models af gangen og ikke kun 1
    for i in range(k):
        agent_model[state,i] = counter[state, i] / np.sum(counter[state,:])
    #print(agent_model)
    return counter, agent_model

@njit
def select_price(s_t_idx, price_grid, epsilon, AV):
    """
"""  args:
        s_t_idx: current state index
        price_grid: price_grid
        epsilon: decay parameter of learning module
        AV: action value
    returns:
        random price or maximized price
    """
"""
    # Exploration
    #epsilon=0.1
    if epsilon >= np.random.uniform(0,1):
        return np.random.choice(price_grid)
    else:
    # Exploitation
        maxedQ_idx=np.argmax(AV[s_t_idx, :])    
        return price_grid[maxedQ_idx]
    
@njit
def Q_func(p_curr_idx, s_curr_idx, i, j, t, alpha, gamma, p_table, Q_table, price_grid, s_next,pi_other,T,AV) -> float: # p_table contains p and s (opponent price)
    """
"""
    args:
        p_curr_idx: current price of player i
        s_curr_idx: current state of player i
        i: player 0
        j: player 1
        t: current period
        alpha: step-size parameter
        gamma: discount factor
        p_table: 2x500.000 array storing prices for player 0 and 1
        Q_table: current Q_table for player i
        price_grid: price_grid
        s_next: next state for player i
    returns:
        updated value for Q_table 
    """
"""
    prev_est = Q_table[p_curr_idx, s_curr_idx]
    s_next_index=np.where(price_grid == s_next)[0][0]
    
    # Beregn de forventede Q-værdier ved at bruge AV
    #AV = calculate_AV(Q_table, pi_other, s_next_index,7)
    
    # Vælg den handling for næste tilstand med den højeste AV-værdi
    Max_AV_idx = np.argmax(AV[s_next_index,:])
    Max_AV=price_grid[Max_AV_idx]
    
    # Beregn den forventede fremtidige værdi ved at bruge den valgte AV-værdi
    
    reward = profit(p_table[i, t], p_table[j, t-2]) + gamma * profit(p_table[i, t], s_next) + gamma**2 * Max_AV
    return (1 - alpha) * prev_est + alpha * reward

@njit
def JAL_AM(alpha, gamma, T, price_grid):
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
    Agent_model_1 = np.ones((k, k)) / k
    Agent_model_2 = np.ones((k, k)) / k
    AV_1= np.ones((k,k))
    AV_2= np.ones((k,k))

    # Initializing N, a counter
    N1 = np.zeros((k, k))
    N2 = np.zeros((k, k))
    
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
    for t in range(t,T):
        p_table[i,t] = p_table[i,t-1]
        p_idx = np.where(price_grid == p_table[i,t])[0][0]
        s_next = p_table[j,t-1]
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        # opdatering af q skal vel flyttes længere ned?
        #N2, Agent_model_2= update_agent_model(Agent_model_2,current_state_idx, np.where(price_grid==s_next)[0][0], N2,k)
        #N2, Agent_model_2= update_agent_model(Agent_model_2,p_idx, np.where(price_grid==s_next)[0][0], N2,k)
        N2, Agent_model_2= update_agent_model(Agent_model_2,p_idx, current_state_idx, N2,k)
        #N2, Agent_model_2= update_agent_model(Agent_model_2,current_state_idx, current_state_idx, N2,k)


        AV_1=update_AV(AV_1,current_state_idx,q1,Agent_model_2)
        q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next, Agent_model_2,T, AV_1)
        s_next_idx=np.where(price_grid==p_table[j,t-1])[0][0]
        p_table[i,t] = select_price( s_next_idx, price_grid, epsilon[t],AV_1)
        p_table[j, t] = p_table[j, t-1]
        

        
        # Store profits for both firms
        profits[i, t] = profit(p_table[i,t], p_table[j,t])
        profits[j, t] = profit(p_table[j,t], p_table[i,t])

        
        
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        
        # changing agents
        #print(Agent_model_1)
        tmp = i
        i = j
        j = tmp
        tmp=q1
        q1=q2
        q2=tmp
        Agent_model_1, Agent_model_2 = Agent_model_2, Agent_model_1
        N1,N2=N2,N1
        AV_1,AV_2=AV_2,AV_1
    return p_table, avg_profs1, avg_profs2
    """

def run_sim(n, k):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
    """
    num_calcs=int(500000/1000-1) # size of avg. profits 
    summed_avg_profitabilities = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    t=n
    # simulating n runs of JAL-AM
    for n in range(0, n):
        p_table, avg_profs1, avg_profs2 = JAL_AM(0.3, 0.95, 500000, k)
        per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0)/2
        avg_prof_gain[n] = per_firm_profit[498]/0.125
        summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)

    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, t



"""@njit
def select_price_asym(true_state, AV, price_grid, epsilon, mu):
    """
"""
    args:
        j: player 1
        t: current period
        p_table: 2x500.000 array storing prices for player 0 and 1
        Q_table: current Q_table
        price_grid: price_grid
        epsilon: decay parameter of learning module
        mu: probability of observing wrong price
    returns:
        random price or maximized price
    """
""" 
    if mu<=np.random.uniform(0,1):
        s_t_idx=true_state 
    else:
        s_t_idx=np.where(price_grid==np.random.choice(price_grid))[0][0]
    
    # Exploration
    if epsilon >= np.random.uniform(0,1):
        return np.random.choice(price_grid)
    else:
    # Exploitation
        maxedQ_idx=np.argmax(AV[s_t_idx,:])
        return price_grid[maxedQ_idx]


@njit
def JAL_AM_Asym(alpha, gamma, T, price_grid, mu):
"""
"""
    args:
        alpha: step-size parameter
        gamma: discount factor
        T: number of simulations
        price_grid: price grid
        mu: probability of observing wrong price
    returns:
        p_table: 2x500.000 array storing all prices set
        avg_profs1: average profit firm 1
        avg_profs2: average profit firm 2
    """
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
    Agent_model_1 = np.ones((k, k)) / k
    Agent_model_2 = np.ones((k, k)) / k
    AV_1= np.ones((k,k))
    AV_2= np.ones((k,k))

    # Initializing N, a counter
    N1 = np.zeros((k, k))
    N2 = np.zeros((k, k))
    
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
    for t in range(t,T):
        if t%2!=0:
            p_table[i,t] = p_table[i,t-1]
            p_idx = np.where(price_grid == p_table[i,t])[0][0]
            s_next = p_table[j,t-1]
            current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
            # opdatering af q skal vel flyttes længere ned?
            #N2, Agent_model_2= update_agent_model(Agent_model_2,current_state_idx, np.where(price_grid==s_next)[0][0], N2,k)
            #N2, Agent_model_2= update_agent_model(Agent_model_2,p_idx, np.where(price_grid==s_next)[0][0], N2,k)
            #N2, Agent_model_2= update_agent_model(Agent_model_2,p_idx, current_state_idx, N2,k)
            N2, Agent_model_2= update_agent_model(Agent_model_2,p_idx, current_state_idx, N2,k)
            #AV_1=update_AV(AV_1,np.where(price_grid==s_next)[0][0],q1,Agent_model_2,current_state_idx,k)
            AV_1=update_AV(AV_1,current_state_idx,q1,Agent_model_2)
            q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next, Agent_model_2,T, AV_1)
            s_next_idx=np.where(price_grid==p_table[j,t-1])[0][0]
            #p_table[i,t] = select_price( s_next_idx, price_grid, epsilon[t], Agent_model_2,k,AV_1)
            p_table[i,t] = select_price(s_next_idx, price_grid, epsilon[t],AV_1)
            p_table[j, t] = p_table[j, t-1]
            # Store profits for both firms
            profits[i, t] = profit(p_table[i,t], p_table[j,t])
            profits[j, t] = profit(p_table[j,t], p_table[i,t])
        else:
            p_table[j,t] = p_table[j,t-1]
            p_idx = np.where(price_grid == p_table[j,t])[0][0]
            s_next = p_table[i,t-1]
            current_state_idx = np.where(price_grid == p_table[i,t-2])[0][0]
            N1, Agent_model_1= update_agent_model(Agent_model_1,p_idx, current_state_idx, N1,k)
            AV_2=update_AV(AV_2,current_state_idx,q2,Agent_model_1)
            q2[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, j,i, t, alpha, gamma, p_table, q2, price_grid, s_next, Agent_model_1,T, AV_2)
            s_next_idx=np.where(price_grid==p_table[i,t-1])[0][0]
            p_table[j,t] = select_price_asym(s_next_idx,AV_2, price_grid, epsilon[t],mu)
            p_table[i, t] = p_table[i, t-1]
            # Store profits for both firms
            profits[i, t] = profit(p_table[i,t], p_table[j,t])
            profits[j, t] = profit(p_table[j,t], p_table[i,t])
        
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        # changing agents
        
    return p_table, avg_profs1, avg_profs2
"""


@njit
def select_price_asym(true_state, price_grid, epsilon, AV, mu):
    if mu <= np.random.uniform(0, 1):
        s_t_idx = true_state 
    else:
        s_t_idx = np.random.randint(len(price_grid))
    
    if epsilon >= np.random.uniform(0, 1):
        return np.random.choice(price_grid)
    else:
        maxedQ_idx = np.argmax(AV[s_t_idx, :])
        return price_grid[maxedQ_idx]
@njit
def JAL_AM_asym(alpha, gamma, T, price_grid, mu):
    epsilon = calculate_epsilon(T)
    i, j = 0, 1
    t = 0
    k = len(price_grid)
    q1, q2 = np.zeros((k, k, k)), np.zeros((k, k, k))  # Adjust dimensions to handle joint actions
    Agent_model_1, Agent_model_2 = np.ones((k, k)) / k, np.ones((k, k)) / k
    AV_1, AV_2 = np.ones((k, k)), np.ones((k, k))
    N1, N2 = np.zeros((k, k)), np.zeros((k, k))
    p_table = np.zeros((2, T))
    profits = np.zeros((2, T))
    avg_profs1, avg_profs2 = [], []

    p_table[i, t] = np.random.choice(price_grid)
    p_table[j, t] = np.random.choice(price_grid)
    t += 1
    p_table[i, t] = np.random.choice(price_grid)
    p_table[j, t] = np.random.choice(price_grid)
    t += 1
    
    for t in range(t, T):
        if t % 2 != 0:
            p_table[i, t] = p_table[i, t-1]
            p_idx = np.where(price_grid == p_table[i, t])[0][0]
            s_next = p_table[j, t-1]
            current_state_idx = np.where(price_grid == p_table[j, t-2])[0][0]
            
            N2, Agent_model_2 = update_agent_model(Agent_model_2, p_idx, current_state_idx, N2, k)
            AV_1 = update_AV(AV_1, current_state_idx, q1, Agent_model_2, price_grid)
            q1[current_state_idx, p_idx] = Q_func(p_idx, current_state_idx, i, j, t, alpha, gamma, p_table, q1, price_grid, s_next, Agent_model_2, T, AV_1)
            
            s_next_idx = np.where(price_grid == p_table[j, t-1])[0][0]
            p_table[i, t] = select_price_asym(s_next_idx, price_grid, epsilon[t], AV_1, mu)
            p_table[j, t] = p_table[j, t-1]
            
            profits[i, t] = profit(p_table[i, t], p_table[j, t])
            profits[j, t] = profit(p_table[j, t], p_table[i, t])
        else:
            p_table[j, t] = p_table[j, t-1]
            p_idx = np.where(price_grid == p_table[j, t])[0][0]
            s_next = p_table[i, t-1]
            current_state_idx = np.where(price_grid == p_table[i, t-2])[0][0]
            
            N1, Agent_model_1 = update_agent_model(Agent_model_1, p_idx, current_state_idx, N1, k)
            AV_2 = update_AV(AV_2, current_state_idx, q2, Agent_model_1, price_grid)
            q2[current_state_idx, p_idx] = Q_func(p_idx, current_state_idx, j, i, t, alpha, gamma, p_table, q2, price_grid, s_next, Agent_model_1, T, AV_2)
            
            s_next_idx = np.where(price_grid == p_table[i, t-1])[0][0]
            p_table[j, t] = select_price_asym(s_next_idx, price_grid, epsilon[t], AV_2, mu)
            p_table[i, t] = p_table[i, t-1]
            
            profits[i, t] = profit(p_table[i, t], p_table[j, t])
            profits[j, t] = profit(p_table[j, t], p_table[i, t])
            
        if t % 1000 == 0:
            avg_profs1.append(np.sum(profits[i, (t-1000):t]) / 1000)
            avg_profs2.append(np.sum(profits[j, (t-1000):t]) / 1000)
    
    return p_table, avg_profs1, avg_profs2


def run_sim_asym(n, k, mu):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
    """
    
    num_calcs = int(500000 / 1000 - 1)  # size of avg. profits 
    summed_profit1=np.zeros(num_calcs)
    summed_profit2=np.zeros(num_calcs)
    summed_avg_profitabilities = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    t = n
    
    # simulating n runs of JAL-AM
    for i in range(n):
        p_table, avg_profs1, avg_profs2 = JAL_AM_asym(0.3, 0.95, 500000, k, mu)
        per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
        avg_prof_gain[i] = per_firm_profit[-1] / 0.125  # Adjust index to access the last element correctly
        summed_profit1=np.sum([summed_profit1,avg_profs1],axis=0)
        summed_profit2=np.sum([summed_profit2,avg_profs2],axis=0)
        summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
    res1=np.divide(summed_profit1, n)
    res2=np.divide(summed_profit2, n)
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, t, res1, res2