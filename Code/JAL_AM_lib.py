import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm
from model_lib import *
from concurrent.futures import ProcessPoolExecutor, as_completed

@njit
def edge_or_focal(edge, focal, p_table):
    """
    args
        edge: counter for edgeworth cycles
        focal: counter for focal pricing
        p_table: price table from a simulation
    returns
        edge:counter for edgeworth cycles
        focal: counter for focal pricing
        is_focal: boolean, focal pricing or not
    """
    avg = p_table[0, -50:]
    cycle = False
    for i in range(2, len(avg)):
        if avg[i] != avg[i - 2]:
            cycle = True
            break

    if cycle:
        edge += 1
        is_focal = False
    else:
        focal += 1
        is_focal = True

    return edge, focal, is_focal

@njit
def update_AV(AV, s, Q, pi_other, action_space):
    """
    Args:
        AV: The action-value matrix to update
        s: The current state
        Q: The Q-value table
        pi_other: The estimated policy of the other agents
        action_space: The action space for the other agents
    returns
        AV: action value 
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
    for i in range(k):
        agent_model[state,i] = counter[state, i] / np.sum(counter[state,:]) # updates column values
    return counter, agent_model

@njit
def select_price(s_t_idx, price_grid, epsilon, AV,Q):
    """
    args
        s_t_idx: current state index
        price_grid: price grid
        epsilon: decay parameter of learning module
        AV: action value
    returns
        random price in learning module or optimal price in action module
    """
   
    # exploration
    if epsilon >= np.random.uniform(0, 1):
        return np.random.choice(price_grid)
    # explotation
    else:
        maxedQ_idx = np.argmax(AV[s_t_idx, :])
        return price_grid[maxedQ_idx]

@njit
def Q_func(p_curr_idx, s_curr_idx, i, j, t, alpha, gamma, p_table, Q_table, price_grid, s_next, AV):
    """
    args
        p_curr_idx: current price of player i as index
        s_curr_idx: current state of player i as index
        i: player 0
        j: player 1
        t: current period
        alpha: step-size parameter
        gamma: discount factor
        p_table: 2x500.000 array storing prices for player 0 and 1
        Q_table: current Q_table for player i
        price_grid: price_grid
        s_next: next state for player i
        AV: action value
    returns:
        updated value for Q_table 
    """
    prev_est = Q_table[s_curr_idx, p_curr_idx]
    s_next_index = np.where(price_grid == s_next)[0][0]
    Max_AV_idx = np.argmax(AV[s_next_index, :])
    Max_AV = AV[s_next_index, Max_AV_idx]
    reward = profit(p_table[i, t], p_table[j, t-2]) + gamma * profit(p_table[i, t], s_next) + gamma * Max_AV
    return (1 - alpha) * prev_est + alpha * reward

@njit
def JAL_AM(alpha, gamma, T, price_grid):
    """
    args
        alpha: step-size parameter
        gamma: discount factor
        T: number of runs
        price_grid: price grid
    returns
        p_table: 2x500.000 array storing prices for player 0 and 1
        avg_profs1: average profitabilities for player 1
        avg_profs2: average profitabilities for player 2
    """
    # initializing parameter values
    epsilon = calculate_epsilon(T)
    i, j = 0, 1
    t = 0
    k = len(price_grid)
    # initializing  q-tables, agent models, action values and counters 
    q1, q2 = np.zeros((k, k, k)), np.zeros((k, k, k))  # Adjust dimensions to handle joint actions
    Agent_model_1, Agent_model_2 = np.ones((k, k)) / k, np.ones((k, k)) / k
    AV_1, AV_2 = np.ones((k, k)), np.ones((k, k))
    N1, N2 = np.zeros((k, k)), np.zeros((k, k))

    p_table = np.zeros((2, T))
    profits = np.zeros((2, T))
    avg_profs1, avg_profs2 = [], []
    # setting prices for players in first 2 periods 
    p_table[i, t] = np.random.choice(price_grid)
    p_table[j, t] = np.random.choice(price_grid)
    t += 1
    p_table[i, t] = np.random.choice(price_grid)
    p_table[j, t] = np.random.choice(price_grid)
    t += 1 # now t = 2
    
    for t in range(t, T):
        # updating counter, action value and q-table
        p_table[i, t] = p_table[i, t-1] 
        p_idx = np.where(price_grid == p_table[i, t])[0][0] # p_it-1
        s_next = p_table[j, t-1] # p_jt-1
        current_state_idx = np.where(price_grid == p_table[j, t-2])[0][0] # p_jt-2

        N2, Agent_model_2 = update_agent_model(Agent_model_2, p_idx, current_state_idx, N2, k)
        AV_1 = update_AV(AV_1, current_state_idx, q1, Agent_model_2, price_grid)
        q1[current_state_idx, p_idx] = Q_func(p_idx, current_state_idx, i, j, t, alpha, gamma, p_table, q1, price_grid, s_next, AV_1)
        # setting price
        s_next_idx = np.where(price_grid == p_table[j, t-1])[0][0]
        p_table[i, t] = select_price(s_next_idx, price_grid, epsilon[t], AV_1,q1)
        p_table[j, t] = p_table[j, t-1]
        # store profits for both firms
        profits[i, t] = profit(p_table[i, t], p_table[j, t])
        profits[j, t] = profit(p_table[j, t], p_table[i, t])
        # compute avg profitability of last 1000 runs for both firms
        if t % 1000 == 0:
            avg_profs1.append(np.sum(profits[i, (t-1000):t]) / 1000)
            avg_profs2.append(np.sum(profits[j, (t-1000):t]) / 1000)
        # changing agents
        i, j = j, i
        q1, q2 = q2, q1
        Agent_model_1, Agent_model_2 = Agent_model_2, Agent_model_1
        N1, N2 = N2, N1
        AV_1, AV_2 = AV_2, AV_1
    
    return p_table, avg_profs1, avg_profs2

def run_sim_JALAM_single_run(alpha, gamma, T, price_grid):
    p_table, avg_profs1, avg_profs2 = JAL_AM(alpha, gamma, T, price_grid)
    per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
    return p_table, avg_profs1, avg_profs2, per_firm_profit

def run_sim_JAL_AM(n, k, show_progress=False):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
        avg_prof_gain: list containing average profit gains of runs
        edge: number of times simulations resulted in Edgeworth price cycle
        focal: number of times simulations resulted in focal price
    """
    num_calcs = int(500000 / 1000 - 1)
    summed_avg_profitabilities = np.zeros(num_calcs)
    summed_profit1 = np.zeros(num_calcs)
    summed_profit2 = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    focal = 0
    edge = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_sim_JALAM_single_run, 0.3, 0.95, 500000, k) for _ in range(n)]
        if show_progress:
            iterator = tqdm(enumerate(as_completed(futures)), total=n)
        else:
            iterator = enumerate(as_completed(futures))
        
        for i, future in iterator:
            p_table, avg_profs1, avg_profs2, per_firm_profit = future.result()
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            summed_profit1 = np.sum([summed_profit1, avg_profs1], axis=0)
            summed_profit2 = np.sum([summed_profit2, avg_profs2], axis=0)
            avg_prof_gain[i] = per_firm_profit[498] / 0.125
            edge, focal, isfocal = edge_or_focal(edge, focal, p_table)
            
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal

# ASYMMETRIC INFORMATION
@njit
def edge_or_focal_asym(edge, focal, p_table, mu, periods):
    tolerance = mu * periods * 1.5
    avg = p_table[0, -periods:]
    cycle = False
    deviations = 0

    for i in range(2, len(avg)):
        if avg[i] != avg[i-2]:
            deviations += 1
            if deviations > tolerance:
                cycle = True
                break
    if cycle:
        edge += 1
        is_focal = False
    else:
        focal += 1
        is_focal = True
    return edge, focal, is_focal

@njit
def select_price_asym(true_state, price_grid, epsilon, AV, mu,Q):
    """
    args
        true_state: true price set by competitor
        price_grid: price grid
        epsilon: decay parameter of learning module
        AV: action value
        mu: probability of observing true state
    returns
        random price in learning module or optimal price in action module
    """
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
    """
    args
        alpha: step-size parameter
        gamma: discount factor
        T: number of runs in simulation
        price_grid: price grid
        mu: probability of observing true state
    returns
        p_table: 2x500.000 array storing prices for player 0 and 1
        avg_profs1: average profitabilities for player 1
        avg_profs2: average profitabilities for player 2
    """
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
            q1[current_state_idx, p_idx] = Q_func(p_idx, current_state_idx, i, j, t, alpha, gamma, p_table, q1, price_grid, s_next, AV_1) # Agent_model_2, T
            
            s_next_idx = np.where(price_grid == p_table[j, t-1])[0][0]
            p_table[i, t] = select_price(s_next_idx, price_grid, epsilon[t], AV_1,q1)
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
            q2[current_state_idx, p_idx] = Q_func(p_idx, current_state_idx, j, i, t, alpha, gamma, p_table, q2, price_grid, s_next, AV_2) #Agent_model_1, T
            
            s_next_idx = np.where(price_grid == p_table[i, t-1])[0][0]
            p_table[j, t] = select_price_asym(s_next_idx, price_grid, epsilon[t], AV_2, mu,q2)
            p_table[i, t] = p_table[i, t-1]
            
            profits[i, t] = profit(p_table[i, t], p_table[j, t])
            profits[j, t] = profit(p_table[j, t], p_table[i, t])
            
        if t % 1000 == 0:
            avg_profs1.append(np.sum(profits[i, (t-1000):t]) / 1000)
            avg_profs2.append(np.sum(profits[j, (t-1000):t]) / 1000)
    
    return p_table, avg_profs1, avg_profs2


def run_sim_JAL_AM_asym(n, k, mu):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
        mu: probability of observing true state
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
        avg_prof_gain: list containing average profit gains of runs
        t: nuumber of simulations
        edge: number of times simulations resulted in Edgeworth price cycle
        focal: number of times simulations resulted in focal price
    """
    
    num_calcs = int(500000 / 1000 - 1)  # size of avg. profits 
    summed_profit1=np.zeros(num_calcs)
    summed_profit2=np.zeros(num_calcs)
    summed_avg_profitabilities = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    focal = 0
    edge = 0

    # simulating n runs of JAL-AM
    for i in range(n):
        p_table, avg_profs1, avg_profs2 = JAL_AM_asym(0.3, 0.95, 500000, k, mu)
        per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
        avg_prof_gain[i] = per_firm_profit[-1] / 0.125  # Adjust index to access the last element correctly
        summed_profit1=np.sum([summed_profit1,avg_profs1],axis=0)
        summed_profit2=np.sum([summed_profit2,avg_profs2],axis=0)
        summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
        edge, focal, p_m = edge_or_focal_asym(edge, focal, p_table, mu, 50)
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal



def run_sim_JALAM_asym_single_run(alpha, gamma, T, price_grid,mu):
    p_table, avg_profs1, avg_profs2 = JAL_AM_asym(alpha, gamma, T, price_grid,mu)
    per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
    return p_table, avg_profs1, avg_profs2, per_firm_profit

def run_sim_JAL_AM_asym(n, k, mu,show_progress=False):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
        mu: probability of observing true state
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
        avg_prof_gain: list containing average profit gains of runs
        t: nuumber of simulations
        edge: number of times simulations resulted in Edgeworth price cycle
        focal: number of times simulations resulted in focal price
    """
    num_calcs = int(500000 / 1000 - 1)
    summed_avg_profitabilities = np.zeros(num_calcs)
    summed_profit1 = np.zeros(num_calcs)
    summed_profit2 = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    focal = 0
    edge = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_sim_JALAM_asym_single_run, 0.3, 0.95, 500000, k,mu) for _ in range(n)]
        if show_progress:
            iterator = tqdm(enumerate(as_completed(futures)), total=n)
        else:
            iterator = enumerate(as_completed(futures))
        
        for i, future in iterator:
            p_table, avg_profs1, avg_profs2, per_firm_profit = future.result()
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            summed_profit1 = np.sum([summed_profit1, avg_profs1], axis=0)
            summed_profit2 = np.sum([summed_profit2, avg_profs2], axis=0)
            avg_prof_gain[i] = per_firm_profit[498] / 0.125
            edge, focal, isfocal = edge_or_focal_asym(edge, focal, p_table,mu,50)
            
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal