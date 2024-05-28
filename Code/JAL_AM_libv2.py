import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm
from model_lib import *

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
def calculate_AV(s, a, Q, agent_model_j):
    """
    Args:
        s: The current state
        a: current action
        Q: The Q-value table
        agent_model_j: The estimated policy of the other agent
    returns
        AV: action value 
    """
    AV = Q[s, a, s] * agent_model_j[s]
    return AV

@njit
def max_AV(s, Q, agent_model_j, price_grid):
    AVs = np.empty(len(price_grid))
    for i in range(len(price_grid)):
        AVs[i] = calculate_AV(s, i, Q, agent_model_j)
    return np.argmax(AVs)


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
    total_actions = np.sum(counter[state,:])

    for i in range(k):
        agent_model[i] = counter[state, i] / total_actions # updates column values
    return counter, agent_model

@njit
def select_price(s_idx, price_grid, epsilon, Q, agent_model_j):
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
        best_a_idx = max_AV(s_idx, Q, agent_model_j, price_grid)
        return price_grid[best_a_idx]
    
        #maxedAV_idx = max(calculate_AV(s_idx, , Q, agent_model_j))
        #return price_grid[maxedAV_idx]

@njit
def Q_func(p_idx, s_idx, i, j, t, alpha, gamma, p_table, Q, price_grid, s_next, agent_model_j):
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
    prev_est = Q[s_idx, p_idx, s_idx]
    max_AV_idx = max_AV(s_idx, Q, agent_model_j, price_grid)
    maxed_AV = price_grid[max_AV_idx]
    reward = profit(p_table[i, t], p_table[j, t-2]) + gamma * profit(p_table[i, t], s_next) + gamma * maxed_AV
    return (1 - alpha) * prev_est + alpha * reward

@njit
def JAL_AM2(alpha, gamma, T, price_grid):
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
    Agent_model_1, Agent_model_2 = np.ones(k) / k, np.ones(k) / k
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
        q1[current_state_idx, p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i, j, t, alpha, gamma, p_table, q1, price_grid, s_next, Agent_model_1)
        # setting price
        s_next_idx = np.where(price_grid == p_table[j, t-1])[0][0]
        p_table[i, t] = select_price(s_next_idx, price_grid, epsilon[t], q1, Agent_model_1)
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

def run_sim(n, k):
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
    # initalizing values
    num_calcs=int(500000/1000-1) # size of avg. profits 
    summed_avg_profitabilities = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    focal = 0
    edge = 0
    # simulating n runs of JAL-AM
    for i in tqdm(range(n), desc='JAL-AM', leave=True):
        p_table, avg_profs1, avg_profs2 = JAL_AM2(0.3, 0.95, 500000, k)
        per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0)/2
        avg_prof_gain[i] = per_firm_profit[498]/0.125
        summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
        edge, focal, p_m = edge_or_focal(edge, focal, p_table)
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal