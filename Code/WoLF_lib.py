from model_lib import *
from tqdm.notebook import tqdm
from tqdm import tqdm
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
        price_grid: grid of prices
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
    max_Q = max(q[next_state_idx, :])
    p_idx=np.where(price_grid == p_table[i,t-1])[0][0]
    q[current_state_idx, p_idx] = q[current_state_idx, p_idx] + alpha * (reward + gamma**2 * max_Q - q[current_state_idx, p_idx])
       
    return q

@njit
def update_policy_WoLF(policy, price_grid, delta_l, delta_w, p_table, q, t, N, j, avg_policy, k):
    """
    args:
        policy: policy of player i to be updated
        price_grid: grid of prices
        delta_l: learning rate when losing
        delta_w: learning ratwe when winning
        p_table: 2x500.000 array containing all prices
        q: Q-talbe of player
        t: current period
        N: counter matrix 
        j: player j
        avg_policy: average policy of player i
        k: length of price g
    returns:
        policy: the updated policy for player i 
        N: counter matrix
        avg_policy: average policy
        q: Q-table for player i
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
    
    for t in range(t, T):
        
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
         
    return avg_profs1, avg_profs2, p_table


def run_sim_wolf_single_run(alpha, delta_win, delta_loss, gamma, price_grid, T):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        delta_win: learning rate when winning
        delta_loss: learning rate when losing
        gamma: discount factor
        price_grid: grid of prices
        T: learning duration
    returns:
        avg_profs1: average profits for firm 1
        avg_profs2: average profitst for firm 2
        p_table: array containing all prices set by both firms
        per_firm_proft: per firm profit
    """
    avg_profs1, avg_profs2, p_table = WoLF_PHC(alpha, delta_win, delta_loss, gamma, price_grid, T)
    per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
    return avg_profs1, avg_profs2, p_table, per_firm_profit

def run_sim_wolf(n, k, show_progress=False):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
        show_progress: Progress bar or not
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
        avg_prof_gain: list containing average profit gains of runs
        edge: number of times simulations resulted in Edgeworth price cycle
        focal: number of times simulations resulted in focal price
    """
    num_calcs = int(500000 / 1000 - 1)  # size of avg. profits 
    summed_avg_profitabilities = np.zeros(num_calcs)
    avg_prof_gain = np.zeros(n)
    focal = 0
    edge = 0
    p_mc = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_sim_wolf_single_run, 0.3, 0.6, 0.1, 0.95,k, 500000) for _ in range(n)]
        if show_progress:
            iterator = tqdm(enumerate(as_completed(futures)), total=n, desc='WoLF-PHC')
        else:
            iterator = enumerate(as_completed(futures))
        
        for i, future in iterator:
            avg_profs1, avg_profs2, p_table, per_firm_profit = future.result()
            avg_prof_gain[i] = per_firm_profit[498] / 0.125
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            edge, focal, p_mc = edge_or_focal(edge, focal, p_table)
            
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal


# ASYMMETRIC INFORMATION

@njit
def edge_or_focal_asym(edge, focal, p_table, mu, periods):
    """
    args
        edge: counter for edgeworth cycles
        focal: counter for focal pricing
        p_table: price table from a simulation
        mu: probablity of observing wrong price
        periods: periods to check for price cycles
    returns
        edge:counter for edgeworth cycles
        focal: counter for focal pricing
        is_focal: boolean, focal pricing or not
    """
    tolerance = mu * periods
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
def select_price_WoLF_asym(epsilon, price_grid, current_state, policy, mu):
    """
    args:
        epsilon: epsilon value to period t
        price_grid: grid of prices
        current_state: current state of player
        policy: policy of player
        mu: probablity of observing wrong price
    returns:
        either a random price or price determined by policy
    """
    
    true_state=np.where(price_grid == current_state)[0][0] # current state (opponent's price)
    if mu<=np.random.uniform(0,1):
        s_t_idx=true_state 
    else:
        s_t_idx=np.where(price_grid==np.random.choice(price_grid))[0][0]
    u = np.random.uniform(0,1)
    if epsilon > u:
        return np.random.choice(price_grid)
    else:
        cumsum = np.cumsum(policy[s_t_idx, :])
        idx = np.searchsorted(cumsum, np.array([u]))[0]
        return price_grid[idx]
    

@njit
def WoLF_PHC_asym(alpha, delta_l, delta_w, gamma, price_grid, T, mu):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        delta_l: learning rate when losing
        delta_w: learning rate when winning
        gamma: discount factor
        price_grid: grid of prices
        T: learning duration
        mu: probability of observing wrong price
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
        if t%2!=0:
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
        else:
            #update Q
            q2=q_func_wolf(q2, alpha, gamma, p_table, price_grid, j, i, t)

            #update N
            current_state_idx = np.where(price_grid == p_table[i,t-2])[0][0]
            N2[current_state_idx] += 1
            
            #update policy 
            policy_2, N2, avg_policy2, q2 = update_policy_WoLF(policy_2, price_grid, delta_l, delta_w, p_table, q2, t, N2, i, avg_policy2, k)
            
            #update prices
            p_table[j,t] = select_price_WoLF_asym(epsilon[t], price_grid, p_table[i,t-1], policy_2,mu)
            p_table[i,t]= p_table[i,t-1]
            
            #update profits
            profits[j, t] = profit(p_table[j,t], p_table[i,t])
            profits[i, t] = profit(p_table[i,t], p_table[j,t])
            
        # Compute profits
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)

        #switching players and their variables
    return avg_profs1, avg_profs2, p_table


def run_sim_wolf_asym_single_run(alpha, delta_win, delta_loss, gamma, price_grid, T, mu):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        delta_win: learning rate when winning
        delta_loss: learning rate when losing
        gamma: discount factor
        price_grid: grid of prices
        T: learning duration
    returns:
        avg_profs1: average profits for firm 1
        avg_profs2: average profitst for firm 2
        p_table: array containing all prices set by both firms
        per_firm_proft: per firm profit
    """
    avg_profs1, avg_profs2, p_table = WoLF_PHC_asym(alpha, delta_win, delta_loss, gamma, price_grid, T, mu)
    per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
    return avg_profs1, avg_profs2, p_table, per_firm_profit

def run_sim_wolf_asym(n, k,mu, show_progress=False):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
        show_progress: progress bar or not
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
        avg_prof_gain: list containing average profit gains of runs
        edge: number of times simulations resulted in Edgeworth price cycle
        focal: number of times simulations resulted in focal price
    """
    num_calcs = int(500000 / 1000 - 1)  # size of avg. profits 
    summed_avg_profitabilities = np.zeros(num_calcs)
    summed_profit1 = np.zeros(num_calcs)
    summed_profit2 = np.zeros(num_calcs)
    avg_prof_gain = np.zeros(n)
    focal = 0
    edge = 0
    p_mc = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_sim_wolf_asym_single_run, 0.3, 0.6, 0.2, 0.95,k, 500000,mu) for _ in range(n)]
        if show_progress:
            iterator = tqdm(enumerate(as_completed(futures)), total=n, desc='WoLF-PHC asym')
        else:
            iterator = enumerate(as_completed(futures))
        
        for i, future in iterator:
            avg_profs1, avg_profs2, p_table, per_firm_profit = future.result()
            avg_prof_gain[i] = per_firm_profit[498] / 0.125
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            summed_profit1 = np.sum([summed_profit1, avg_profs1], axis=0)
            summed_profit2 = np.sum([summed_profit2, avg_profs2], axis=0)
            edge, focal, p_mc = edge_or_focal_asym(edge, focal, p_table,mu,50)
    avg_summed_profit1 = np.divide(summed_profit1, n)
    avg_summed_profit2 = np.divide(summed_profit2, n)       
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal, avg_summed_profit1, avg_summed_profit2

###################CONVERGENCE##############################

@njit
def WoLF_PHC_convergence(alpha, delta_l, delta_w, gamma, price_grid, T):
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

    q1list=[]
    q2list=[]
    
    for t in range(t, T):
        
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
        if t>=499000:  
            if t%2!=0:
                q1list.append(q1)
                q2list.append(q2)
            else:
                q1list.append(q2)
                q2list.append(q1)
         
    return  p_table, q1list,q2list


@njit
def WoLF_PHC_asym_convergence(alpha, delta_l, delta_w, gamma, price_grid, T, mu):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        delta_l: learning rate when losing
        delta_w: learning rate when winning
        gamma: discount factor
        price_grid: grid of prices
        T: learning duration
        mu: probability of observing wrong price
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
    q1list=[]
    q2list=[]
    
    for t in range(t, T-1):
        if t%2!=0:
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
        else:
            #update Q
            q2=q_func_wolf(q2, alpha, gamma, p_table, price_grid, j, i, t)

            #update N
            current_state_idx = np.where(price_grid == p_table[i,t-2])[0][0]
            N2[current_state_idx] += 1
            
            #update policy 
            policy_2, N2, avg_policy2, q2 = update_policy_WoLF(policy_2, price_grid, delta_l, delta_w, p_table, q2, t, N2, i, avg_policy2, k)
            
            #update prices
            p_table[j,t] = select_price_WoLF_asym(epsilon[t], price_grid, p_table[i,t-1], policy_2,mu)
            p_table[i,t]= p_table[i,t-1]
            
            #update profits
            profits[j, t] = profit(p_table[j,t], p_table[i,t])
            profits[i, t] = profit(p_table[i,t], p_table[j,t])
            
        # Compute profits
        if t % 1000 == 0:
            profitability = np.sum(profits[i, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[j, (t-1000):t])/1000
            avg_profs2.append(profitability)
        if t>=499000:  
            q1list.append(q1)
            q2list.append(q2)

        #switching players and their variables
    return p_table, q1list,q2list
