from model_lib import *
from WoLF_lib import *
from q_lib import *

@njit
def WoLF_PHC_vs_q_learning(alpha, delta_l, delta_w, gamma, price_grid, T):
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
    player="WOLF"
    for t in range(t, T-1):
        if player =="WOLF":
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
            #let Q-learning player play
            player="Q"
        else:
            p_table[i,t] = p_table[i,t-1]
            p_idx = np.where(price_grid == p_table[i,t])[0][0]
            s_next = p_table[j,t-1]
            #s_next_idx = np.where(price_grid == s_next)[0][0]
            current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
            q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next)

            p_table[i, t] = select_price(j, t, p_table, q1, price_grid, epsilon[t])
            p_table[j, t] = p_table[j, t-1]
            #let WOLF player play
            player="WOLF"

        # Store profits for both firms
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

