import numpy as np
from model_lib import *


# Update model based on actions
def update_predictive_model(pi_hat, current_state, current_price, counter, price_grid):
    current_state_idx = np.where(price_grid == current_state)[0][0]
    current_price_idx = np.where(price_grid == current_price)[0][0]
    pi_hat[current_state_idx, current_price_idx] = counter[current_state_idx, current_price_idx] / np.sum(counter[current_state_idx])
    counter[current_state_idx, current_price_idx] += 1
    return pi_hat

# Predict next action based on current state
def predict_action(model, current_state_index, price_grid):
    
    return 



def Q_func_JAL_AM(p_curr_idx, s_curr_idx, predicted_action_index, alpha, gamma, Q_table, price_grid, s_next):
    
    return 

"""
def AV(Q_table, current_state, pi_hat, price_grid):
    current_state_idx = np.where(price_grid == current_state)[0][0]
    q_sum = 0
    model_sum = 0
    for i in range(len(price_grid)):
        q_sum += Q_table[current_state_idx, i]
        model_sum *= pi_hat[]


def select_price_JAL_AM(model, Q_table, price_grid, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(price_grid)
    else:
        
        np.argmax((:,state))
        return 


def JAL_AM( T, price_grid, alpha, gamma, epsilon):
    
    epsilon = calculate_epsilon(T)
    i = 0
    j = 1
    t = 0
    # Initializing Q-functions
    k = len(price_grid)
    q1 = np.zeros((k, k)) 
    q2 = np.zeros((k, k))
    
    predictive_model1 = np.zeros((len(k), len(k)))/len(k)
    predictive_model2 = np.zeros((len(k), len(k)))/len(k)
    # Initializing N, a counter
    C1 = np.zeros(k,k)
    C2 = np.zeros(k,k)
    # Initializing profits
    p_table = np.zeros((2,T))
    profits = np.zeros((2,T))
    avg_profs1 = []
    avg_profs2 = []

    p_table[i, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[j, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[i, t] = np.random.choice(price_grid) #p_table[i, t-1]
    t += 1 # now t = 2

    # Loop through the simulation time
    for t in range(t, T):
        # observe state
        s_curr = p_table[j,t-2]


        p_table[i, t] = select_price_JAL_AM(predictive_model1, q1, price_grid, epsilon[t])
        
        (j, t, p_table, q1, price_grid, epsilon[t])


        p_table[i,t] = p_table[i,t-1]

        p_idx = np.where(price_grid == p_table[i,t])[0][0]
        s_next = p_table[j,t-1]
        #s_next_idx = np.where(price_grid == s_next)[0][0]
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next)

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
        # Update predictive model based on actions
        # ...
        # Use the updated predictive model in Q-value updates and decision-making
        # ...
"""