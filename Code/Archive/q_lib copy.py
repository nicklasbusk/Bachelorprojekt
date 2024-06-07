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
def Q_func(p_curr_idx, s_curr_idx, i, j, t, alpha, gamma, p_table, Q_table, price_grid, s_next) -> float: # p_table contains p and s (opponent price)
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
    prev_est = Q_table[p_curr_idx, s_curr_idx]
    s_next_index=np.where(price_grid == s_next)[0][0]
    maxed_Q = max(Q_table[:, s_next_index])
    reward = profit(p_table[i, t], p_table[j, t-2]) + gamma * profit(p_table[i, t], s_next) + gamma**2 * maxed_Q
    return (1 - alpha) * prev_est + alpha * reward




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
def Q_learner(alpha, gamma, T, price_grid):
    """
    args:
        alpha: step-size parameter
        gamma: discount factor
        T: learning duration
        price_grid: price_grid
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
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
    # setting prices for players in first 2 periods 
    p_table[i, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[j, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[i, t] = np.random.choice(price_grid) 
    t += 1 # now t = 2

    for t in range(t, T):
        # updating q-tables 
        p_table[i,t] = p_table[i,t-1]
        p_idx = np.where(price_grid == p_table[i,t])[0][0]
        s_next = p_table[j,t-1]
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next)
        # setting price
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


def run_sim_Q_single_run(alpha, gamma, T, price_grid):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        gamma: discount factor
        T: learning duration
        price_grid: grid of prices
    returns:
        p_table: array containing all prices set by both firms
        avg_profs1: average profits for firm 1
        avg_profs2: average profitst for firm 2
        per_firm_proft: per firm profit
    """
    p_table, avg_profs1, avg_profs2 = Q_learner(alpha, gamma, T, price_grid)
    per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
    return p_table, avg_profs1, avg_profs2, per_firm_profit

def run_sim_Q(n, k, show_progress=False):
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
    num_calcs = int(500000 / 1000 - 1)
    summed_avg_profitabilities = np.zeros(num_calcs)
    summed_profit1 = np.zeros(num_calcs)
    summed_profit2 = np.zeros(num_calcs)
    avg_prof_gain = np.zeros((n))
    focal = 0
    edge = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_sim_Q_single_run, 0.3, 0.95, 500000, k) for _ in range(n)]
        
        # Select the appropriate iterator based on the show_progress flag
        if show_progress:
            iterator = tqdm(enumerate(as_completed(futures)), total=n, desc='Q-learning')
        else:
            iterator = enumerate(as_completed(futures))
        
        for i, future in iterator:
            p_table, avg_profs1, avg_profs2, per_firm_profit = future.result()
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            summed_profit1 = np.sum([summed_profit1, avg_profs1], axis=0)
            summed_profit2 = np.sum([summed_profit2, avg_profs2], axis=0)
            avg_prof_gain[i] = per_firm_profit[498] / 0.125
            edge, focal, p_m = edge_or_focal(edge, focal, p_table)
            
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal

# ASYMETRIC INFORMATION
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
def select_price_asym(j, t, p_table, Q_table, price_grid, epsilon, mu):
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
    
    true_state=np.where(price_grid == p_table[j, t-1])[0][0] # current state (opponent's price)
    if mu<=np.random.uniform(0,1):
        s_t_idx=true_state 
    else:
        s_t_idx=np.where(price_grid==np.random.choice(price_grid))[0][0]
    
    # Exploration
    if epsilon >= np.random.uniform(0,1):
        return np.random.choice(price_grid)
    else:
    # Exploitation
        maxedQ_idx = np.argmax(Q_table[:, s_t_idx])
        return price_grid[maxedQ_idx]

@njit
def Q_asym(alpha, gamma, T, price_grid, mu):
    """
    args:
        alpha: step-size parameter
        gamma: discount factor
        T: learning duration
        price_grid: price_grid
        mu: probability of observing wrong price
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
        avg_profs0: player 0 list of average profit for each 1000 period
        avg_profs1: player 1 list of average profit for each 1000 period
    """
    # Initializing values
    epsilon = calculate_epsilon(T)
    #i = 0
    #j = 1
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
    p_table[0, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[1, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[0, t] = np.random.choice(price_grid) #p_table[i, t-1]
    t += 1 # now t = 2
    liste=[]
    y=10
    for t in range(t, T):
        if t%2!=0:
            p_table[0,t] = p_table[0,t-1]# Det er ligemeget om det er -1 eller -2 da den sætter prisen 2 gange i træk
            p_idx = np.where(price_grid == p_table[0,t])[0][0]
            s_next = p_table[1,t-1]
            #s_next_idx = np.where(price_grid == s_next)[0][0]
            current_state_idx = np.where(price_grid == p_table[1,t-2])[0][0]
            q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, 0,1, t, alpha, gamma, p_table, q1, price_grid, s_next)

            p_table[0, t] = select_price(1, t, p_table, q1, price_grid, epsilon[t])
            p_table[1, t] = p_table[1, t-1]

            # Store profits for both firms
            profits[0, t] = profit(p_table[0,t], p_table[1,t])
            profits[1, t] = profit(p_table[1,t], p_table[0,t])
        else:
            p_table[1,t] = p_table[1,t-1]# Det er ligemeget om det er -1 eller -2 da den sætter prisen 2 gange i træk
            p_idx = np.where(price_grid == p_table[1,t])[0][0]
            s_next = p_table[0,t-1]
            #s_next_idx = np.where(price_grid == s_next)[0][0]
            current_state_idx = np.where(price_grid == p_table[0,t-2])[0][0]
            q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, 1,0, t, alpha, gamma, p_table, q1, price_grid, s_next)
            
            #questionable state for select price asym
            p_table[1, t] = select_price_asym(0, t, p_table, q1, price_grid, epsilon[t], mu)
            p_table[0, t] = p_table[0, t-1]

            # Store profits for both firms
            profits[1, t] = profit(p_table[1,t], p_table[0,t])
            profits[0, t] = profit(p_table[0,t], p_table[1,t])
            if t>=240000:
                liste.append(q1)


        # compute avg profitability of last 1000 runs for both firms
        if t % 1000 == 0:
            profitability = np.sum(profits[0, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[1, (t-1000):t])/1000
            avg_profs2.append(profitability)
            
        # changing agents
        #tmp = i
        #i = j
        #j = tmp
        tmp=q1
        q1=q2
        q2=tmp        
    return p_table, avg_profs1, avg_profs2

def run_sim_Q_asym_single_run(alpha, gamma, T, price_grid, mu):
    """
    args:
        alpha: step-size parameter that regulates how quickly new information replaces old information
        gamma: discount factor
        T: learning duration
        price_grid: grid of prices
        mu: probability of observing wrong price
    returns:
        avg_profs1: average profits for firm 1
        avg_profs2: average profitst for firm 2
        p_table: array containing all prices set by both firms
        per_firm_proft: per firm profit
    """
    p_table, avg_profs1, avg_profs2 = Q_asym(alpha, gamma, T, price_grid, mu)
    per_firm_profit = np.sum([avg_profs1, avg_profs2], axis=0) / 2
    return p_table, avg_profs1, avg_profs2, per_firm_profit

def run_sim_Q_Asym(n, k,mu, show_progress=False):
    """
    args:
        n: number of runs simulated
        k: length of price action vector
        mu: probability of observing wrong price
    returns:
        avg_avg_profitabilities: average of average profits over n simulations
        res1: summed average profits of firm 1
        res2: summed average profits of firm 2
        edge: number of times simulations resulted in Edgeworth price cycles
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
        
        futures = [executor.submit(run_sim_Q_asym_single_run, 0.3, 0.95, 500000, k,mu) for _ in range(n)]
        if show_progress:
            iterator = tqdm(enumerate(as_completed(futures)), total=n, desc='Q-learning asym')
        else:
            iterator = enumerate(as_completed(futures))
        
        for i, future in iterator:
            p_table, avg_profs1, avg_profs2, per_firm_profit = future.result()
            summed_avg_profitabilities = np.sum([summed_avg_profitabilities, per_firm_profit], axis=0)
            summed_profit1 = np.sum([summed_profit1, avg_profs1], axis=0)
            summed_profit2 = np.sum([summed_profit2, avg_profs2], axis=0)
            avg_prof_gain[i] = per_firm_profit[498] / 0.125
            edge, focal, p_m = edge_or_focal_asym(edge, focal, p_table,mu,50)
            
    avg_avg_profitabilities = np.divide(summed_avg_profitabilities, n)
    return avg_avg_profitabilities, avg_prof_gain, edge, focal


# fra v2
@njit
def Klein_simulation_FD(alpha, gamma, T, price_grid):
    """
    args:
        alpha: step-size parameter
        gamma: discount factor
        T: learning duration
        price_grid: price_grid
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
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
    # Setting prices for players in first 2 periods 
    p_table[i, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[j, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[i, t] = np.random.choice(price_grid) #p_table[i, t-1]
    t += 1 # now t = 2

    for t in range(t, T):
        p_table[i,t] = p_table[i,t-1]
        p_idx = np.where(price_grid == p_table[i,t])[0][0]
        s_next = p_table[j,t-1]
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next)

        # player i forces deviation. Selects a lower price than collusion price
        if t==499950:
            p_table[i,t] = price_grid[current_state_idx - 1]
        else:
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
        
    return p_table, profits, avg_profs1, avg_profs2

def run_simFD(n, k):
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
        p_table, profits, avg_profs1, avg_profs2 = Klein_simulation_FD(0.3, 0.95, 500000, k)
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














@njit
def Q_learner_convergence(alpha, gamma, T, price_grid):
    """
    args:
        alpha: step-size parameter
        gamma: discount factor
        T: learning duration
        price_grid: price_grid
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
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
    # setting prices for players in first 2 periods 
    p_table[i, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[j, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[i, t] = np.random.choice(price_grid) 
    t += 1 # now t = 2


    q1list=[]
    q2list=[]
    for t in range(t, T):
        # updating q-tables 
        p_table[i,t] = p_table[i,t-1]
        p_idx = np.where(price_grid == p_table[i,t])[0][0]
        s_next = p_table[j,t-1]
        current_state_idx = np.where(price_grid == p_table[j,t-2])[0][0]
        q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, i,j, t, alpha, gamma, p_table, q1, price_grid, s_next)
        # setting price
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
        if t>=499000:
            if t%2!=0:
                q1list.append(q1)
                q2list.append(q2)
            else:
                q1list.append(q2)
                q2list.append(q1)  
        
    return p_table, q1list, q2list





@njit
def Q_asym_convergence(alpha, gamma, T, price_grid, mu):
    """
    args:
        alpha: step-size parameter
        gamma: discount factor
        T: learning duration
        price_grid: price_grid
    returns:
        p_table: 2x500.000 array, with all prices set by player 0 and 1
        avg_profs0: player 0 list of average profit for each 1000 period
        avg_profs1: player 1 list of average profit for each 1000 period
    """
    # Initializing values
    epsilon = calculate_epsilon(T)
    #i = 0
    #j = 1
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
    p_table[0, t] = np.random.choice(price_grid) # firm 1 sets price
    t += 1
    p_table[1, t] = np.random.choice(price_grid) # firm 2 sets price
    p_table[0, t] = np.random.choice(price_grid) #p_table[i, t-1]
    t += 1 # now t = 2
    q1list=[]
    q2list=[]
    for t in range(t, T):
        if t%2!=0:
            p_table[0,t] = p_table[0,t-1]# Det er ligemeget om det er -1 eller -2 da den sætter prisen 2 gange i træk
            p_idx = np.where(price_grid == p_table[0,t])[0][0]
            s_next = p_table[1,t-1]
            #s_next_idx = np.where(price_grid == s_next)[0][0]
            current_state_idx = np.where(price_grid == p_table[1,t-2])[0][0]
            q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, 0,1, t, alpha, gamma, p_table, q1, price_grid, s_next)

            p_table[0, t] = select_price(1, t, p_table, q1, price_grid, epsilon[t])
            p_table[1, t] = p_table[1, t-1]

            # Store profits for both firms
            profits[0, t] = profit(p_table[0,t], p_table[1,t])
            profits[1, t] = profit(p_table[1,t], p_table[0,t])
        else:
            p_table[1,t] = p_table[1,t-1]# Det er ligemeget om det er -1 eller -2 da den sætter prisen 2 gange i træk
            p_idx = np.where(price_grid == p_table[1,t])[0][0]
            s_next = p_table[0,t-1]
            #s_next_idx = np.where(price_grid == s_next)[0][0]
            current_state_idx = np.where(price_grid == p_table[0,t-2])[0][0]
            q1[p_idx, current_state_idx] = Q_func(p_idx, current_state_idx, 1,0, t, alpha, gamma, p_table, q1, price_grid, s_next)
            
            #questionable state for select price asym
            p_table[1, t] = select_price_asym(0, t, p_table, q1, price_grid, epsilon[t], mu)
            p_table[0, t] = p_table[0, t-1]

            # Store profits for both firms
            profits[1, t] = profit(p_table[1,t], p_table[0,t])
            profits[0, t] = profit(p_table[0,t], p_table[1,t])
        


        # compute avg profitability of last 1000 runs for both firms
        if t % 1000 == 0:
            profitability = np.sum(profits[0, (t-1000):t])/1000
            avg_profs1.append(profitability)
            profitability = np.sum(profits[1, (t-1000):t])/1000
            avg_profs2.append(profitability)
            
        tmp=q1
        q1=q2
        q2=tmp 
        if t>=499000:  
            if t%2!=0:
                q1list.append(q1)
                q2list.append(q2)
            else:
                q1list.append(q2)
                q2list.append(q1)  
    return p_table, q1list,q2list