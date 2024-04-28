import matplotlib.pyplot as plt
import time

import numpy as np

from StrategyIteration import *
def normal_vec(vectr):              # normalize a vector
    sum = np.sum(vectr)
    for i in range(len(vectr)):
        vectr[i] = vectr[i] / sum
    return np.array(vectr)
def pro(s,a,dd):                ########### Probability of going from s to dd by action a
    return kernels[s,a,dd]

def reward(s,a):                ########### reward of taking action a in state s
    return rewards[s,a]

def sigma(r,s,a,V):             ########### sigma function defined in experiments section. 
    pv=0
    for k in range(S_n):
        pv=pv+(1-r)*V[k]*pro(s,a,k)
    return pv+r*min(V)

def pi(gamma,r,s,V):            ########## applies one step of value iteration on node s
    com_pi=[]
    for j in All_actions:
        com_pi.append((1-gamma)*reward(s,j)+gamma*sigma(r,s,j,V))
    return np.argmax(com_pi)

def PI(gamma,r,V):              ######### applies one step of value iteration for all nodes
    oo=[]
    for s_PI in range(S_n):
        oo.append(pi(gamma,r,s_PI,V))
    return oo

############################################
def run_RVI(limit_value, eps):
    r = 0.4
    V = [0] * S_n
    # RR=[]
    t = 0

    print("running VI...")
    while True:
    # for counter in range(300):
        gamma = (t + 1) / (t + 2)
        t = t + 1
        pi1 = PI(gamma, r, V)
        # RR.append(V[0])
        if t % 3000 == 0:
            print(t, V[0])
        W = []
        gamma = (t + 1) / (t + 2)
        for j in range(S_n):
            W.append(V[j])

        for s_train in All_states:
            W[s_train] = (1 - gamma) * reward(s_train, 0) + gamma * sigma(r, s_train, 0, V)
            for a_train in All_actions:
                W[s_train] = max(W[s_train],
                                 (1 - gamma) * reward(s_train, a_train) + gamma * sigma(r, s_train, a_train, V))
        V = []
        for s_new in range(S_n):
            V.append(W[s_new])
        if abs(V[0]-limit_value)<eps:
            break
    return V

def span_distance(A,B):
    C = [A[i]-B[i] for i in range(len(A))]
    return max(C)-min(C)

def run_RRVI(limit_value, s_star, eps):
    r=0.4
    V_VI = [0]*S_n
    W_VI = [0]*S_n         # V[i] - V[s_start] = 0
    prev_W = None
    k = 0
    while prev_W is None or span_distance(prev_W,W_VI) >= eps:
        V_tmp = [0]*S_n
        W_tmp = [0]*S_n
        for s in range(S_n):
            V_tmp[s] = max([reward(s,a)+sigma(r,s,a,W_VI) for a in range(A_n)])
        for s in range(S_n):
            W_tmp[s] = V_tmp[s] - V_tmp[s_star]
        prev_W = W_VI.copy()
        W_VI = W_tmp
        V_VI = V_tmp
        k = k+1
    print(f"VI finished after {k} steps")
    policy = {}
    for s in range(S_n):
        policy[s] = np.argmax([reward(s,a)+sigma(r,s,a,V_VI) for a in range(A_n)])
    V_PE = [0] * S_n
    W_PE = [0] * S_n  # V[i] - V[s_start] = 0
    prev_W = None
    limit = [limit_value] * S_n
    while prev_W is None or span_distance(prev_W, W_PE) >= eps:
        V_tmp = [0] * S_n
        W_tmp = [0] * S_n
        for s in range(S_n):
            V_tmp[s] = reward(s, policy[s]) + sigma(r, s, policy[s], W_PE)
        for s in range(S_n):
            W_tmp[s] = V_tmp[s] - V_tmp[s_star]
        prev_W = W_PE.copy()
        W_PE = W_tmp
        V_PE = V_tmp
    return [V_PE[s] - W_PE[s] for s in range(S_n)]


def run_RPPI():
    R = 0.4
    game_states = {}
    for s in All_states:
        game_states[s] = State(str(s))
        for a in All_actions:
            game_states[(s, a)] = State(f"({s},{a})")

    game_actions = []

    for s in All_states:
        for a in All_actions:
            current_action = Action(reward(s, a))

            current_action.add_new_state(game_states[(s, a)], 1)

            game_states[s].add_action(current_action)

            game_actions.append(current_action)
            # print(str(current_action))
            for s_prime in All_states:  # for each (s,a,s') we have one action that is active only in (s,a)
                current_action = Action(reward(s, a))
                for s_second in All_states:
                    if s_second == s_prime:
                        current_action.add_new_state(game_states[s_second], pro(s, a, s_second) * (1 - R) + R)
                    else:
                        current_action.add_new_state(game_states[s_second], pro(s, a, s_second) * (1 - R))
                game_states[(s, a)].add_action(current_action)
                game_actions.append(current_action)
                # print(str(current_action))
        # print(f"up to now: {round(time.time() * 1000) - RPPI_time}")

    game = Game(10 ** -5)

    for s in All_states:
        game.add_max_state(game_states[s])
        for a in All_actions:
            game.add_min_state(game_states[s, a])
    game.check_game()
    print(f"making game: {round(time.time() * 1000) - RPPI_time}ms")

    print("running mean average...")
    result = game.mean_average_strategy_iteration(0.1, 10 ** -5)
    print("RPPI value: " + str(result['values'][0]))
    return result['values'][0]



def make_MDP(n):
    global  All_states,All_actions, S_n, A_n, rewards, kernels
    All_states = list(range(0, n))
    All_actions = list(range(0, n+10))
    S_n = len(All_states)
    A_n = len(All_actions)
    rewards = np.ones((S_n, A_n))
    kernels = np.ones((S_n, A_n, S_n))
    for s in All_states:
        for a in All_actions:
            kernels[s, a] = normal_vec(np.random.random(S_n))

    for s in range(S_n):
        for a in range(A_n):
            rewards[s,a]=np.random.random()
    # print(max([reward(s,a) for s in range(S_n) for a in range(A_n)]))


if __name__ == '__main__':
    RVI_time_plot = []
    RPPI_time_plot = []
    RRVI_time_plot = []
    for n in range(1,51):
        print(f"----------------------- n={n} ------------------------")
        ################# making MDP #######################
        np.random.seed(1)
        make_MDP(n)
        # for i in range(S_n):
        #     for j in range(A_n):
        #         print(f"({i},{j}) -> {reward(i,j)}")
        ################# making MDP #######################

        RPPI_time = round(time.time() * 1000)
        RPPI_value = run_RPPI()
        RPPI_time = round(time.time() * 1000) - RPPI_time
        RPPI_time_plot.append(RPPI_time)
        print(f"RPPI_time: {RPPI_time}")

        RVI_time = round(time.time()*1000)
        RVI_value=run_RVI(RPPI_value, 10 ** -3)
        print("RVI Value: " + str(RVI_value[0]))
        RVI_time = round(time.time()*1000) - RVI_time
        RVI_time_plot.append(RVI_time)
        print(f"RVI time: {RVI_time}")

        RRVI_time = round(time.time()*1000)
        RRVI_value = run_RRVI(RPPI_value,0,10**-6)
        print("RRVI Value: " + str(RRVI_value[0]))
        RRVI_time = round(time.time() * 1000) - RRVI_time
        RRVI_time_plot.append(RRVI_time)
        print(f"RRVI time: {RRVI_time}")





    plt.plot(RVI_time_plot, label = 'RVI')
    plt.plot(RRVI_time_plot, label='RRVI')
    plt.plot(RPPI_time_plot,label = 'RPPI')
    plt.xlabel('Number of states')
    plt.ylabel('Time (ms)')
    plt.yscale("log")
    plt.legend()
    plt.savefig("contamination.png")


    file = open('RVI_time_con.txt','w')
    file.write(str(RVI_time_plot))
    file.close()

    file = open('RPPI_time_con.txt','w')
    file.write(str(RPPI_time_plot))
    file.close()

    file = open('RRVI_time_con.txt', 'w')
    file.write(str(RRVI_time_plot))
    file.close()