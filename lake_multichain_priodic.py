import matplotlib.pyplot as plt
import time
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


def run_RPPI(eps):
    # pre-condition: R<1/3
    R = 0.2
    game_states = {}
    for state in All_states:
        game_states[state] = State(str(state))
        for action in All_actions:
            game_states[(state, action)] = State(f"({state},{action})")

    game_actions = []

    for state in All_states:
        for action in All_actions:
            if state in holes:
                current_action = Action(reward(state, action))
                current_action.add_new_state(game_states[(state, action)], 1)
                game_states[state].add_action(current_action)
                game_actions.append(current_action)

                current_action = Action(reward(state, action))
                current_action.add_new_state(game_states[state],1)
                game_states[(state,action)].add_action(current_action)
                game_actions.append(current_action)
            else:
                current_action = Action(reward(state, action))
                current_action.add_new_state(game_states[(state, action)], 1)
                game_states[state].add_action(current_action)
                game_actions.append(current_action)

                p = [pro(state, action, k) for k in range(S_n)]
                succ = [get_left(state), get_up(state), get_right(state), get_down(state)]

                corners = []
                for tilt in range(4):
                    # print("state:",state,"action:",action,"tilt:",tilt)
                    p0 = p.copy()
                    if succ[tilt]!=state:   #succ[tilt] is not blocked
                        if p0[succ[tilt]]!=1:
                            t = R/(1-p0[succ[tilt]]-R)
                            p0[succ[tilt]] = p0[succ[tilt]]+t       # increase prob(succ[tilt]) by t
                                                                # after normalization, prob(succ[tilt]) is increased by R and other successors have decreased uniformly
                    elif succ[(tilt+2)%4]!=state and p0[succ[(tilt+2)%4]]!=0:       #succ[tilt+2] is not blocked and has non-zero prob
                        if succ[(tilt+2)%4]!=1:
                            t = -R/(1-succ[(tilt+2)%4]+R)
                            p0[succ[(tilt+2)%4]]=p0[succ[(tilt+2)%4]]+t     # increase prob(succ[tilt+2]) by t
                                                                        # after normalization, prob(succ[tilt+2]) is decreased by R and other successors have increased uniformly

                    p0 = normal_vec(p0)
                    corners.append(p0)


                for prob in corners:
                    current_action = Action(reward(state,action))
                    for s_prime in set(succ):
                        current_action.add_new_state(game_states[s_prime], prob[s_prime])
                    game_states[(state,action)].add_action(current_action)
                    game_actions.append(current_action)

    game = Game(10**-5)

    for state in All_states:
        game.add_max_state(game_states[state])
        for action in All_actions:
            game.add_min_state(game_states[state, action])
    game.check_game()
    print(f"making game: {round(time.time() * 1000) - RPPI_time}ms")

    print("running mean average...")
    result = game.mean_average_strategy_iteration(0.1, eps)
    print("RPPI value: " + str(result['values'][0]))
    return result['values'][0]

def get_right(state):
    i=int(state/grid_size)
    j=int(state%grid_size)
    state_right = state if j == grid_size - 1 else state + 1
    return state_right
def get_left(state):
    i = int(state / grid_size)
    j = int(state % grid_size)
    state_left = state if j == 0 else state - 1
    return state_left
def get_up(state):
    i = int(state / grid_size)
    j = int(state % grid_size)
    state_up = state if i == 0 else state - grid_size
    return state_up
def get_down(state):
    i = int(state / grid_size)
    j = int(state % grid_size)
    state_down = state if i == grid_size - 1 else state + grid_size
    return state_down
def make_MDP(n):
    global  All_states,All_actions, S_n, A_n, rewards, kernels, holes, grid_size
    grid_size = n
    All_states = list(range(0, n*n))
    holes = np.random.randint(low=0,high=n*n,size=n)
    if 0 in holes:
        holes=np.delete(holes,np.where(holes==0))
    for i in range(n):
        for j in range(n):
            if i*n+j in holes:
                print("#",end="")
            else:
                print(".",end="")
        print()
    All_actions = [0,1,2,3]     # left, up, right, down
    S_n = n*n
    A_n = 4
    kernels = np.zeros((S_n, A_n, S_n))
    for i in range(n):
        for j in range(n):
            state = i*n+j
            if state in holes:
                for action in range(4):
                    p= np.zeros(S_n)
                    p[state]=1
                    kernels[state,action]=normal_vec(p)
            else:
                succ = [get_left(state),get_up(state),get_right(state),get_down(state)]

                for action in range(4):
                    p = np.zeros(S_n)
                    p[succ[action]] = 1
                    p[succ[(action+1)%4]] = 1
                    p[succ[(action+3)%4]] = 1
                    p[state] = 0
                    if np.all(p==0):
                        p[succ[(action+2)%4]]=1
                        p[state] = 0
                    kernels[state, action] = normal_vec(p)

    S_n = len(All_states)
    A_n = len(All_actions)
    rewards = np.zeros((S_n, A_n))
    for i in range(n):
        for j in range(n):
            s = i*n+j
            for a in range(A_n):
                rewards[s,a]=i+j


if __name__ == '__main__':
    RPPI_time_plot = []
    for n in range(2,11):
        print(f"------------------------ n={n} ------------------------")
        ################# making MDP #######################
        np.random.seed(1)
        make_MDP(n)
        for s in All_states:
            for a in All_actions:
                ppp = []
                for s_prime in All_states:
                    ppp.append(pro(s, a, s_prime))
                # print("state:",s,"action:",a,"prob:",ppp)
        ################# making MDP #######################

        RPPI_time = round(time.time() * 1000)
        RPPI_value = run_RPPI(10**-3)
        RPPI_time = round(time.time() * 1000) - RPPI_time
        RPPI_time_plot.append(RPPI_time)
        print(f"RPPI_time: {RPPI_time}")


    # plt.plot(RPPI_time_plot, label='Reduction to Stochastic Games')
    # plt.xlabel('Number of states')
    # plt.ylabel('Time (ms)')
    # plt.legend()
    # plt.show()


    file = open('RPPI_time_lake_multichain.txt','w')
    file.write(str(RPPI_time_plot))
    file.close()