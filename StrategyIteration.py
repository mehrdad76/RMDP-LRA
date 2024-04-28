import math
import stormpy
import numpy as np
from enum import Enum


class Player(Enum):
    MIN = 1
    MAX = 2


class Action:
    def __init__(self, reward):
        self.reward = reward
        self.next_states = {}

    def add_new_state(self, state, prob):
        self.next_states[state] = prob

    def __str__(self):
        ret = f"reward: {self.reward}\n"
        for state in self.next_states.keys():
            ret = ret + str(state)+": "+str(self.next_states[state])+"\n"
        return ret


class State:
    def __init__(self, id=None):
        self.actions = []
        self.id = id

    def add_action(self, action):
        self.actions.append(action)

    def __str__(self):
        return "("+str(self.id)+")"
class Strategy:
    def __init__(self):
        self.state_to_action_index = {}

    def change(self, state, action_index):
        self.state_to_action_index[state] = action_index

    def __eq__(self, other):
        if other is None:
            return False
        if len(self.state_to_action_index) != len(other.state_to_action_index):
            return False
        for state in self.state_to_action_index:
            if state not in other.state_to_action_index:
                return False
            if self.state_to_action_index[state] != other.state_to_action_index[state]:
                return False
        return True


class StrategyProfile:
    def __init__(self, max_strategy, min_strategy):
        self.min_strategy = min_strategy
        self.max_strategy = max_strategy


class Game:
    def __init__(self, equality_error=0):
        self.max_states = set()
        self.min_states = set()
        self.equality_error = equality_error
        self.saved_max_strategy = None
        self.next_state_id = 0
        self.states_in_order = []

    def add_max_state(self, state):
        state.id = self.next_state_id
        self.next_state_id += 1
        self.max_states.add(state)
        self.states_in_order.append((state, Player.MAX))

    def add_min_state(self, state):
        state.id = self.next_state_id
        self.next_state_id += 1
        self.min_states.add(state)
        self.states_in_order.append((state, Player.MIN))

    def get_number_of_states(self):
        return len(self.max_states) + len(self.min_states)

    def check_game(self):
        for state in self.max_states:
            if not state.actions:
                raise Exception('each state should have action!')
            for action in state.actions:
                prob_sum = 0
                for nex_state in action.next_states:
                    prob_sum += action.next_states[nex_state]
                if prob_sum < 1-self.equality_error or prob_sum>1+self.equality_error:
                    raise Exception('sum of next states should be 1!')

    def get_random_max_strategy(self):
        max_strategy = Strategy()
        for state in self.max_states:
            max_strategy.change(state, 0)
        return max_strategy

    def get_random_min_strategy(self):
        min_strategy = Strategy()
        for state in self.min_states:
            min_strategy.change(state, 0)
        return min_strategy

    def get_optimal_min_strategy(self, max_strategy, gamma, epsilon, initial_values):
        self.saved_max_strategy, values = self.discounted_strategy_iteration_with_fixed_max_strategy(max_strategy, gamma
                                                                                                     , epsilon
                                                                                                     , initial_values
                                                                                                     , self.saved_max_strategy)
        return self.saved_max_strategy, values

    def extract_values(self, max_strategy, min_strategy, gamma, epsilon, initial_values):
        values = initial_values
        value_tmp = dict()
        for state in self.max_states:
            value_tmp[state] = values[state]
        for state in self.min_states:
            value_tmp[state] = values[state]

        max_dif = math.inf
        while max_dif > epsilon:
            for state in self.max_states:
                action_index = max_strategy.state_to_action_index[state]
                action = state.actions[action_index]
                E = 0
                for suc in action.next_states:
                    E = E + action.next_states[suc] * values[suc]
                value_tmp[state] = action.reward + gamma * E
            for state in self.min_states:
                action_index = min_strategy.state_to_action_index[state]
                action = state.actions[action_index]
                E = 0
                for suc in action.next_states:
                    E = E + action.next_states[suc] * values[suc]
                value_tmp[state] = action.reward + gamma * E
            new_max = -math.inf
            for state in self.max_states:
                new_max = max(new_max, abs(values[state] - value_tmp[state]))
            for state in self.min_states:
                new_max = max(new_max, abs(values[state] - value_tmp[state]))
            max_dif = new_max
            for state in self.max_states:
                values[state] = value_tmp[state]
            for state in self.min_states:
                values[state] = value_tmp[state]
        return values

    def extract_max_strategy(self, values, pre_max_strategy, gamma):
        new_max_strategy = Strategy()
        for state in self.max_states:
            pre_strategy_value = None
            max_value = -math.inf
            max_index = None
            for i in range(len(state.actions)):
                action = state.actions[i]
                new_value = action.reward
                for suc in action.next_states:
                    new_value += gamma * action.next_states[suc] * values[suc]
                if pre_max_strategy.state_to_action_index[state] == i:
                    pre_strategy_value = new_value
                if new_value >= max_value - self.equality_error:
                    max_index = i
                    max_value = new_value
            if pre_strategy_value >= max_value - self.equality_error:
                new_max_strategy.state_to_action_index[state] = pre_max_strategy.state_to_action_index[state]
            else:
                new_max_strategy.state_to_action_index[state] = max_index

        return new_max_strategy

    def extract_min_strategy(self, values, pre_min_strategy, gamma):
        new_min_strategy = Strategy()
        for state in self.min_states:
            pre_strategy_value = None
            min_value = math.inf
            min_index = None
            for i in range(len(state.actions)):
                action = state.actions[i]
                new_value = action.reward
                for suc in action.next_states:
                    new_value += gamma * action.next_states[suc] * values[suc]
                if pre_min_strategy.state_to_action_index[state] == i:
                    pre_strategy_value = new_value
                if new_value <= min_value + self.equality_error:
                    min_index = i
                    min_value = new_value
            if pre_strategy_value <= min_value + self.equality_error:
                new_min_strategy.state_to_action_index[state] = pre_min_strategy.state_to_action_index[state]
            else:
                new_min_strategy.state_to_action_index[state] = min_index

        return new_min_strategy

    def discounted_strategy_iteration(self, gamma, epsilon, initial_values, start_max_strategy=None):
        new_max_strategy = start_max_strategy
        if new_max_strategy is None:
            new_max_strategy = self.get_random_max_strategy()
        pre_max_strategy, optimal_min_strategy, values = None, None, None

        values = initial_values
        while pre_max_strategy != new_max_strategy:
            pre_max_strategy = new_max_strategy
            optimal_min_strategy, values = self.get_optimal_min_strategy(pre_max_strategy, gamma, epsilon, values)
            # values = self.extract_values(pre_max_strategy, optimal_min_strategy, gamma, epsilon)
            new_max_strategy = self.extract_max_strategy(values, pre_max_strategy, gamma)

        return {'max_strategy': new_max_strategy, 'min_strategy': optimal_min_strategy, 'rewards': values}

    def get_storm_model_fix_min_strategy(self, strategy, player):
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                              has_custom_row_grouping=True, row_groups=0)
        mdp_rewards = []
        number_of_choices = 0
        for state, player1 in self.states_in_order:
            builder.new_row_group(number_of_choices)
            if player1 is not player:
                for action in state.actions:
                    for next_state in action.next_states:
                        builder.add_next_value(number_of_choices, next_state.id, action.next_states[next_state])
                    number_of_choices += 1
                    mdp_rewards.append(action.reward)
            else:
                action_index = strategy.state_to_action_index[state]
                action = state.actions[action_index]
                for next_state in action.next_states:
                    builder.add_next_value(number_of_choices, next_state.id, action.next_states[next_state])
                number_of_choices += 1
                mdp_rewards.append(action.reward)
        builder.new_row_group(number_of_choices)
        builder.add_next_value(number_of_choices, self.get_number_of_states(), 1)
        mdp_rewards.append(0)

        transition_matrix = builder.build()
        reward_models = {'coin_flips': stormpy.SparseRewardModel(optional_state_action_reward_vector=mdp_rewards)}
        state_labeling = stormpy.storage.StateLabeling(self.get_number_of_states() + 1)
        components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                                   reward_models=reward_models, rate_transitions=False)
        mdp = stormpy.storage.SparseMdp(components)

        return mdp

    def is_optimal_mean_payoff(self, max_strategy, min_strategy, epsilon):
        storm_mdp = self.get_storm_model_fix_min_strategy(min_strategy, Player.MIN)
        formula_str = 'Rmax=? [ LRA ]'
        formula = stormpy.parse_properties(formula_str)[0]
        max_rewards = stormpy.check_model_sparse(storm_mdp, formula)

        storm_mdp = self.get_storm_model_fix_min_strategy(max_strategy, Player.MAX)
        formula_str = 'Rmin=? [ LRA ]'
        formula = stormpy.parse_properties(formula_str)[0]
        min_rewards = stormpy.check_model_sparse(storm_mdp, formula)
        min_rewards = (str(min_rewards.get_values()).replace(']', '').replace('[', '')
                       .replace(',', ''))
        max_rewards = (str(max_rewards.get_values()).replace(']', '').replace('[', '')
                       .replace(',', ''))
        min_rewards = np.array([float(s) for s in str(min_rewards).split() if s.replace(".", "").isnumeric()])
        max_rewards = np.array([float(s) for s in str(max_rewards).split() if s.replace(".", "").isnumeric()])
        return max(abs(min_rewards - max_rewards)) <= epsilon, min_rewards

    def mean_average_strategy_iteration(self, initial_gamma, epsilon, start_max_strategy=None):
        max_strategy = start_max_strategy
        if max_strategy is None:
            max_strategy = self.get_random_max_strategy()
        gamma = initial_gamma
        index = 0

        discounted_values = {x: 0 for x in self.max_states}
        discounted_values.update({x: 0 for x in self.min_states})
        while True:
            discounted_result = self.discounted_strategy_iteration(gamma, epsilon, discounted_values, max_strategy)
            max_strategy = discounted_result['max_strategy']
            min_strategy = discounted_result['min_strategy']
            discounted_values = discounted_result['rewards']
            is_optimal, rewards = self.is_optimal_mean_payoff(max_strategy, min_strategy, epsilon)
            if is_optimal:
                break
            index = index+1
            print(index)
            gamma = 1 - (1 - gamma) / 2

        return {'max_strategy': max_strategy, 'min_strategy': min_strategy, 'values': rewards}

    def discounted_strategy_iteration_with_fixed_max_strategy(self, max_strategy, gamma, epsilon, initial_values, start_strategy=None):
        new_min_strategy = start_strategy
        if new_min_strategy is None:
            new_min_strategy = self.get_random_min_strategy()
        pre_min_strategy = None

        values = initial_values
        while pre_min_strategy != new_min_strategy:
            pre_min_strategy = new_min_strategy
            values = self.extract_values(max_strategy, pre_min_strategy, gamma, epsilon, values)
            new_min_strategy = self.extract_min_strategy(values, pre_min_strategy, gamma)

        return new_min_strategy, values


if __name__ == '__main__':
    ac1 = Action(0)         #Action(reward)
    ac2 = Action(3)
    ac3 = Action(3)
    ac4 = Action(0)
    ac5 = Action(1)
    ac6 = Action(1)
    s1 = State()
    s2 = State()
    s3 = State()
    ac1.add_new_state(s2, 1)
    ac2.add_new_state(s1, 1)
    ac3.add_new_state(s3, 1)
    ac4.add_new_state(s2, 1)
    ac5.add_new_state(s1, 1)
    ac6.add_new_state(s3, 1)
    s1.add_action(ac1)
    s2.add_action(ac2)
    s3.add_action(ac4)
    s1.add_action(ac6)
    s2.add_action(ac3)
    s3.add_action(ac5)
    g = Game(10**-6)
    g.add_max_state(s2)
    g.add_min_state(s1)
    g.add_min_state(s3)
    g.check_game()  # TODO not necessary
    result = g.mean_average_strategy_iteration(0.1, 10**-6)
    max_strategy, min_strategy, rewards = result['max_strategy'], result['min_strategy'], result['values']
    print(min_strategy.state_to_action_index)
    print(max_strategy.state_to_action_index)
    print(rewards)
