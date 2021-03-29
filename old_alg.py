import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# Also known as "step size", determines how quickly an agent learns, or rather, how much weight NEW information
# is given over old information. In a fully deterministic environment, a weight of 1 will produce the most optimal
# results, and thus it could be used here, however this is not in the spirit of q-learning! Additionally,
# I am discretizing a continuous environment so it is indeed rounded to a certain degree which would potentially
# cause some issues...
LEARNING_FACTOR = 0.1

# This determines the importance of future rewards - a value of 0 results in a simpleminded, greedy algorithm
# where a value of 1 can, particularly with a limited knowledge of the environment, lead to some infinitely long
# chains of actions and infinitely negative losses as it strives to find that terminal goal.
#
# a consistent approach is to start with a low number, to allow for some simple 'strategies' to develop, and gradually
# increase the discount factor to favor more 'long-term planning'
DISCOUNT_FACTOR = 0.95

# Simply the number of consecutive state and action sequences the agent attempts before resetting the environment
# to the initial condition and attempting another route. This is set much much higher than is needed, but terminates
# on reaching the goal, allowing for lots of exploration initially.
EPISODES = 25000

SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=DISCRETE_OS_SIZE + [env.action_space.n])


def get_discrete_state(state):
    internal_discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(internal_discrete_state.astype(int))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
            print(q_table)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            # Q_new = (1 - α) * Q(s_t, a_t) + α * (r_t + γ * maxQ(s_t+1, a))
            #
            # What this means is, the new Q value is simply the summation of three factors:
            #
            # current value weighted by the learning rate (determines the impact of the freshly-gathered information)
            #                                   (1 - α) * Q(s_t, a_t)
            # the reward obtained for when action a_t is taken while in state s_t weighed by learning rate
            #                                   (α * r(s_t, a_t))
            # the maximum reward that can be obtained in the next state weighted by learning rate AND discount factor
            # the discount factor increases the agent's leeway for exploration
            #                                   (αγ* max(Q(s_t+1, a))
            #
            # thus, the action (a_t) selected by the algorithm is decided at each time (t) based upon what it believes
            # is the best decision (or close to it, as the discount factor allows for exploration), and the result for
            # action is stored as r_t and the Q value for this new state is saved in the old one. The agent makes
            # essentially random decisions until finally a goal state is reached once, then the results that lead to
            # success are backpropogated and the real learning begins
            new_q = (1 - LEARNING_FACTOR) * current_q + LEARNING_FACTOR * (reward + DISCOUNT_FACTOR * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print("WE DID IT BOYS! Episode: ", episode)
            q_table[discrete_state + (action,)] = 0  # GOAL REACHED!!! a reward of 0, the highest praise
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
