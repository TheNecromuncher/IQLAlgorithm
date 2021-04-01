import numpy as np
from game2dboard import Board
from tkinter import messagebox

NUM_ROWS = 20
NUM_COLS = 20
display = Board(NUM_COLS, NUM_ROWS)
GOAL_STATE = [0, NUM_COLS - 1]
START_STATE = [NUM_ROWS - 1, 0]

obstacle_states = [(5, 5), (2, 3), (2, 1), (2, 2), (2, 3)]
ACTIONS = [0, 1, 2, 3]

'''
for x in range(NUM_ROWS):
    for y in range(NUM_COLS):
        print(q_table[x][y])
'''


def state_reward(state):
    if state == GOAL_STATE:
        return 0
    else:
        return -1


def manhattan_distance(A, B):
    return abs(A[0] - B[0]) + abs(A[1] - B[1])


# both takes the action and verifies it is both within grid bounds and does not enter into an obstacle
def take_action(state, action):
    if action == 0:
        # right
        next_state = (state[0] + 1, state[1])
    elif action == 1:
        # up
        next_state = (state[0], state[1] - 1)
    elif action == 2:
        # left
        next_state = (state[0] - 1, state[1])
    elif action == 3:
        # down
        next_state = (state[0], state[1] + 1)
    else:
        print("this is not good this is not good")
        raise Exception("HE'S TRYING TO ESCAPE CALL THE COAST GUARD")
    # verifies action is allowed, i.e. not off of grid or into an obstacle
    if (next_state[0] >= 0) and (next_state[0] <= NUM_ROWS - 1):
        if (next_state[1] >= 0) and (next_state[1] <= NUM_COLS - 1):
            if next_state not in obstacle_states:
                return next_state
    print("NOT GOOD")
    return state


# chooses action that results in minimal manhattan distance from goal, preferring actions that involve no turns
# as described in paper
def calc_best_action(state, previous_state):
    resulting_states = []
    # take every possible action
    for action in ACTIONS:
        resulting_states.append([state, action, take_action(state, action)])
        if resulting_states[-1][0] == resulting_states[-1][2]:
            resulting_states.pop(-1)
    # check each resulting state's distance to goal
    for i in range(len(resulting_states)):
        # resulting_states[a][b]; a = state, b = distance to goal
        resulting_states[i] = [*resulting_states[i], manhattan_distance(resulting_states[i][2], GOAL_STATE)]
    # sort by distance to goal (by highest distance)
    resulting_states.sort(key=lambda x: x[3])
    #
    # resulting_states[i] = [STATE, ACTION, RESULTING STATE, DISTANCE TO GOAL]
    #
    # check if theres a tie for first
    if resulting_states[0][3] == resulting_states[1][3]:
        # if the first one checked has no turns, pick it
        if previous_state[0] == state[0] and state[0] == resulting_states[0][2][0]:
            return resulting_states[0][1]
        elif previous_state[1] == state[1] and state[1] == resulting_states[0][2][1]:
            return resulting_states[0][1]
        else:
            # the first one had turns, so pick the 2nd one
            # (if they both had turns then it's arbitrary, so this 2nd one is a safe pick regardless)
            return resulting_states[0][1]
    else:
        return resulting_states[0][1]


def generate_q_table():
    # In the IQL algorithm, oddly enough, since it is highly deterministic and almost every variable needed is given
    # at the start, the q_table becomes a useful data structure to store the optimal decision at a given grid location,
    # rather than as a complex data structure that is iterated upon to hone in on the optimal solution through a sort of
    # reinforcement learning... not very much like q-learning if you ask me! but, definitely inspired by it!
    #
    # as a result, the q table need only store ONE q value for the BEST action at any given (x, y) location
    # so, the size of the q table is [NUM_ROWS] + [NUM_COLS] = O(n), scales linearly as size increases, where a q_table
    # normally scales in a manner like [NUM_ROWS + NUM COLS] + NUM_ACTIONS (one entry per action)
    # which is going to be O(n^3) worst case or O(n^2) in a standard case considering it has to be iterated upon
    # in an episodic, unlocked fashion and every entry in the q table is updated
    q_table = np.random.uniform(low=-1, high=-1, size=[NUM_ROWS] + [NUM_COLS])
    q_table[GOAL_STATE[0]][GOAL_STATE[1]] = 0
    return q_table


def plan_actions(q_table):
    current_state = START_STATE
    previous_state = START_STATE
    for x in range(NUM_ROWS):
        for y in range(NUM_COLS):
            q_table[x][y] = calc_best_action([x, y], previous_state)
            previous_state = current_state
            current_state = [x, y]
    return q_table


def generate_board():
    display.cell_size = 25
    display.title = "IQL Algorithm"
    display.cursor = None
    display.margin = 10
    display.grid_color = "black"
    display.margin_color = "grey"
    display.cell_color = "white"
    display.fill(None)


def display_grid(state, action):
    x = state[0]
    y = state[1]
    if [y, x] == GOAL_STATE:
        display[x][y] = 'treasure_chest'
    if [y, x] == START_STATE:
        display[x][y] = 'dwarf'
    if (x, y) in obstacle_states:
        display[x][y] = 'spike_pit'
    if action == 0:
        display[y][x] = 'right_arrow'
    elif action == 1:
        display[y][x] = 'up_arrow'
    elif action == 2:
        display[y][x] = 'left_arrow'
    elif action == 3:
        display[y][x] = 'down_arrow'
    if (x, y) in obstacle_states:
        display[x][y] = 'spike_pit'


def main():
    # INITIALIZATION PHASE
    q_table = generate_q_table()
    # PLANNING PHASE (PATHFINDING PHASE)
    q_table = plan_actions(q_table)
    print(q_table)
    # TRAVEL PHASE (following the path & displaying result)
    # first, generate board
    generate_board()
    for x in range(NUM_ROWS):
        for y in range(NUM_COLS):
            if [y, x] == GOAL_STATE:
                display[x][y] = 'treasure_chest'
            if [y, x] == START_STATE:
                display[x][y] = 'dwarf'
            if (x, y) in obstacle_states:
                display[x][y] = 'spike_pit'
            if q_table[x][y] == 0:
                display[y][x] = 'right_arrow'
            elif q_table[x][y] == 1:
                display[y][x] = 'up_arrow'
            elif q_table[x][y] == 2:
                display[y][x] = 'left_arrow'
            elif q_table[x][y] == 3:
                display[y][x] = 'down_arrow'
            if (x, y) in obstacle_states:
                display[x][y] = 'spike_pit'

    display.show()
    print(display)


if __name__ == '__main__':
    main()
    # exec(open("old_alg.py").read())
