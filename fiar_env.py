from enum import Enum
import numpy as np
# import matplotlib.pyplot as plt
import gym
from gym import spaces
from scipy import ndimage
import rendering
import state_utils

# from numpy.random import choice_

BLACK = 0
WHITE = 1
TURN_CHNL = 2
INVD_CHNL = 3
DONE_CHNL = 4
NUM_CHNLS = 5


def action2d_ize(action):
    map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9).reshape(9, 4).T)
    print(map)

    action2d = np.where(map == action)
    return int(action2d[0]), int(action2d[1])

def action1d_ize(action):
    map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9).reshape(9, 4).T)
    return map[action[0], action[1]]

def winning(state, player=0):
    if np.all(state[TURN_CHNL] == player):
        return 1
    else:
        return -1

def turn(state):
    """
    :param state:
    :return: Who's turn it is (govars.BLACK/govars.WHITE)
    """
    return int(np.max(state[TURN_CHNL]))

def areas(state):
    '''
    Return black area, white area
    '''

    all_pieces = np.sum(state[[BLACK, WHITE]], axis=0)
    empties = 1 - all_pieces

    empty_labels, num_empty_areas = ndimage.measurements.label(empties)

    black_area, white_area = np.sum(state[BLACK]), np.sum(state[WHITE])
    for label in range(1, num_empty_areas + 1):
        empty_area = empty_labels == label
        neighbors = ndimage.binary_dilation(empty_area)
        black_claim = False
        white_claim = False
        if (state[BLACK] * neighbors > 0).any():
            black_claim = True
        if (state[WHITE] * neighbors > 0).any():
            white_claim = True
        if black_claim and not white_claim:
            black_area += np.sum(empty_area)
        elif white_claim and not black_claim:
            white_area += np.sum(empty_area)

    return black_area, white_area

def fiar_check(state, loc=False):
    # check four in a row
    black_white = 1 if np.all(state[2]==1) else 0   # 0:black 1:white

    state = np.copy(state[black_white,:,:])

    def horizontal_check(state_b, loc=False):
        for i in range(state_b.shape[0]):
            fiar_ = 0
            continuous_ = False
            loc_set = []
            for j in range(state_b.shape[1]):
                if state_b[i,j] == 1:
                    fiar_ += 1
                    continuous_ = True
                    loc_set.append([i,j])
                else:
                    fiar_ = 0
                    continuous_ = False
                    loc_set = []
                if (fiar_ == 4) and continuous_:
                    if loc:
                        return True, loc_set
                    else:
                        return True
        if loc:
            return False, loc_set
        else:
            return False

    def vertical_check(state_b, loc=False):
        for j in range(state_b.shape[1]):
            fiar_ = 0
            continuous_ = False
            loc_set = []
            for i in range(state_b.shape[0]):
                if state_b[i,j] == 1:
                    fiar_ += 1
                    continuous_ = True
                    loc_set.append([i,j])
                else:
                    fiar_ = 0
                    continuous_ = False
                    loc_set = []
                if (fiar_ == 4) and continuous_:
                    if loc:
                        return True, loc_set
                    else:
                        return True
        if loc:
            return False, loc_set
        else:
            return False

    # check_board = np.zeros((9,4))
    def horizontal_11to4_check(state_b, loc=False):
        for offset_i in range(max(state_b.shape[1],state_b.shape[0])-1,-1,-1):
            fiar_ = 0
            continuous_ = False
            loc_set = []
            for j in range(max(state_b.shape[1],state_b.shape[0])):
                i = offset_i + j
                if i<state_b.shape[0] and j <state_b.shape[1]:
                    if i>=state_b.shape[0]:
                        break
                    # check_board[i,j]=1
                    if state_b[i,j] == 1:
                        fiar_ += 1
                        continuous_ = True
                        loc_set.append([i,j])
                    else:
                        fiar_ = 0
                        continuous_ = False
                        loc_set = []
                    if (fiar_ == 4) and continuous_:
                        if loc:
                            return True, loc_set
                        else:
                            return True
        if loc:
            return False, loc_set
        else:
            return False

    def horizontal_1to7_check(state_b, loc=False):
        # for offset_j in range(state_b.shape[1]-1,-1,-1): # 3
        for offset_j in range(max(state_b.shape[1],state_b.shape[0])): # 3
            fiar_ = 0
            continuous_ = False
            loc_set = []
            for i in range(max(state_b.shape[1],state_b.shape[0])):
                j = offset_j - i
                if i<state_b.shape[0] and j <state_b.shape[1]:
                    if j<0:
                        break
                    # check_board[i,j]=1
                    if state_b[i,j] == 1:
                        fiar_ += 1
                        continuous_ = True
                        loc_set.append([i,j])
                    else:
                        fiar_ = 0
                        continuous_ = False
                        loc_set = []
                    if (fiar_ == 4) and continuous_:
                        if loc:
                            return True, loc_set
                        else:
                            return True
        if loc:
            return False, loc_set
        else:
            return False

    if loc is False:
        return 1 if np.any([horizontal_check(state), vertical_check(state), horizontal_11to4_check(state), horizontal_1to7_check(state),]) else 0
    else:
        switch, locset =vertical_check(state,loc=True)
        if switch:
            return switch, locset
        switch, locset =horizontal_check(state,loc=True)
        if switch:
            return switch, locset
        switch, locset =horizontal_11to4_check(state,loc=True)
        if switch:
            return switch, locset
        switch, locset =horizontal_1to7_check(state,loc=True)
        if switch:
            return switch, locset
        return False, []


def next_state(state, action1d):
    # Deep copy the state to modify
    state = np.copy(state)

    # Initialize basic variables
    board_shape = state.shape[1:]
    pass_idx = np.prod(board_shape)
    action2d = action1d % board_shape[0], action1d // board_shape[0]  # previous

    player = turn(state)
    ko_protect = None

    # Assert move is valid
    if not ((0 <= action2d[0] < state.shape[1]) and (0 <= action2d[1] < state.shape[2])):
        raise ValueError("Invalid move", action2d)

    # Add piece
    state[player, action2d[0], action2d[1]] = 1

    # Update FIAR ending status
    state[DONE_CHNL] = fiar_check(state)

    if np.any(state[DONE_CHNL] == 0): # proceed if it is not ended
        # Switch turn
        state_utils.set_turn(state)

    return state


def game_ended(state):
    """
    :param state:
    :return: 0/1 = game not ended / game ended respectively
    """
    m, n = state.shape[1:]
    return int(np.count_nonzero(state[4] == 1) >= 1)


def invalid_moves(state):
    # return a fixed size binary vector
    if game_ended(state):
        return np.zeros(action_size(state))
    return np.append(state[INVD_CHNL].flatten(), 0)

def str_(state):
    board_str = ' '

    size_x = state.shape[1]
    size_y = state.shape[2]
    for i in range(size_y):  # 행 수에 따라 반복
        board_str += '   {}'.format(i)
    board_str += '\n  '
    board_str += '----' * size_y + '-'
    board_str += '\n'
    for j in range(size_x):  # 열 수에 따라 반복
        board_str += '{} |'.format(j)
        for i in range(size_y):
            if state[0, j, i] == 1:
                board_str += ' B'
            elif state[1, j, i] == 1:
                board_str += ' W'
            elif state[2, j, i] == 1:
                board_str += ' .'
            else:
                board_str += '  '

            board_str += ' |'

        board_str += '\n  '
        board_str += '----' * size_y + '-'
        board_str += '\n'

    done = game_ended(state)
    t = turn(state)
    board_str += '\tTurn: {}, Game Over: {}\n'.format('B' if t == 0 else 'W', done)
    return board_str



def action_size(state=None, board_size: int = None):
    # return number of actions
    if state is not None:
        m, n = state.shape[1:]
    elif board_size is not None:
        m, n = board_size, board_size
    else:
        raise RuntimeError('No argument passed')
    # return m * n + 1
    return m * n


class Fiar(gym.Env):
    def __init__(self, player=0):
        self.player = player    #  0: black,  1: white

        self.state_ = self.init_state()
        self.observation_space = spaces.Box(np.float32(0), np.float32(NUM_CHNLS),
                                                shape=(NUM_CHNLS, 9, 4))
        self.action_space = spaces.Discrete(action_size(self.state_))
        self.done = False
        self.action = None
        self.states = {}
        self.current_player = self.state_[0]
        self.players = [0, 1]  # player1 and player2

        # end, winner = self.game_end()


    def do_move(self, move):

        self.availables = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9)).tolist()
        self.states[move] = self.player
        self.availables.remove(move)
        self.player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move


    def game_end(self):
        winner = winning(self.state_, self.player)
        if winner != 0:
            return True, winner
        elif not len(self.observation_space.shape):
            return True
        return False

    def winner(self):
        """
        Get's the winner in BLACK's perspective
        :return: 1 for black's win, -1 for white's win
        """
        if self.game_ended():
            return winning(self.state_, self.player )
        else:
            return 0

    def init_state(self):
        """
        The state of the game is a numpy array
        * Are values are either 0 or 1

        * Shape [NUM_CHNLS, SIZE, SIZE]

        CHANNELS:
        0 - Black pieces
        1 - White pieces
        2 - Turn (0: black, 1: white)
        3 - Invalid moves (including ko-protection)
        4 - Game over

        SIZE: 9,4 since it is 4x9 board
        """
        return np.zeros((5,9,4))

    def step(self, action=None):
        '''
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info
        '''
        assert not self.done

        if action is None:
            # Generate a random action1d that is not equal to the previous one
            possible_actions = list(range(action_size(self.state_)))
            if self.action is not None:
                action1d = np.random.choice(possible_actions)
        else:
            if isinstance(action, tuple):  # Check if action is a tuple
                action1d = action[0] + action[1] * 9  # Convert 2D to 1D
            else:
                action1d = action

        self.state_ = next_state(self.state_, action1d)
        self.done = game_ended(self.state_)

        return np.copy(self.state_), self.reward(), self.done, self.info()

    def reset(self):
        '''
        Reset state, go_board, curr_player, prev_player_passed,
        done, return state
        '''
        self.state_ = self.init_state()
        self.done = False
        return np.copy(self.state_)


    def info(self):
        """
        :return: Debugging info for the state
        """
        return {
            'turn': turn(self.state_),
            'invalid_moves': invalid_moves(self.state_)
        }

    def state(self):
        """
        :return: copy of state
        """
        return np.copy(self.state_)

    def game_ended(self):
        return self.done


    def __str__(self):
        return str_(self.state_)

    def reward(self):
        return self.winner()

    def close(self):
        if hasattr(self, 'window'):
            assert hasattr(self, 'pyglet')
            self.window.close()
            self.pyglet.app.exit()

    def render(self,mode='terminal'):
        if mode == 'terminal':
            print(self.__str__())

            return None
        elif mode == 'human':

            import pyglet
            from pyglet.window import mouse
            from pyglet.window import key

            screen = pyglet.canvas.get_display().get_default_screen()
            # window_width = int(min(screen.width, screen.height) * 2 / 3)
            # window_height = int(window_width * 1.2)
            window_height = int(min(screen.width, screen.height) * 2 / 4)
            window_width = int(window_height * 2)
            # window_height = int(window_height * 1.5)
            window = pyglet.window.Window(window_width, window_height)
            # [1920,1080] monitor --> [1080,540] window

            self.window = window
            self.pyglet = pyglet
            self.user_action = None

            # Set Cursor
            cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
            window.set_mouse_cursor(cursor)

            # Outlines
            lower_x_grid_coord = window_width * 0.075 # [1920,1080] monitor --> [81,] and [999,]
            board_x_size = window_width * 0.85 # 918
            upper_x_grid_coord = board_x_size + lower_x_grid_coord  # [1920,1080] monitor --> [81,] and [999,]

            delta = board_x_size / (9.0 - 1) # board_size / (self.size - 1)

            lower_y_grid_coord = window_height * 0.075 # [1920,1080] monitor --> [81,40.5] and [999,499.5]
            board_y_size = window_height * 0.85 # 459
            upper_y_grid_coord = lower_y_grid_coord + delta*(4.0-1) # [1920,1080] monitor --> [81,40.5] and [999,499.5]

            piece_r = delta / 3.3  # radius

            @window.event
            def on_draw():
                pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
                window.clear()

                pyglet.gl.glLineWidth(3)
                batch = pyglet.graphics.Batch()

                # Define the new grid dimensions
                grid_size = (9, 4)

                # Update the grid rendering
                rendering.draw_grid(batch, delta, grid_size, [lower_x_grid_coord, upper_x_grid_coord],
                                    [lower_y_grid_coord, upper_y_grid_coord])

                # Info on top of the board
                rendering.draw_info(batch, window_width, window_height, upper_y_grid_coord, self.state_)

                # Inform user what they can do
                rendering.draw_command_labels(batch, window_width, window_height)

                # Draw the title
                rendering.draw_title(batch, window_width, window_height)

                # Draw the pieces
                rendering.draw_pieces(batch, [lower_x_grid_coord, lower_y_grid_coord], delta, piece_r, grid_size,
                                      self.state_)

                batch.draw()

            @window.event
            def on_mouse_press(x, y, button, modifiers):
                if button == mouse.LEFT:
                    grid_x = (x - lower_x_grid_coord)
                    grid_y = (y - lower_y_grid_coord)
                    x_coord = round(grid_x / delta)
                    y_coord = round(grid_y / delta)
                    print('PRESSED:\n [' + str(x) + ',' + str(y) + ']')
                    print('GRID:   \n [' + str(grid_x) + ',' + str(grid_y) + ']')
                    print('COORD:  \n [' + str(x_coord) + ',' + str(y_coord) + ']')
                    try:
                        self.window.close()
                        pyglet.app.exit()
                        self.user_action = (x_coord, y_coord)
                    except:
                        pass

            @window.event
            def on_key_press(symbol, modifiers):
                if symbol == key.P:
                    self.window.close()
                    pyglet.app.exit()
                    self.user_action = None
                elif symbol == key.R:
                    self.reset()
                    self.window.close()
                    pyglet.app.exit()
                elif symbol == key.E:
                    self.window.close()
                    pyglet.app.exit()
                    self.user_action = -1

            pyglet.app.run()

            return self.user_action