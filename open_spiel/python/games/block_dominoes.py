# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Block Dominoes implemented in Python.

https://en.wikipedia.org/wiki/Dominoes#Blocking_game
"""

import copy

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_PIPS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# The first player to play is the one holding the highest rank tile.
# The rank of tiles is the following:
#   1. Highest double.
#   2. If none of the players hold a double, then highest weight.
#   3. If the highest weighted tile of both players has the same weight
#      then the highest single edge of the highest weighted tile.

# full deck sorted by rank:
_DECK = [(6., 6.), (5., 5.), (4., 4.), (3., 3.), (2., 2.), (1., 1.), (0., 0.),
         (5., 6.),
         (4., 6.),
         (3., 6.), (4., 5.),
         (2., 6.), (3., 5.),
         (1., 6.), (2., 5.), (3., 4.),
         (0., 6.), (1., 5.), (2., 4.),
         (0., 5.), (1., 4.), (2., 3.),
         (0., 4.), (1., 3.),
         (0., 3.), (1., 2.),
         (0., 2.),
         (0., 1.)]

_EDGES = [None, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

_POINTS_FOR_DOMINO = 10.


class Action:
  """Represent player possible action."""

  def __init__(self, player, tile, edge):
    self.player = player
    self.tile = tile
    self.edge = edge

  def __str__(self):
    return f"p{self.player} tile:{self.tile} pip:{self.edge}"

  def __repr__(self):
    return self.__str__()


def create_possible_actions():
  actions = []
  for player in range(_NUM_PLAYERS):
    for tile in _DECK:
      for edge in _EDGES:
        if edge in tile or edge is None:  # can we play tile on edge?
          actions.append(Action(player, tile, edge))
  return actions


_ACTIONS = create_possible_actions()
_ACTIONS_STR = [str(action) for action in _ACTIONS]

_HAND_SIZE = 7

_MAX_GAME_LENGTH = 28

_GAME_TYPE = pyspiel.GameType(
    short_name="python_block_dominoes",
    long_name="Python block dominoes",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(_ACTIONS),
    max_chance_outcomes=len(_DECK),
    # first player hand: (6,6) (6,5) (5,5) (6,4) (4,5) (6,3) (4,4)
    # second player hand is empty. can be reduced.
    min_utility=-69,
    max_utility=69,
    num_players=_NUM_PLAYERS,
    # deal: 14 chance nodes + play: 14 player nodes
    max_game_length=_MAX_GAME_LENGTH,
    utility_sum=0.0,
)


class BlockDominoesGame(pyspiel.Game):
  """A Python version of Block Dominoes."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return BlockDominoesState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return BlockDominoesObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params
    )


class BlockDominoesState(pyspiel.State):
  """A python version of the Block Dominoes state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.actions_history = []
    self.open_edges = []
    self.hands = [[], []]
    self.deck = copy.deepcopy(_DECK)
    self._game_over = False
    self._next_player = pyspiel.PlayerId.CHANCE
    self.blocked_pips = [[False] * len(_PIPS), [False] * len(_PIPS)]


  def get_other_edge(self, edge):
    assert len(self.open_edges) != 0
    i = self.open_edges.index(edge)
    return self.open_edges[1-i]
    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    if len(self.deck) > 14:
      return pyspiel.PlayerId.CHANCE
    return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    assert player == self._next_player
    return self.get_legal_actions(player)

  def get_legal_actions_helper(self, player):
    """Returns a list of legal actions."""
    assert player >= 0

    actions = []
    hand = self.hands[player]

    # first move, no open edges
    if not self.open_edges:
      for tile in hand:
        actions.append(Action(player, tile, None))
    else:
      for tile in hand:
        if tile[0] in self.open_edges:
          actions.append(Action(player, tile, tile[0]))
        if tile[0] != tile[1] and tile[1] in self.open_edges:
          actions.append(Action(player, tile, tile[1]))

    return actions



  def get_legal_actions(self, player):
    """Returns a list of legal actions."""
    assert player >= 0

    actions = []
    hand = self.hands[player]

    # first move, no open edges
    if not self.open_edges:
      for tile in hand:
        actions.append(Action(player, tile, None))
    else:
      for tile in hand:
        if tile[0] in self.open_edges:
          actions.append(Action(player, tile, tile[0]))
        if tile[0] != tile[1] and tile[1] in self.open_edges:
          actions.append(Action(player, tile, tile[1]))

    actions_idx = [_ACTIONS_STR.index(str(action)) for action in actions]
    actions_idx.sort()
    return actions_idx

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    p = 1.0 / len(self.deck)
    return [(_DECK.index(i), p) for i in self.deck]


  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      hand_to_add_tile = (
          self.hands[0] if len(self.hands[0]) != _HAND_SIZE else self.hands[1]
      )
      tile = _DECK[action]
      self.deck.remove(tile)
      hand_to_add_tile.append(tile)

      if not len(self.hands[0]) == len(self.hands[1]) == _HAND_SIZE:
        return  # another tiles to deal

      # check which hand is playing first, and assigned it to player 0
      hand0_starting_value = min(map(_DECK.index, self.hands[0]))
      hand1_starting_value = min(map(_DECK.index, self.hands[1]))
      if hand0_starting_value > hand1_starting_value:
        self.hands[0], self.hands[1] = self.hands[1], self.hands[0]

      for hand in self.hands:
        hand.sort()

      self._next_player = 0
    else:
      action = _ACTIONS[action]
      self.actions_history.append(action)
      my_idx = self.current_player()
      my_hand = self.hands[my_idx]
      my_hand.remove(action.tile)
      self.update_open_edges(action)

      if not my_hand:
        self._game_over = True  # player played his last tile
        return

      opp_idx = 1 - my_idx
      opp_legal_actions = self.get_legal_actions(opp_idx)

      if opp_legal_actions:
        self._next_player = opp_idx
        return

      for i in self.open_edges:
        self.blocked_pips[opp_idx][int(i)] = True

      my_legal_actions = self.get_legal_actions(my_idx)
      if my_legal_actions:
        self._next_player = my_idx
        return

      self._game_over = True  # both players are blocked

  def update_open_edges(self, action):
    if not self.open_edges:
      self.open_edges = list(action.tile)
    else:
      self.open_edges.remove(action.edge)
      new_edge = (
          action.tile[0] if action.tile[0] != action.edge else action.tile[1]
      )
      self.open_edges.append(new_edge)

    self.open_edges.sort()

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal {_DECK[action]}"
    return _ACTIONS_STR[action]

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""

    if not self.is_terminal():
      return [0, 0]

     sum_of_pips = [sum(t[0] + t[1] for t in self.hands[0]), sum(t[0] + t[1] for t in self.hands[1])]

    if sum_of_pips[0] == sum_of_pips[1]:
      return [0, 0]

    if sum_of_pips[1] > sum_of_pips[0]:
      winner_id = 0
    else:
      winner_id = 1

    if len(self.hands[winner_id]) == 0:








    return [-sum_of_pips0, sum_of_pips0]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    hand0 = [str(c) for c in self.hands[0]]
    hand1 = [str(c) for c in self.hands[1]]
    history = [str(a) for a in self.actions_history]
    return f"hand0:{hand0} hand1:{hand1} history:{history}"


class BlockDominoesObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", 2, (2,)),
              ("count_unseen_pips", 7, (7,)),
              ("hand", 21, (7, 3)),
              ("hand_sizes", 2, (2,)),
              ("actions", 35, (7, 5)),
              ("edges", 3, (3,))]

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index : index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

    self.tensor.fill(0)

    if "player" in self.dict:
      self.dict["player"][player] = 1
      self.dict["player"][1 - player] = 0


      if "count_unseen_pips" in self.dict:
        pips = [7.] * 7
        for action in state.actions_history:
          pips[int(action.tile[0])] -=1.
          if action.tile[0] != action.tile[1]: # double only count ones
            pips[int(action.tile[1])] -=1.

        for tile in state.hands[player]:
          pips[int(tile[0])] -=1.
          if tile[0] != tile[1]: # double only count ones
            pips[int(tile[1])] -=1.

        # todo: consider blocks

        self.dict["count_unseen_pips"] = np.array(pips)

    if "hand_sizes" in self.dict:
      my_hand_size = len(state.hands[player])
      opp_hand_size = len(state.hands[1 - player])
      self.dict["hand_sizes"][0] = my_hand_size/7.
      self.dict["hand_sizes"][1] = opp_hand_size/7.

    if "edges" in self.dict:
      if state.open_edges:
        self.dict["edges"][0] = state.open_edges[0]/6.
        self.dict["edges"][1] = state.open_edges[1]/6.
        self.dict["edges"][2] = 1.
      else:
        self.dict["edges"][0] = 0.0
        self.dict["edges"][1] = 0.0
        self.dict["edges"][2] = 0.0


    if "hand" in self.dict:
      for i, tile in enumerate(state.hands[player]):
        self.dict["hand"][i][0] = tile[0]/6.
        self.dict["hand"][i][1] = tile[1]/6.
        self.dict["hand"][i][2] = 1.0

    if "actions" in self.dict:
      for i, action in enumerate(state.get_legal_actions_helper(player)):
        self.dict["actions"][i][0] = action.tile[0]/6.
        self.dict["actions"][i][1] = action.tile[1]/6.

        if action.edge is None:
          self.dict["actions"][i][2] = action.tile[0]/6. # first action in game, open edges after the action are the played tile pips
          self.dict["actions"][i][3] = action.tile[1]/6.
        else:
          self.dict["actions"][i][2] = action.tile[1]/6. if action.tile[0] == action.edge else action.tile[0]/6.
          self.dict["actions"][i][3] = state.get_other_edge(action.edge)/6. if action.edge is not None else 0.0

        self.dict["actions"][i][4] = 1.0 # possible action


  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "hand" in self.dict:
      pieces.append(f"hand:{state.hands[player]}")
    if "actions_history" in self.dict:
      pieces.append(f"history:{str(state.actions_history)}")
    if "last_action" in self.dict and state.actions_history:
      pieces.append(f"last_action:{str(state.actions_history[-1])}")
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, BlockDominoesGame)
