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

"""Python Deep CFR example."""
import datetime
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np
import time

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr_tf2
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 1, "Number of iterations")
flags.DEFINE_integer("num_traversals", 3000, "Number of traversals/games")
flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")

####### current ############3

def main(unused_argv):

  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  start_time = time.perf_counter()

  root_file = f"/home/mor/PycharmProjects/open_spiel/algo_results/{datetime.datetime.now()}"
  save_strategy_memories = f"{root_file}/strategy"
  save_advantage_networks = f"{root_file}/advantage"
  save_lost = f"{root_file}/losses"

  os.makedirs(save_strategy_memories)
  os.makedirs(save_advantage_networks)
  os.makedirs(save_lost)

 # TODO: why they use adam istead of SGD:
      #  . We perform 4,000 mini-batch stochastic gradient descent (SGD) iterations using a batch size of 10,000 and perform parameter
      # updates using the Adam optimizer (Kingma & Ba, 2014)
      # with a learning rate of 0.001, with gradient norm clipping
      # to 1. For HULH we use 32,000 SGD iterations and a batch

 # todo: dont the same net aritecture + our is why smaller
 # todo: memory_capacity=4e7, int the article he saying 4e7 infosets for each player

  deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
      game,
      policy_network_layers=(16, 16),
      advantage_network_layers=(16, 16),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=20000,
      batch_size_strategy=20000,
      memory_capacity=int(4e7),
      policy_network_train_steps=20000,
      advantage_network_train_steps=20000,
      reinitialize_advantage_networks=True,
      save_strategy_memories=save_strategy_memories,
      save_advantage_networks=save_advantage_networks,
      infer_device="cpu",
      train_device="cpu")

  policies = []
  for i in range(2):
      policy_net, advantage_losses, policy_loss = deep_cfr_solver.solve()
      deep_cfr_solver.save_policy_network(root_file + '/policy_network_iter' + str(i))
      policies.append(policy_net)

      np.save(save_lost + f'/advantage_losses_{i}.npy', np.array(advantage_losses, dtype=object), allow_pickle=True)
      np.save(save_lost + f'/policy_losses_{i}.npy', policy_loss, allow_pickle=True)


  avg_lst = []
  for i in range(len(policies)):
      if i == len(policies)- 1:
          break
      returns_list1 = simulate_games(game, [policies[i],policies[i+1]], 5000)
      returns_list2 = simulate_games(game, [policies[i+1],policies[i]], 5000)

      logging.info( f"returns of {i} against {i+1}:\n"+ str(returns_list1))
      logging.info( f"returns of {i} against {i+1}:\n"+ str(returns_list2))
      returns_list2_opp = [[ret[1],ret[0]] for ret in returns_list2]
      returns=returns_list1 + returns_list2_opp
      avg1=sum([ret[0] for ret in returns])/len(returns)
      avg2=sum([ret[1] for ret in returns])/len(returns)
      logging.info( f"avg of returns of {i} against {i+1}:\n"+ str([avg1,avg2]))
      avg_lst.append([avg1, avg2])

  with open(f"{root_file}/bots_by_iterations_result.pkl", "wb") as f:
      pickle.dump(avg_lst, f)
  print(f"experiment time: {time.perf_counter() - start_time:0.4f} seconds")

def simulate_games(game, agents, games_num):
    returns_list = []
    for _ in range(games_num):
        returns_list.append(simulate(game, agents))

    return returns_list

def simulate(game, agents):
    state = game.new_initial_state()
    rng = np.random.RandomState(42)

    # Print the initial state
    logging.info("INITIAL STATE")
    logging.info( str( state ) )

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        current_player = state.current_player()
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len( outcomes )
            logging.info( "Chance node with " + str( num_actions ) + " outcomes" )
            action_list, prob_list = zip( *outcomes )
            action = rng.choice(action_list, p=prob_list)
            logging.info( f"Sampled outcome: {state.action_to_string( state.current_player(), action )}")
            state.apply_action( action )
        else:
            # Decision node: sample action for the single current player
            legal_actions = state.legal_actions()
            for action in legal_actions:
                logging.info( "Legal action: {} ({})".format(
                    state.action_to_string( current_player, action ), action ) )
            nn = agents[current_player]
            prob = deep_cfr_tf2.DeepCFRSolver.action_probabilities_s(nn, state)
            action_list, prob_list = zip(*prob)

            prob_list = [p/sum(prob_list) for p in prob_list]
            action = np.random.choice(action_list, p=list(prob_list))

            action_string = state.action_to_string( current_player, action )
            logging.info("Player " + str(current_player) + ", chose action: " + action_string)
            state.apply_action( action )

        logging.info( "" )
        logging.info( "NEXT STATE:" )
        logging.info( str( state ) )

    # Game is now done. logging.info utilities for each player
    returns = state.returns()
    for pid in range( game.num_players() ):
        logging.info( "Utility for player {} is {}".format( pid, returns[pid] ) )
    return returns

if __name__ == "__main__":
  app.run(main)