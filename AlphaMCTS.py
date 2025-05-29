import torch
import chess
import math
import numpy as np
import copy
from collections import defaultdict
from encoder_decoder import encode_legal_moves, decode_from_index, encode
from model import NeuralNetwork as NN
from dataclasses import dataclass
import pickle
import datetime

from typing import List, Tuple



class Node():
    def __init__(self, board: chess.Board, parent = None, move = None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = {}
        self.child_priors = torch.zeros(4864, dtype = torch.float32)
        self.n = torch.zeros(4864, dtype = torch.float32)
        self.v = torch.zeros(4864, dtype = torch.float32)
        self.legal_moves = None
        self.is_expanded = False
        self.action_idxes = []

    @property
    def node_n(self):
        return self.parent.n[self.move]

    @node_n.setter
    def node_n(self, value):
        self.parent.n[self.move] = value
    
    @property
    def node_v(self):
        return self.parent.v[self.move]
    
    @node_v.setter
    def node_v(self, value):
        self.parent.v[self.move] = value

     # * methods
    def child_Q(self):
        return self.v / (1 + self.n)
    
    def child_U(self):
       return math.sqrt(self.node_n) * (abs(self.child_priors) / (1 + self.n))
    
    def best_child(self):
        
        combined = self.child_Q() + self.child_U()
        # print('combined_values: ',combined)
        # print(f'action_idxes: {self.action_idxes}')
        if self.action_idxes != []:
            bestmove = self.action_idxes[torch.argmax(combined[self.action_idxes])]
            bestmove = bestmove.item()
            # print(f'bestmove : {bestmove} from if')
        else:
            bestmove = torch.argmax(combined)
            bestmove = bestmove.item()
            # print(f'bestmove : {bestmove} from else')
        return bestmove
    
    def select(self):
        current = self
        while current.is_expanded:
          best_move = current.best_child()
        #   print(f'this is the best move: {best_move}')
          current = current.maybe_add_child(best_move)
        return current
    
    def maybe_add_child(self, move):
        if move not in self.children:
            # print(move)
            board_copy = copy.deepcopy(self.board)
            actual_move = decode_from_index(move)
            board_copy.push(actual_move)
            self.children[move] = Node(board_copy, move = move, parent = self)
        return self.children[move]


    def add_dirichlet_noise(self, action_idxs: torch.Tensor, child_priors: torch.Tensor) -> torch.Tensor:
        valid_child_priors = child_priors[action_idxs]
        noise = np.random.dirichlet(alpha=np.full(valid_child_priors.shape[0], 0.3))
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * torch.from_numpy(noise).float()
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    
    
    def expand(self, child_priors):
        self.is_expanded = True
        # print(f'node is expanded set to: {self.is_expanded}')
        self.legal_moves = list(self.board.legal_moves)
    
        if self.legal_moves == []:
            # print('set to false')
            self.is_expanded = False
        
        legal_move_mask = encode_legal_moves(self.board, False)
        # print(f'legal_mask_shape: {legal_move_mask.shape}')
        action_idxs = torch.where(legal_move_mask == 1)[0]
        self.action_idxes = action_idxs
        # print(f'action_idxes: {self.action_idxes.shape}')
        
        masked_priors = child_priors * legal_move_mask
        # print(masked_priors.shape)
        if self.parent is not None and self.parent.parent is None:
            masked_priors = self.add_dirichlet_noise(action_idxs, masked_priors)
        self.child_priors = masked_priors
        # print(masked_priors)


    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.node_n += 1
            if current.board.turn == chess.WHITE:
                current.node_v += value_estimate
            elif current.board.turn == chess.BLACK:
                current.node_v -= value_estimate
            current = current.parent

    
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.v = defaultdict(float)
        self.n = defaultdict(float)

def UCT_Search(game_state, num_reads, net, config):
    root = Node(game_state, move = None, parent = DummyNode())
    for i in range(num_reads):
        if config.debug: print('==================================')
        if config.debug: print(f'this is the {i} iteration', '\n\n')
        leaf = root.select()
        if config.debug: print(f'leaf_selected with leaf_expanded {leaf.is_expanded}: ', leaf.board)
        encoded_s = encode(leaf.board).unsqueeze(0)
        v, p,_,_,_ = net(encoded_s)
        p = p.squeeze(0)
        v = v.item()
        if config.debug: print(f'policy from neural network : {p.shape}')
        if config.debug: print(f'value: {v}')
        if leaf.board.is_game_over():
            leaf.backup(v)
        leaf.expand(p)
        leaf.backup(v)
    return torch.argmax(root.n), root


def get_policy(root):
    policy = torch.zeros(4864, dtype=torch.float32)
    for idx in torch.where(root.n!=0)[0]:
        policy[idx] = root.n[idx]/root.n.sum()
    return policy


@dataclass
class ModelConfig:
    n_channels = 22
    n_filters = 256
    n_BLOCKS = 5
    SE_channels = 32
    policy_channels = 76

@dataclass
class MCTSConfig:
    debug = False
    NUM_SIMULATIONS = 2
    MAX_MOVES = 200


# config = ModelConfig()
# # mcts_config = MCTSConfig()
# board = chess.Board()
# net = NN(config)
# encoded_s = encode(board).unsqueeze(0)
# # print(encoded_s.shape)
# v,p,_,_,_ = net(encoded_s)
# # print(p.shape,v.shape)
# p = p.squeeze(0)
# v = v.item()
# print(p.shape,v)
# root = Node(board, move = None, parent = DummyNode())
# leaf = root.select()
# print(leaf.board)
# print(leaf.n, leaf.v)
# leaf.expand(p)
# print(leaf.child_priors[457])
# best_move, root = UCT_Search(board, 10, net, mcts_config)
# policy = get_policy(root)
# print(policy[10])
# print(root.n)
# print(f'best_move: {best_move}')
# print(f'root.n: {root.n[root.action_idxes]}')
# print(f'root.v: {root.v[root.action_idxes]}')
# print(f'root.priors: {root.child_priors[root.action_idxes]}')

def MCTS_self_play(model, num_games, device, config):
    for i in range(num_games):
        # print(f'<=======game no {i}=============>')
        current_board = chess.Board()
        dataset = []
        value = 0
        i=0
        while not current_board.is_game_over():
            print('move no',i)
            i+=1
            board_state = encode(current_board).to(device)
            best_move, root = UCT_Search(current_board, 100, model, config)
            best_move = best_move.item()
            move = decode_from_index(best_move)
            current_board.push(move)

            policy = get_policy(root)
            dataset.append([board_state.cpu(), policy.cpu()])

        value = {'1-0':1, '0-1':-1, '1/2-1/2': 0}.get(current_board.result(),0)

        dataset_p = []
        for idx,(s,p) in enumerate(dataset):
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset

        filename = f"selfplay_game_{i}_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(dataset_p, f)

        print(f"Saved game {i} to {filename}")

@dataclass
class ModelConfig:
    n_channels = 22
    n_filters = 256
    n_BLOCKS = 5
    SE_channels = 32
    policy_channels = 76

import torch.multiprocessing as mp
import os


@dataclass
class MCTSConfig:
    debug = False

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = ModelConfig()
    net = NN(config)
    if torch.cuda.is_available():
        net.cuda()
    mcts_config = MCTSConfig()
    net.eval()

    processes = []
    for i in range(8):
        p = mp.Process(target=MCTS_self_play, args=(net, 5, i, mcts_config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()




def generate_game(model, device) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    board = chess.Board()
    game_data = []
    mcts_config = MCTSConfig()
    i=0
    while not board.is_game_over() and i < mcts_config.MAX_MOVES:
        i+=1
        print(f'move no. {i}')
        board_state = encode(board).to(device)
        best_move, root = UCT_Search(board, mcts_config.NUM_SIMULATIONS, model, mcts_config)
        best_move = best_move.item()
        move = decode_from_index(best_move)
        board.push(move)
        policy = get_policy(root).cpu()
        game_data.append([board_state, policy])
    
    value = {'1-0':1, '0-1':-1, '1/2-1/2': 0}.get(board.result(),0)
    return [(s, p, value) for s, p in game_data]





# * test random game
# import random
# board = chess.Board()
# i=0
# while not board.is_game_over():
#     moves = list(board.legal_moves)
#     moveint = torch.randint(len(moves),(1,1)).item()
#     move = moves[moveint]
#     board.push(move)
#     print(i)
#     i+=1

