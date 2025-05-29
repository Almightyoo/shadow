# import torch
# import math
# from encoder_decoder import  encode_action, encode_board
# from collections import defaultdict
# import chess


# class UCTNode:
#     def __init__(self, board, move, parent = None):
#         self.board = board
#         self.move = move
#         self.is_expanded = False
#         self.parent = parent
#         self.children = {}
#         self.child_priors = torch.zeros(4864, dtype = torch.float32)
#         self.child_total_value = torch.zeros(4864, dtype = torch.float32)
#         self.child_number_visits = torch.zeros(4864, dtype = torch.float32)
#         self.action_idxes = []



#     # * propeties -> number_visit of this node
#     # * propeties -> total_value  of this node
#     @property
#     def number_visits(self):
#         return self.parent.child_number_visits[self.move]

#     @number_visits.setter
#     def number_visits(self, value):
#         self.parent.child_number_visits[self.move] = value
    
#     @property
#     def total_value(self):
#         return self.parent.child_total_value[self.move]
    
#     @total_value.setter
#     def total_value(self, value):
#         self.parent.child_total_value[self.move] = value



#     # * methods
#     def child_Q(self):
#         return self.child_total_value / (1 + self.child_number_visits)
    
#     def child_U(self):
#         return math.sqrt(self.number_visits) * (abs(self.child_priors) / (1 + self.child_number_visits))
    
#     def best_child(self):
#         combined = self.child_Q() + self.child_U()
#         if self.action_idxes:
#             bestmove = self.action_idxes[torch.argmax(combined[self.action_idxes])]
#         else:
#             bestmove = torch.argmax(combined)
#         return bestmove
    
#     def selection(self):
#         current = self
#         while current.is_expanded:
#           best_move = current.best_child()
#           current = current.maybe_add_child(best_move)
#         return current
    

#     #decode_n_move_pieces is going to take the index of the move and play the move return board
#     def maybe_add_child(self, move):
#         if move not in self.children:
#             copy_board = self.board
#             copy_board = self.decode_n_move_pieces(copy_board, move)
#             self.children[move] = UCTNode(copy_board, move, parent=self)
#         return self.children[move]
    
#     def expansion(self, child_priors):
#         self.is_expanded = True
#         action_idxs = []
#         c_p = child_priors
#         for action in self.board.legal_moves():
#             if not action:
#                 initial_pos, final_pos, underpromote = encode_action(action)
#                 action_idxs.append(encode_action(self.board, initial_pos, final_pos, underpromote))
#         if action_idxs == []:
#             self.is_expanded = False
#         self.action_idxs = action_idxs
#         for i in range(len(child_priors)):
#             if i not in action_idxs:
#                 c_p[i] = 0.0
#         self.child_priors = c_p

#     def backup(self, value_estimate: float):
#         current = self
#         while current.parent is not None:
#             current.number_visits += 1
#             if current.board.player == chess.WHITE: # same as current.parent.game.player = 0
#                 current.total_value += (1*value_estimate) # value estimate +1 = white win
#             elif current.board.player == chess.BLACK: # same as current.parent.game.player = 1
#                 current.total_value += (-1*value_estimate)
#             current = current.parent


    
# class DummyNode(object):
#     def __init__(self):
#         self.parent = None
#         self.child_total_value = defaultdict(float)
#         self.child_number_visits = defaultdict(float)

# def UCT_search(state, num_reads, net):
#     root = UCTNode(state, move = None, parent = DummyNode())
#     for i in range(num_reads):
#         leaf = root.selection
#         encoded_state = encode_board(leaf.board)
#         encoded_state = encoded_state.transpose(2,0,1)
#         child_priors, value_estimate = net(encoded_state)
#         if leaf.game.check_status() == True and leaf.game.in_check_possible_moves() == []:
#             leaf.backup(value_estimate); continue
#         leaf.expand(child_priors) # need to make sure valid moves
#         leaf.backup(value_estimate)
#     return torch.argmax(root.child_number_visits), root


# def MCTS_self_play(chessnet,num_games):
#     for idxx in range(0, num_games):
#         current_board = chess.Board()
#         dataset = []
#         states = []
#         value = 0
#         # you have to playout till termination add states best_move get, decode and move piece, get policy, get value


# print(defaultdict(float))