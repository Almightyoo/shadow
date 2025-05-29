import unittest
import chess
import torch
from encoder_decoder import encode, encode_action, decode_from_index
import random

import os
import matplotlib.pyplot as plt
import numpy as np


class TestChessEncoding(unittest.TestCase):    
    def test_board_encoding_shape(self):
        board = chess.Board()
        encoded = encode(board)
        self.assertEqual(encoded.shape, (22, 8, 8))

    def test_encode_decode_action(self):
        board = chess.Board()
        # arr = torch.zeros(76*8*8, dtype = torch.float32)
        for _ in range(100):
            board = chess.Board()
            while not board.is_game_over():
                for move in board.legal_moves:
                    print(f'move: {move}')
                    i, j = divmod(move.from_square, 8)
                    # print(f'initial position: {(i,j)}')
                    x, y = divmod(move.to_square, 8)
                    # print(f'initial position: {(x,y)}')
                    piece = board.piece_at(move.from_square)  

                    # print(move.promotion)
                    idx, x1, y1 = encode_action(piece.symbol(), (i, j), (x, y), move.promotion)
                    # print('promo',move.promotion)
                    index = idx*64 + (x1*8 + y1)
                    # print(index)
                    # print(f'encoded dims: {idx, x1, y1}')
                    # arr = arr.view(76,8,8)
                    # arr[idx][x1][y1]=1
                    # arr = arr.view(76*8*8)
                    # cookedindex = torch.where(arr == 1)
                    # print(f'action to index: {torch.where(arr == 1)}')
                    # print(f'calculated index: {index}')
                    
                    
                    decoded_move = decode_from_index(index)
                    # print(f'move after encoding and decoding: {move}')
                    self.assertEqual(decoded_move, move)
                move = random.choice(list(board.legal_moves))
                board.push(move)
        print(f'shits working clean af fuck yeah!!!!!!!')
            
import torch
import torch.nn as nn
from model import NeuralNetwork, ModelConfig


# * cant remember where I used it but okay
# def test_model_forward_pass():
#     config = ModelConfig()
#     config.n_channels = 22
#     config.n_filters = 128
#     config.n_BLOCKS = 3
#     config.SE_channels = 16
#     config.policy_channels = 76
    
#     model = NeuralNetwork(config)
    
#     batch_size = 2
#     input_tensor = torch.randn(batch_size, config.n_channels, 8, 8)
#     legal_mask = torch.ones(batch_size, 76*8*8)
    
#     value, policy, loss, policy_loss, value_loss = model(input_tensor, legal_mask)
#     assert value.shape == (batch_size,), f"Value shape should be ({batch_size},), got {value.shape}"
#     assert policy.shape == (batch_size, 76*8*8), f"Policy shape should be ({batch_size}, {76*8*8}), got {policy.shape}"
#     assert loss is None, "Loss should be None when no targets provided"
    
    
#     targets = {
#         'value': torch.randn(batch_size),
#         'policy': torch.randint(0, 76*8*8, (batch_size,))
#     }
#     value, policy, loss, policy_loss, value_loss = model(input_tensor, legal_mask, targets)
#     print(f'value: {value}')
#     print(f'policy: {policy.shape}')
#     assert isinstance(loss, torch.Tensor), "Loss should be a tensor when targets are provided"
#     assert loss.item() > 0, "Loss should be positive"
    
#     print("Forward pass test passed!")


# if __name__ == "__main__":
#     test_model_forward_pass()
#     print("All tests passed!")

# tce = TestChessEncoding()
# tce.test_encode_decode_action()






def inspect_selfplay_data(data_dir="selfplay_data", num_games_to_check=3):
    game_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    game_files = game_files[:num_games_to_check]
    
    if not game_files:
        print(f"No game files found in {data_dir}")
        return
    
    print(f"Found {len(game_files)} game files. Inspecting first {num_games_to_check}...")
    
    for i, game_file in enumerate(game_files):
        filepath = os.path.join(data_dir, game_file)
        data = torch.load(filepath)
        
        print(f"\nGame {i+1}: {game_file}")
        print(f"Number of positions: {len(data['values'])}")
        
        sample_indices = np.linspace(0, len(data['policies'])-1, 3, dtype=int)
        
        for idx in sample_indices:
            policy = data['policies'][idx]
            value = data['values'][idx].item()
            
            print(f"\nPosition {idx}:")
            print(f"Value (predicted outcome): {value:.2f}")
            print(f"Policy shape: {policy.shape}")
            
            plt.figure(figsize=(12, 4))
            plt.bar(range(len(policy)), policy.numpy())
            plt.title(f"Game {i+1}, Position {idx} Policy Distribution")
            plt.xlabel("Move Index")
            plt.ylabel("Probability")
            plt.show()
            
            topk_values, topk_indices = torch.topk(policy, 5)
            print("Top 5 moves:")
            for val, idx in zip(topk_values, topk_indices):
                print(f"  Move {idx.item()}: {val.item():.4f}")

if __name__ == "__main__":
    # inspect_selfplay_data()
    print(os.cpu_count())








    
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess.pgn
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = ModelConfig()
model = NeuralNetwork(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

checkpoint_path = '/kaggle/input/i-dont-know/pytorch/default/1/chess_model (7).pt'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("[INFO] Loaded existing model checkpoint.")
else:
    print('wth stop')

class ChessDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer  

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        board_tensor, (idx_action, x, y), value = self.buffer[idx]
        policy_target = torch.zeros(76, 8, 8)
        policy_target[idx_action, x, y] = 1
        return board_tensor.float(), policy_target.view(-1), torch.tensor([value], dtype=torch.float32)


data_buffer = []
game_count = 0
pgn = open('/kaggle/input/ccrldataset/CCRL.pgn')

while True:
    # print(f'playing game no. {game_count + 1}')
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    
    game_count += 1
    

    
    result = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[game.headers['Result']]
    board = game.board()

    for move in game.mainline_moves():
        encoded_board = encode(board).to(device)
        i, j = divmod(move.from_square, 8)
        x, y = divmod(move.to_square, 8)
        piece = board.piece_at(move.from_square)
        idx_action, x, y = encode_action(piece.symbol(), (i, j), (x, y), move.promotion)
        data_buffer.append((encoded_board, (idx_action, x, y), result))
        board.push(move)

    

    if game_count % 1000 == 0:
        print(f'\n[INFO] Training after {game_count} games ({len(data_buffer)} positions)')
        dataset = ChessDataset(data_buffer)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
        model.train()
        running_loss = 0
        pbar = tqdm(loader, desc=f"Epoch @ {game_count} games", dynamic_ncols=True)
    
        for i, (boards, policy_targets, values) in enumerate(pbar):
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            values = values.squeeze(-1).to(device)
    
            targets = {
                'value': values,
                'policy': policy_targets.argmax(dim=1)
            }
    
            _, _, loss, _, _ = model(boards, targets)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix(loss=f'{avg_loss:.4f}')
    
        torch.save(model.state_dict(), '/kaggle/working/chess_model.pt')
        data_buffer.clear()
