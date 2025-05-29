import chess
import chess.pgn
import torch
from model import NeuralNetwork, ModelConfig
from encoder_decoder import encode, decode_from_index
from AlphaMCTS import UCT_Search, MCTSConfig
from datetime import datetime
import os

class SelfPlayPGNGenerator:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig()
        self.mcts_config = MCTSConfig()
        self.model = self._load_latest_checkpoint(checkpoint_dir)
        self.model.eval()
    
    def _load_latest_checkpoint(self, checkpoint_dir):
        """Load the most recent model checkpoint"""
        model = NeuralNetwork(self.config).to(self.device)
        
        checkpoint_path = 'chess_model (9).pt'
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model_state_dict'])
            print("[INFO] Loaded existing model checkpoint.")
        return model
    
    def play_game(self, num_simulations=700):
        """Play one full game and return the PGN"""
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "Self-play"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "AI"
        game.headers["Black"] = "AI"
        node = game
        
        while not board.is_game_over():

            best_move, _ = UCT_Search(
                board,
                num_simulations,
                self.model,
                self.mcts_config
            )
            
            move = decode_from_index(best_move.item())
            board.push(move)
            
            # Add to PGN
            node = node.add_variation(move)
        
        game.headers["Result"] = board.result()
        return game
    
    def save_pgn(self, pgn, filename=None):
        """Save PGN to file"""
        if filename is None:
            filename = f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        
        with open(filename, "w") as f:
            print(pgn, file=f)
        print(f"Saved PGN to {filename}")
        return filename

if __name__ == "__main__":
    player = SelfPlayPGNGenerator()
    
    game = player.play_game(num_simulations=100)
    player.save_pgn(game)
    
    # To play multiple games:
    # for i in range(5):
    #     game = player.play_game()
    #     player.save_pgn(game, filename=f"game_{i+1}.pgn")