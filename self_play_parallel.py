from AlphaMCTS import UCT_Search, get_policy
from model import NeuralNetwork as NN
from encoder_decoder import decode_from_index, encode
import datetime
import torch
import torch.multiprocessing as mp
import chess
from typing import List, Tuple, Dict
import os
from dataclasses import dataclass
import copy
from queue import Empty

mp.set_start_method('spawn', force=True)

@dataclass
class ModelConfig:
    n_channels: int = 22
    n_filters: int = 256
    n_BLOCKS: int = 5
    SE_channels: int = 32
    policy_channels: int = 76

@dataclass
class MCTSConfig:
    debug: bool = False
    num_simulations: int = 100
    cpuct: float = 1.0

class SelfPlayWorker:
    def __init__(self, model, worker_id: int, config: ModelConfig, mcts_config: MCTSConfig) -> None:

        self.model = copy.deepcopy(model)
        self.worker_id = worker_id
        self.config = config
        self.mcts_config = mcts_config
        self.device = f'cuda:{worker_id}' if torch.cuda.device_count() > worker_id else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def run_game(self) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        current_board = chess.Board()
        game_data = []
        i=0
        while not current_board.is_game_over():
            print(f'move {i}')
            i+=1
            board_state = encode(current_board).to(self.device)
            
            best_move, root = UCT_Search(current_board, self.mcts_config.num_simulations, self.model, self.mcts_config)
            policy = get_policy(root).to(self.device)
            
            
            game_data.append((
                board_state.cpu(),
                policy.cpu(),
                0.0
            ))
            
            move = decode_from_index(best_move.item())
            current_board.push(move)
        
        result_value = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[current_board.result()]
        return [(s, p, result_value) for s, p, _ in game_data]

def parallel_self_play(
    model: torch.nn.Module,
    num_games: int,
    num_workers: int,
    config: ModelConfig,
    mcts_config: MCTSConfig,
    output_dir: str = "selfplay_data"
) -> None:
    
    os.makedirs(output_dir, exist_ok=True)

    result_queue = mp.Queue()
    processes = []
    
    games_per_worker = num_games // num_workers
    remaining_games = num_games % num_workers
    
    for worker_id in range(num_workers):
        worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
        if worker_games == 0:
            continue
        
        p = mp.Process(
            target=worker_process,
            args=(model, worker_id, worker_games, config, mcts_config, result_queue)
        )
        p.start()
        processes.append(p)
    
    completed_games = 0
    
    while completed_games < num_games:
        try:
            game_data = result_queue.get(timeout=30)
            if game_data is None: 
                continue
                
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"game_{completed_games}_{timestamp}.pt")
            
            torch.save({
                'states': torch.stack([d[0] for d in game_data]),
                'policies': torch.stack([d[1] for d in game_data]),
                'values': torch.tensor([d[2] for d in game_data], dtype=torch.float32)
            }, filename)
            
            completed_games += 1
            print(f"Saved game {completed_games}/{num_games} to {filename}")
        except Empty:
            if all(not p.is_alive() for p in processes):
                break
    
    for p in processes:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
    
    print(f"Completed {completed_games} games. Saved to {output_dir}")

def worker_process(
    model: torch.nn.Module,
    worker_id: int,
    num_games: int,
    config: ModelConfig,
    mcts_config: MCTSConfig,
    result_queue: mp.Queue
) -> None:
    try:
        worker = SelfPlayWorker(model, worker_id, config, mcts_config)
        for _ in range(num_games):
            game_data = worker.run_game()
            result_queue.put(game_data)
    except Exception as e:
        print(f"Worker {worker_id} failed with error: {str(e)}")
        result_queue.put(None)

if __name__ == "__main__":
    config = ModelConfig()
    mcts_config = MCTSConfig()
    model = NN(config)
    
    if torch.cuda.is_available():
        model.cuda()
    
    parallel_self_play(
        model=model,
        num_games=4,
        num_workers=4,
        config=config,
        mcts_config=mcts_config,
        output_dir="selfplay_data"
    )







