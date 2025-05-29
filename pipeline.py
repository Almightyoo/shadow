from model import NeuralNetwork as NN
from AlphaMCTS import MCTS_self_play
from self_play_parallel import MCTSConfig, ModelConfig, parallel_self_play
from train import train_model
import torch
import torch.multiprocessing as mp


from dataclasses import dataclass



if __name__ == "__main__":
    for i in range(10):
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

        train_model(
            config,
            data_dir="selfplay_data",
            checkpoint_dir="checkpoints",
            resume_checkpoint="checkpoints/checkpoint_epoch_20_20250421_211246.pt",
            batch_size=64,
            num_epochs=20
        )

