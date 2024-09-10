
from gflownet.config import Config, init_empty
from gflownet.tasks.qm9 import QM9GapTrainer


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.num_workers = 32
    config.num_training_steps = 10000
    config.num_validation_gen_steps = 1
    config.validate_every = 1000000000
    config.print_every = 100
    config.checkpoint_every = None
    config.log_dir = "./logs/test_aut_qm9"
    config.opt.lr_decay = 10000
    config.task.qm9.h5_path = "qm9.h5"
    config.task.qm9.model_path = "mxmnet_gap_model.pt"
    config.overwrite_existing_exp = True
    config.continue_training = False
    config.task.qm9.correct_automorphism = True

    trial = QM9GapTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()

