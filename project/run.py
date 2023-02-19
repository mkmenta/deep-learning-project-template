import os
import sys
from argparse import ArgumentParser
from getpass import getuser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from project.data import MNISTDataModule
from project.model import Backbone
from project.solver import Solver


def args():
    parser = ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--only-test', action='store_true')
    parser.add_argument('--half-precision', action='store_true')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--max-epochs', default=20, type=int)
    parser.add_argument('--exp-name', default="debug", type=str)
    parser.add_argument('--exp-dir', default="~/Experiments", type=str)
    parser.add_argument('--fast-dev-run', action='store_true')

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = Solver.add_model_specific_args(parser)
    args = parser.parse_args()
    args.exp_dir = os.path.expanduser(args.exp_dir)
    return args


def main(args):
    pl.seed_everything(1234)

    # Data
    datamodule = MNISTDataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = Backbone(hidden_dim=args.hidden_dim)

    # Solver (PytorchLighting Module)
    solver = Solver(model, hparams=args)

    # Callbacks
    callbacks = []

    # Logger
    neptune_logger = None
    args.username = getuser()
    args.run_command = ' '.join(sys.argv)
    if args.exp_name != 'debug':
        neptune_logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project=os.environ["NEPTUNE_PROJECT_NAME"],
            name=args.exp_name,
            log_model_checkpoints=False
        )
        neptune_logger.log_hyperparams(args.__dict__)
        args.exp_dir = os.path.join(args.exp_dir, args.exp_name, neptune_logger.version)

        # Checkpoints
        callbacks.extend([
            # saves top-K checkpoints based on "val_loss" metric
            ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                dirpath=args.exp_dir,
                filename="checkpoint-best-{epoch:02d}",
            ),
            # saves last-K checkpoints based on "global_step" metric
            # make sure you log it inside your LightningModule
            ModelCheckpoint(
                every_n_epochs=1,
                dirpath=args.exp_dir,
                filename="checkpoint-last-{epoch:02d}",
            )
        ])

    # Run
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        precision=16 if args.half_precision else 32,
        logger=neptune_logger,
        default_root_dir=args.exp_dir,
        callbacks=callbacks,
        log_every_n_steps=100,
        fast_dev_run=5 if args.fast_dev_run else False
    )
    if not args.only_test:
        trainer.fit(solver, datamodule=datamodule)
    result = trainer.test(solver, datamodule=datamodule)
    print(result)


if __name__ == '__main__':
    main(args())
