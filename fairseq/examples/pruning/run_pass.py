#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable
from math import log, exp

import numpy as np
import torch
import random

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig
from fairseq.trainer import Trainer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Voita et al.")

temperature = 0.33
gamma = -0.1
zeta = 1.1
hp_a = temperature * log(-gamma / zeta)

def sig(x):
    return 1 / (1 + exp(-x))


def sig_tensor(x):
    return 1 / (1 + torch.exp(-x))


def func_q0(x):
    return sig(hp_a - x)


def func_q0_tensor(x):
    return sig_tensor(hp_a - x)


def func_q1(x):
    return sig(hp_a + x)


def func_conc_sig_ratio(x):
    sig_ratio = sig(hp_a + x) * sig(-hp_a - x) / (sig(hp_a - x) * sig(-hp_a + x))
    return float(max(sig_ratio, 1))


def func_conc_q0_prod(x, i):
    prod_layer = func_q0_tensor(x)
    prod_layer[i] = 1.
    return float(prod_layer.prod())


def main(cfg: DictConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    seed1 = 34
    np.random.seed(seed1)
    seed2 = 34
    utils.set_torch_seed(seed2)
    print('seed1 is %d, seed2 is %d' % (seed1, seed2))

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    # logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    new_reg_coeff = 0.1
    reg_ratio = 2
    decay_factor = 0.9          # decay
    clip_lb = -5
    clip_ub = 5
    close_threshold = -5

    is_conc_on = False
    conc_coeff_ub = 500
    afterep = 20
    num_conc_ep = 3

    # original cfg.pruning.sparsity_rate is indeed density
    cfg.pruning.sparsity_rate = 1 - cfg.pruning.sparsity_rate / model.total_head_num
    target_gates = round(cfg.pruning.sparsity_rate * model.total_head_num)
    close_gates = 0         # number of closed gates

    while epoch_itr.next_epoch_idx <= max_epoch:
        print('BEFORE TRAINING')
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        if close_gates != target_gates:
            new_reg_coeff = new_reg_coeff * reg_ratio

        if epoch_itr.next_epoch_idx == afterep:
            is_conc_on = True

        conc_coeff = 0

        decay_power = epoch_itr.next_epoch_idx - 2
        head_confidence = model.get_confidence()
        max_confidence = torch.max(head_confidence)
        if epoch_itr.next_epoch_idx > 1 and max_confidence > 0:
            if is_conc_on:
                conc_coeff = conc_coeff_ub
            # normalize confidence
            norm_confidence = torch.div(head_confidence, max_confidence)
            print('normalized confidence is')
            print(norm_confidence)

            print('decay factor is', decay_factor)
            print('decay power is', decay_power)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n.split('.')[-1] == "log_a" and p.requires_grad:
                        p_shape = p.shape
                        clipped = torch.clamp(p, clip_lb, clip_ub).flatten()

                        # retrieve normalized confidence as reopen prob base
                        name_splits = n.split('.')
                        coder = name_splits[0] + '_' + name_splits[3]
                        layer = int(name_splits[2])
                        offset = -1
                        if coder == 'encoder_self_attn':
                            offset = 0 + layer
                        elif coder == 'decoder_encoder_attn':
                            offset = model.head_size[0] / 3 + layer
                        elif coder == 'decoder_self_attn':
                            offset = model.head_size[0] / 3 * 2 + layer
                        else:
                            print('unknown coder type!', n)
                        offset = int(offset)
                        # normalized confidence scores for the offset-th layer
                        reopen_base = norm_confidence[offset]

                        for i in range(len(clipped)):
                            reopen_prob = reopen_base[i] * decay_factor ** decay_power
                            if clipped[i] == clip_lb and random.uniform(0, 1) < reopen_prob:
                                clipped[i] = clip_ub
                        p.copy_(clipped.view(p_shape))

                        if is_conc_on:
                            for i in range(len(clipped)):
                                if clipped[i] > clip_lb:
                                    conc_coeff_lb = 2 * func_conc_sig_ratio(clipped[i]) / func_conc_q0_prod(clipped, i)
                                    conc_coeff = min(conc_coeff, conc_coeff_lb)
        model.reset_confidence()

        # if (close_layers >= max_close_layer or close_layers <= pre_close_layers) and is_conc_on:
        if num_conc_ep <= 0 and is_conc_on:
            is_conc_on = False

        if is_conc_on:
            num_conc_ep -= 1

        conc_coeff = 0
        model.apply_gates(new_reg_coeff, cfg.pruning.sparsity_rate, conc_coeff=conc_coeff)
        print('reg_coeff is, ', model.get_reg_coeff())
        print('conc_coeff is ', conc_coeff)

        print('loga is')
        print(model.get_loga())
        print('gates are')
        print(model.get_gate_values())

        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)

        print('AFTER TRAINING')
        print('loga is')
        loga = model.get_loga()
        print(loga)
        print('gates are')
        print(model.get_gate_values())

        close_gates = int(torch.sum((loga <= close_threshold).float()))
        print('num of close gates is %d' % close_gates)
        close_layers = int(sum([i.prod() for i in (loga <= close_threshold)]))
        print('num of close layers is %d' % close_layers)
        exp_density = sum([func_q1(i) for i in loga.flatten().tolist()]) / model.total_head_num
        exp_saprsity = sum([func_q0(i) for i in loga.flatten().tolist()]) / model.total_head_num
        print('exp_density is %.3f, exp_sparsity is %.3f' % (exp_density, exp_saprsity))

        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        print('valid loss is', valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    # logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    prev_num = getattr(should_stop_early, "num_runs", None)
    print('num_runs and patience is ', prev_num, cfg.checkpoint.patience)
    if prev_best is None or is_better(valid_loss, prev_best) or valid_loss < 0:
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging if distributed_utils.is_master(cfg.distributed_training) else False
        ),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                # stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                # progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or num_updates >= max_update
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or num_updates >= max_update
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(cfg, valid_losses[0])
        or num_updates >= max_update
        or (
            cfg.optimization.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60)
            > cfg.optimization.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        # logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        # logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    options.add_pruning_args(parser)
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    torch.set_printoptions(precision=2)
    cli_main()
