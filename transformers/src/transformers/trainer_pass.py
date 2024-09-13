import inspect
import json
import math
import os
import random
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .file_utils import is_datasets_available, is_torch_tpu_available
from .integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    distributed_broadcast_scalars,
    distributed_concat,
    set_seed,
)
from .training_args import TrainingArguments
from .utils import logging
from .trainer import (
    torch_distributed_zero_first,
    SequentialDistributedSampler,
    get_tpu_sampler,
    Trainer
)

from math import log, exp, floor


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
    return max(1e-8, float(prod_layer.prod()))


def convert_gate_to_mask(gates, num_of_heads=None):
    if num_of_heads is not None:
        head_mask = torch.zeros_like(gates)
        current_heads_to_keep = gates.view(-1).sort(descending = True)[1]
        current_heads_to_keep = current_heads_to_keep[:num_of_heads]
        head_mask = head_mask.view(-1)
        head_mask[current_heads_to_keep] = 1.0
        head_mask = head_mask.view_as(gates)
    else:
        head_mask = (gates > 0.5).float()
    return head_mask


_use_native_amp = False
_use_apex = False

temperature = 0.33
gamma = -0.1
zeta = 1.1
hp_a = temperature * log(-gamma / zeta)

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_wandb_available():
    import wandb

if is_comet_available():
    import comet_ml

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)

class PASSTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs,
    ):
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics,
            tb_writer, optimizers, **kwargs
        )

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        print('local_rank is', self.args.local_rank)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                print('Continuing training from checkpoint', model_path)
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        clip_lb = -5
        clip_ub = 5
        conc_coeff_ub = 3000
        conc_coeff_init = 0
        coeff_sparsifier_init = 1e-5

        def _schedule_sparsifier(accumulate_step):
            return coeff_sparsifier_init * 1000 ** (accumulate_step / 1000)

        def _schedule_concentrator(current_step):
            if 1000 < current_step < 5000 and current_step % 1000 == 0:
                return True
            else:
                return False

        def _schedule_clip_reopen_reset(current_step):
            if 1000 < current_step < 5000 and current_step % 1000 == 0:
                return True
            else:
                return False

        total_head_num = 144
        target_sparsity = model.get_target_sparsity()
        target_gates = int(target_sparsity * total_head_num)
        coeff_sparsifier = coeff_sparsifier_init
        conc_coeff = conc_coeff_init
        sparsifier_counter = 0
        model.apply_reg_coeff(reg_coeff=coeff_sparsifier, conc_coeff=conc_coeff)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue
                tr_loss += self.training_step(model, inputs)
                self.total_flos += self.floating_point_ops(inputs)

                loga = model.get_loga_values()
                close_gates = int(torch.sum((loga <= clip_lb).float()))
                open_gates = int(torch.sum((loga >= clip_ub).float()))
                close_layers = int(sum([i.prod() for i in (loga <= clip_lb)]))

                if close_gates != target_gates or open_gates != (total_head_num - target_gates):
                    coeff_sparsifier = _schedule_sparsifier(accumulate_step=sparsifier_counter)
                    sparsifier_counter += 1

                min_conc_coeff = conc_coeff_ub

                # clip log_a & reopen w.r.t. probs
                if _schedule_clip_reopen_reset(current_step=self.global_step):
                    # normalize confidence
                    confidence = model.get_confidence()
                    norm_confidence = confidence / confidence.max()
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n.split('.')[-1] == "log_a" and p.requires_grad:
                                p_shape = p.shape
                                clipped = torch.clamp(p, clip_lb, clip_ub).flatten()

                                # retrieve normalized confidence as reopen prob base
                                name_splits = n.split('.')
                                offset = int(name_splits[3])
                                reopen_base = norm_confidence[offset]
                                for i in range(len(clipped)):
                                    reopen_prob = reopen_base[i]
                                    if clipped[i] <= clip_lb and random.uniform(0, 1) < reopen_prob:
                                        clipped[i] = clip_ub
                                p.copy_(clipped.view(p_shape))

                                for i in range(len(clipped)):
                                    if clipped[i] > clip_lb:
                                        conc_coeff_lb = 2 * func_conc_sig_ratio(clipped[i]) / func_conc_q0_prod(
                                            clipped, i)
                                        min_conc_coeff = min(min_conc_coeff, conc_coeff_lb)
                    model.reset_confidence()

                    loga = model.get_loga_values()
                    close_gates = int(torch.sum((loga <= clip_lb).float()))

                if close_gates >= target_gates:
                    conc_coeff = 0
                elif _schedule_concentrator(current_step=self.global_step):
                    conc_coeff = min_conc_coeff

                model.apply_reg_coeff(reg_coeff=coeff_sparsifier, conc_coeff=conc_coeff)

                if self.global_step % 1000 == 0:
                    print('reg_coeff is', coeff_sparsifier)
                    print('conc_coeff is', conc_coeff)

                    print('num of close gates is %d' % close_gates)
                    print('num of close layers is %d' % close_layers)

                    print('pruning masks')
                    print(convert_gate_to_mask(model.get_gate_values()))
                    print("")

                # perform step() and back-prop
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 1:
                        metrics = self.evaluate()
                        self._report_to_hp_search(trial, epoch, metrics)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        if self.hp_search_backend is not None and trial is not None:
                            run_id = (
                                trial.number
                                if self.hp_search_backend == HPSearchBackend.OPTUNA
                                else tune.get_trial_id()
                            )
                            checkpoint_folder += f"-run-{run_id}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_process_zero():
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            epoch_pbar.close()
            train_pbar.update(1)
            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)
