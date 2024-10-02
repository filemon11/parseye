from supar import AttachJuxtaposeConstituencyParser
from supar.models.const.aj import AttachJuxtaposeConstituencyModel
from supar.modules import TransformerEmbedding

from supar.utils import Config, Dataset, Embedding
from supar.utils.transform import Batch
from supar.utils.config import Config
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.parallel import gather, is_dist, is_master, reduce
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.field import Field, RawField, SubwordField
from supar.utils.tokenizer import Tokenizer
from supar.utils.fn import download, get_rng_state, set_rng_state
from supar import MODEL, PARSER
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.optim import InverseSquareRootLR, LinearLR
from supar.utils.metric import Metric

from supar.utils.common import BOS, EOS, NUL, PAD, UNK

from data import EyeTracking, EyeTrackingSentence, FloatField
from metrics import EyeTrackingMetric

from torch.nn import Module
import torch.nn as nn
import torch
from torch.optim import SGD, Optimizer
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler

import dill
import sys
import contextlib
from contextlib import contextmanager
import pickle
import tempfile
import os
from datetime import datetime, timedelta
import shutil

from typing import Self, Literal, Iterable, Union, Any, Optional

logger = get_logger(__name__)

PATH_TO_PRETRAINED = "../results/models-con/ptb/abs-mgpt-lstm"
PATH_TO_INPUT = "testfile.txt"

Mode = Literal["aj", "vanilla", "both"]
Shift = Literal["current", "previous", "both"]

class DualTransformerModule(Module):

    COMPONENT = AttachJuxtaposeConstituencyModel

    def __init__(self, model_1 : AttachJuxtaposeConstituencyModel,
                 model_2 : AttachJuxtaposeConstituencyModel,
                 shift : Shift,
                 out_dim : int = 1):
        super().__init__()

        self.args_1 = model_1.args
        self.args_2 = model_2.args

        self.model_1 : AttachJuxtaposeConstituencyModel = model_1
        """aj"""
        
        self.model_2 : AttachJuxtaposeConstituencyModel = model_2
        """vanilla"""

        self.shift = shift
        factor : int = 2 if shift == "both" else 1

        self.aj_head : Module = nn.Linear(model_1.args.n_encoder_hidden * factor, out_dim)
        self.vanilla_head : Module = nn.Linear(model_2.args.n_encoder_hidden * factor, out_dim)
        self.both_head : Module = nn.Linear((model_1.args.n_encoder_hidden + model_2.args.n_encoder_hidden) * factor, out_dim)

        self.criterion = nn.MSELoss()

    def forward(
        self,
        words : torch.LongTensor,
        feats : list[torch.LongTensor],
        mode : Mode,
        use_vq : bool
    ) -> tuple[torch.Tensor]:
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor:
                Contextualized output hidden states of shape ``[batch_size, seq_len, n_model]`` of the input.
        """

        qloss_1 = 0
        qloss_2 = 0

        if mode == "aj" or mode == "both":
                
            x_1 = self.model_1.encode(words, feats)

            if use_vq:
                # pass through vector quantization
                x_1, qloss_1 = self.model_1.vq_forward(x_1)

        if mode == "vanilla" or mode == "both":
            x_2 = self.model_2.encode(words, feats)

            if use_vq:
                # pass through vector quantization
                x_2, qloss_2 = self.model_2.vq_forward(x_2)

        match mode:
            case "aj":
                x_in = x_1
            
            case "vanilla":
                x_in = x_2
            
            case "both":
                x_in = torch.cat([x_1, x_2], dim = 2)

            case _:
                raise Exception("unknown")
        
        x_in = self.create_x_in(x_in)

        match mode:
            case "aj":
                return self.aj_head(x_in), qloss_1
            
            case "vanilla":
                return self.vanilla_head(x_in), qloss_2
            
            case "both":
                return self.both_head(x_in), qloss_1 + qloss_2
        
        raise Exception(f"Invalid mode '{mode}'")
    
    def create_x_in(self, x):
        if self.shift == "previous":
            return x[:, 0:-2]
        elif self.shift == "current":
            return x[:, 1:-1]
        elif self.shift == "both":
            return torch.concat((x[:, 0:-2], x[:, 1:-1]), dim = 2)
        
    def loss(
        self,
        x: torch.Tensor,
        y: list[torch.Tensor]) -> torch.Tensor:

        l = torch.tensor(0, device = x.device, dtype = x.dtype)

        for s_x, s_y in zip(x, y):
            if len(s_x) == 0:
                continue
            s_y = s_y.to(x.device)
            l += self.criterion(s_x.view(-1), s_y.view(-1))

        return l

    def load_pretrained(self, embed = None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
            nn.init.zeros_(self.word_embed.weight)
        return self

class DualTransformer():

    MODEL = DualTransformerModule
    NAME = 'dual-transformer-aj-vanilla'

    def __init__(self, model : DualTransformerModule,
                 transform,
                 args : dict[str, str] = dict()):       # args should contain y_field, shift, use_vq
        
        self.model : DualTransformerModule = model

        self.transform = transform

        self.args = Config(**args)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def sync_grad(self):
        return self.step % self.args.update_steps == 0 or self.step % self.n_batches == 0

    @contextmanager
    def sync(self):
        context = getattr(contextlib, 'suppress' if sys.version < '3.7' else 'nullcontext')
        if is_dist() and not self.sync_grad:
            context = self.model.no_sync
        with context():
            yield
    
    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        mode : Mode,
        epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        beam_size: int = 1,
        delete: set = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
        equal: dict = {'ADVP': 'PRT'},
        verbose: bool = True,
        **kwargs
     ) -> None:
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            epochs (int):
                The number of training iterations.
            patience (int):
                The number of consecutive iterations after which the training process would be early stopped if no improvement.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            clip (float):
                Clips gradient of an iterable of parameters at specified value. Default: 5.0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
        """

        # TODO

        #print(self.args.data, type(self.args.data))

        args = self.args.update(locals())

        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        eval_batch_size = args.get('eval_batch_size', batch_size)
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
            eval_batch_size = eval_batch_size // dist.get_world_size()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        args.even = args.get('even', is_dist())

        train = Dataset(self.transform, args.train, **args).build(batch_size=batch_size,
                                                                  n_buckets=buckets,
                                                                  shuffle=True,
                                                                  distributed=is_dist(),
                                                                  even=args.even,
                                                                  n_workers=workers)
        dev = Dataset(self.transform, args.dev, **args).build(batch_size=eval_batch_size,
                                                              n_buckets=buckets,
                                                              shuffle=False,
                                                              distributed=is_dist(),
                                                              even=False,
                                                              n_workers=workers)
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test, **args).build(batch_size=eval_batch_size,
                                                                    n_buckets=buckets,
                                                                    shuffle=False,
                                                                    distributed=is_dist(),
                                                                    even=False,
                                                                    n_workers=workers)
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")
        loader, sampler = train.loader, train.loader.batch_sampler
        args.steps = len(loader) * epochs // args.update_steps
        args.save(f"{args.path}.yaml")

        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(module=self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get('find_unused_parameters', True),
                             static_graph=args.get('static_graph', False))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD, fp16_compress_hook)
        if args.wandb and is_master():
            import wandb
            # start a new wandb run to track this script
            wandb.init(config=args.primitive_config,
                       project=args.get('project', self.NAME),
                       name=args.get('name', args.path),
                       resume=self.args.checkpoint)
        self.step, self.epoch, self.best_e, self.patience = 1, 1, 1, patience
        # uneven batches are excluded
        self.n_batches = min(gather(len(loader))) if is_dist() else len(loader)
        self.best_metric, self.elapsed = Metric(), timedelta()
        if args.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                sampler.set_epoch(self.epoch)
            except AttributeError:
                logger.warning("No checkpoint found. Try re-launching the training procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            with self.join():
                # we should reset `step` as the number of batches in different processes is not necessarily equal
                self.step = 1
                for batch in bar:
                    for i, _ in enumerate(batch[0]):
                        if len(batch[0][i]) != len(batch[1][i]) +2:
                            print("Word length:", len(batch[0][i]))
                            print("Rest length:", len(batch[1][i]))
                        if len(batch[0][i]) != len(batch.fields[self.args.y_field][i]) +2:
                            print("Word length:", len(batch[0][i]))
                            print("y length:", len(batch.fields[self.args.y_field][i]))
                        #l = len(batch[0][i])
                        #assert all(len(f[i]) == l for f in batch), "Fields do not have the same length"

                    with self.sync():
                        with torch.autocast(self.device, enabled=args.amp):
                            loss = self.train_step(batch, mode)
                        self.backward(loss)
                        loss = loss.item()
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
                    # log metrics to wandb
                    if args.wandb and is_master():
                        wandb.log({'lr': self.scheduler.get_last_lr()[0], 'loss': loss})
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            with self.join(), torch.autocast(self.device, enabled=args.amp):
                metric = self.reduce(sum([self.eval_step(i, mode) for i in progress_bar(dev.loader)], Metric()))

                logger.info(f"{'dev:':5} {metric}")
                
                if args.wandb and is_master():
                    wandb.log({'dev': metric.values, 'epochs': epoch})
                if args.test:
                    test_metric = sum([self.eval_step(i, mode) for i in progress_bar(test.loader)], Metric())
                    logger.info(f"{'test:':5} {self.reduce(test_metric)}")
                    
                    if args.wandb and is_master():
                        wandb.log({'test': test_metric.values, 'epochs': epoch})

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                print("Break!")
                break
        if is_dist():
            dist.barrier()

        #best = self.load(**args)
        ## only allow the master device to save models
        #if is_master():
        #    best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")

        with open(f'{self.args.path}_result.txt', 'w') as file:
            file.write(f"{'dev:':5} {self.best_metric}")
        #if args.test:
        #    best.model.eval()
        #    with best.join():
        #        test_metric = sum([best.eval_step(i, mode) for i in progress_bar(test.loader)], Metric())
        #        logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")
        if args.wandb and is_master():
            wandb.finish()

        #self.evaluate(data=args.test, mode = mode, batch_size=batch_size)
        #self.predict(args.test, batch_size=batch_size, buckets=buckets, workers=workers)

        #with open(f'{self.args.folder}/status', 'w') as file:
        #    file.write('finished')

    def evaluate(
        self,
        data: Union[str, Iterable],
        mode : Mode,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        beam_size: int = 1,
        delete: set = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
        equal: dict = {'ADVP': 'PRT'},
        verbose: bool = True,
        **kwargs
    ):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
        data = Dataset(self.transform, **args)
        data.build(batch_size=batch_size,
                   n_buckets=buckets,
                   shuffle=False,
                   distributed=is_dist(),
                   even=False,
                   n_workers=workers)
        logger.info(f"\n{data}")

        logger.info("Evaluating the data")
        start = datetime.now()
        self.model.eval()
        with self.join():
            bar, metric = progress_bar(data.loader), Metric()
            for batch in bar:
                metric += self.eval_step(batch, mode)
                bar.set_postfix_str(metric)
            metric = self.reduce(metric)
        elapsed = datetime.now() - start
        logger.info(f"{metric}")
        logger.info(f"{elapsed}s elapsed, "
                    f"{sum(data.sizes)/elapsed.total_seconds():.2f} Tokens/s, "
                    f"{len(data)/elapsed.total_seconds():.2f} Sents/s")

        with open(f'{self.args.folder}/metrics.pickle', 'wb') as file:
            pickle.dump(obj=metric, file=file)

        return metric
    
    def predict(
        self,
        data: str | Iterable,
        mode: Mode,
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        args = self.args.update(locals())

        init_logger(logger, verbose = verbose)

        self.transform.eval()

        logger.info("Loading the data")
        if cache:
            bin = os.path.join(os.path.dirname(args.path), 'bin')
            
        if is_dist():
            batch_size = batch_size // dist.get_world_size()

        data = Dataset(self.transform, **args)

        data.build(batch_size=batch_size,
                   n_buckets=buckets,
                   shuffle=False,
                   distributed=is_dist(),
                   even=False,
                   n_workers=workers)
        
        logger.info(f"\n{data}")

        logger.info("Making predictions on the data")
        start = datetime.now()
        self.model.eval()

        with tempfile.TemporaryDirectory() as t:
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for batch in progress_bar(data.loader):
                batch = self.pred_step(batch, mode)
                if is_dist() or args.cache:
                    for s in batch.sentences:
                        with open(os.path.join(t, f"{s.index}"), 'w') as f:
                            f.write(str(s) + '\n')
            elapsed = datetime.now() - start

            if is_dist():
                dist.barrier()
            tdirs = gather(t) if is_dist() else (t,)
            if pred is not None and is_master():
                logger.info(f"Saving predicted results to {pred}")
                with open(pred, 'w') as f:
                    # merge all predictions into one single file
                    if is_dist() or args.cache:
                        sentences = (os.path.join(i, s) for i in tdirs for s in os.listdir(i))
                        for i in progress_bar(sorted(sentences, key=lambda x: int(os.path.basename(x)))):
                            with open(i) as s:
                                shutil.copyfileobj(s, f)
                    else:
                        for s in progress_bar(data):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if is_dist():
                dist.barrier()
        logger.info(f"{elapsed}s elapsed, "
                    f"{sum(data.sizes)/elapsed.total_seconds():.2f} Tokens/s, "
                    f"{len(data)/elapsed.total_seconds():.2f} Sents/s")

        if not cache:
            return data

    
    def train_step(self, batch: Batch, mode : Mode) -> torch.Tensor:
        words = batch[0]
        x, qloss = self.model(words, None, mode, self.args.use_vq)
        loss = self.model.loss(x, batch.fields[self.args.y_field]) + qloss
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch, mode : Mode) -> EyeTrackingMetric:
        words = batch[0]
        x, qloss = self.model(words, None, mode, self.args.use_vq)
        x_in = x
        y_in = batch.fields[self.args.y_field]
        loss = self.model.loss(x_in, y_in) + qloss

        return EyeTrackingMetric(loss, x_in, y_in)
    
    @torch.no_grad()
    def pred_step(self, batch: Batch, mode : Mode) -> Batch:
        words = batch[0]
        x, _ = self.model(words, None, mode, self.args.use_vq)

        x = x
        batch.preds = list(x)
        return batch

    def backward(self, loss: torch.Tensor, **kwargs):
        loss /= self.args.update_steps
        if hasattr(self, 'scaler'):
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def clip_grad_norm_(
        self,
        params: Union[Iterable[torch.Tensor], torch.Tensor],
        max_norm: float,
        norm_type: float = 2
    ) -> torch.Tensor:
        self.scaler.unscale_(self.optimizer)
        return nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    def clip_grad_value_(
        self,
        params: Union[Iterable[torch.Tensor], torch.Tensor],
        clip_value: float
    ) -> None:
        self.scaler.unscale_(self.optimizer)
        return nn.utils.clip_grad_value_(params, clip_value)
    
    def reduce(self, obj: Any) -> Any:
        if not is_dist():
            return obj
        return reduce(obj)
    
    @contextmanager
    def join(self):
        context = getattr(contextlib, 'suppress' if sys.version < '3.7' else 'nullcontext')
        if not is_dist():
            with context():
                yield
        elif self.model.training:
            with self.model.join():
                yield
        else:
            try:
                dist_model = self.model
                # https://github.com/pytorch/pytorch/issues/54059
                if hasattr(self.model, 'module'):
                    self.model = self.model.module
                yield
            finally:
                self.model = dist_model
        
    def init_optimizer(self) -> Optimizer:
        return SGD(self.model.parameters(), 
                   lr = self.args.lr,
                   momentum = self.args.momentum,
                   dampening = self.args.dampening,
                   weight_decay = self.args.weight_decay)
    
    def init_scheduler(self) -> _LRScheduler:
        scheduler = LinearLR(optimizer=self.optimizer,
                                 warmup_steps=self.args.get('warmup_steps', int(self.args.steps*self.args.get('warmup', 0))),
                                 steps=self.args.steps)
        return scheduler
    
    @classmethod
    def create_aj_vanilla_from_pretrained(cls, path_to_pretrained : str,
                                          new_path : str,
                                          y_field : str,
                                          shift : bool = False,
                                          finetune : bool = False,
                                          use_vq : bool = False,
                                          out_dim : int = 1,
                                          **kwargs) -> Self:

        ### Load pre-trained parsing model ###

        aj_parser : AttachJuxtaposeConstituencyParser
        aj_parser = AttachJuxtaposeConstituencyParser.load(os.path.join(path_to_pretrained, "parser.pt"), **kwargs)

        aj_parser.args.path = new_path
        aj_parser.model.args.path = new_path

        aj_parser.args.finetune = finetune
        aj_parser.model.args.finetune = finetune
        aj_parser.model.encoder.finetune = finetune
        aj_parser.model.encoder.model = aj_parser.model.encoder.model.requires_grad_(finetune)

        ### Load vanilla mGPT model ###

        vanilla_parser : AttachJuxtaposeConstituencyParser
        vanilla_parser = AttachJuxtaposeConstituencyParser.load(os.path.join(path_to_pretrained, "parser.pt"), **kwargs)

        vanilla_parser.args.path = new_path
        vanilla_parser.model.args.path = new_path

        vanilla_parser.args.finetune = finetune
        vanilla_parser.model.args.finetune = finetune

        vanilla_parser.model.encoder = TransformerEmbedding(name = aj_parser.args.bert,
                                                            n_layers = aj_parser.args.n_bert_layers,
                                                            n_out = aj_parser.args.n_encoder_hidden,
                                                            pooling = aj_parser.args.bert_pooling,
                                                            pad_index = aj_parser.args.pad_index,
                                                            mix_dropout = aj_parser.args.mix_dropout,
                                                            finetune = aj_parser.args.finetune)
        
        # Replace transform for EyeTracking transform

        transform = cls.build_transform(**aj_parser.args)
        transform.WORD = aj_parser.transform.WORD

        aj_parser.transform = transform
        vanilla_parser.transform = transform

        kwargs["y_field"] = y_field
        kwargs["shift"] = shift
        kwargs["use_vq"] = use_vq
        kwargs["path"] = new_path
        #args = {"y_field" : y_field,
        #        "shift" : shift,
        #        "use_vq" : use_vq}

        # Create DualTransformer and move it to one device
        
        dual_transformer = cls(DualTransformerModule(aj_parser.model, vanilla_parser.model, shift, out_dim), transform, kwargs)

        dual_transformer.model.to(aj_parser.device)

        return dual_transformer
    
    @classmethod
    def load(
        cls,
        path: str,
        reload: bool = False,
        src: str = 'github',
        **kwargs
    ) -> Self:
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            checkpoint (bool):
                If ``True``, loads all checkpoint states to restore the training process. Default: ``False``.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        args = Config(**locals())
        if not os.path.exists(path):
            path = download(MODEL[src].get(path, path), reload=reload)

        state = torch.load(path, map_location='cpu')
        
        args_1 = state['args_1'].update(args)
        args_2 = state['args_2'].update(args)
        args = state['args'].update(args)

        component_1 = cls.MODEL.COMPONENT(**args_1)
        component_2 = cls.MODEL.COMPONENT(**args_2)
        model = cls.MODEL(component_1, component_2, args.shift)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)

        transform = state['transform']
        parser = cls(model, transform, args)
        parser.model.to(parser.device)
        return parser

    def save(self, path: str) -> None:
        model = self.model
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        print(state_dict.keys())
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args' : self.args,
                 'args_1': model.args_1,
                 'args_2': model.args_2,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    @classmethod
    def build_transform(cls, **kwargs):
        args = Config(**locals())

        TAG, CHAR = None, None

        t = TransformerTokenizer(args.bert)

        WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, eos=t.eos, fix_len=args.fix_len, tokenize=t, delay=0)
        WORD.vocab = t.vocab

        NFIX = FloatField('nFix', pad=0, unk=0, bos=0, eos=0, use_vocab=False)
        FFD = FloatField('FFD', pad=0, unk=0, bos=0, eos=0, use_vocab=False)
        GPT = FloatField('GPT', pad=0, unk=0, bos=0, eos=0, use_vocab=False)
        TRT = FloatField('TRT', pad=0, unk=0, bos=0, eos=0, use_vocab=False)
        FIXPROP = FloatField('FixProp', pad=0, unk=0, bos=0, eos=0, use_vocab=False)
        
        return EyeTracking(WORD=(WORD, CHAR), NFIX=NFIX, FFD=FFD, GPT=GPT, TRT=TRT, FIXPROP=FIXPROP)


    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.transform.WORD[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        
        transform = cls.build_transform(**args)

        train = Dataset(transform, args.train, **args)
        
        transform.WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
        
        args.update({
            'n_words': len(transform.WORD.vocab),
            'pad_index': transform.WORD.pad_index,
            'unk_index': transform.WORD.unk_index,
            'bos_index': transform.WORD.bos_index,
            'eos_index': transform.WORD.eos_index,
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        component_1 = cls.MODEL.COMPONENT(**args).load_pretrained(transform.WORD.embed if hasattr(transform.WORD, 'embed') else None)
        component_2 = cls.MODEL.COMPONENT(**args).load_pretrained(transform.WORD.embed if hasattr(transform.WORD, 'embed') else None)

        logger.info(f"{component_1}\n")
        logger.info(f"{component_2}\n")

        model = cls.MODEL(component_1, component_2, args.shift, args.out_dim)
        
        parser = cls(model, transform, args)
        parser.model.to(parser.device)
        return parser
    
    def save_checkpoint(self, path: str) -> None:
        model = self.model

        checkpoint_state_dict = {k: getattr(self, k) for k in ['epoch', 'best_e', 'patience', 'best_metric', 'elapsed']}
        checkpoint_state_dict.update({'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'scaler_state_dict': self.scaler.state_dict(),
                                      'rng_state': get_rng_state()})
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args_1': model.args_1,
                 'args_2': model.args_2,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'checkpoint_state_dict': checkpoint_state_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)
