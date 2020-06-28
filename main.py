# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import logging
import utils
import time
import argparse
import datetime
import os
import model as models

from torch import optim
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def set_seed(args):
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, data_reader, args):
    batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    dataloader = data_reader.get_dataloader("train", batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.decay_rate > 0.:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.decay_rate)
    else:
        scheduler = None
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(data_reader.dataset["train"]))
    logger.info("  Num epochs = %d", args.epochs)
    
    model.zero_grad()
    tr_loss = 0.
    eval_h1 = 0.
    
    for epoch in trange(0, int(args.epochs), desc="Epoch"):
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        epoch_loss = list()
        t0 = time.time()
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"h_idx": batch[0][:, 0], "r_idx": batch[0][:, 1], "labels": batch[1]}
            outputs = model(**inputs)
            probas, loss = outputs
            
            if args.n_gpu > 1:
                loss = loss.mean()
            
            loss.backward()
            epoch_loss.append(loss.item())
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        logger.info("Epoch %d -- time %0.5f -- epoch_loss %0.9f" % (epoch+1, time.time()-t0, np.mean(epoch_loss)))
        if (epoch + 1) % args.eval_epoch == 0:
            with torch.no_grad():
                metrics = evaluate(model, data_reader, args, prefix="valid")
                if metrics["H@1"] >= eval_h1:
                    if hasattr(model, "module"):
                        model_to_save = model.module
                    else:
                        model_to_save = model
                    model_to_save.save(args.output_dir, "best", metrics)
                    eval_h1 = metrics["H@1"]
    
    # Inference with best and final model
    logger.info("Testing with final model ...")
    metrics = evaluate(model, data_reader, args, prefix="test")
    if hasattr(model, "module"):
        model_to_save = model.module
    else:
        model_to_save = model
    model_to_save.save(args.output_dir, "final", metrics)
    
    logger.info("Testing with best model ...")
    best_model = models.BaseModel.load(models.LowFER, args.output_dir, "best")
    if args.n_gpu > 1:
        best_model = torch.nn.DataParallel(best_model)
    best_model.to(args.device)
    metrics = evaluate(best_model, data_reader, args, prefix="test")


def evaluate(model, data_reader, args, prefix="valid"):
    batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataloader = data_reader.get_dataloader(prefix, batch_size)
    
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    logger.info("***** Running evaluation `{}` *****".format(prefix))
    logger.info("  Num examples = %d", len(data_reader.dataset[prefix]))
    logger.info("  Batch size = %d", batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {"h_idx": batch[0][:, 0], "r_idx": batch[0][:, 1], "labels": batch[1]}
            outputs = model(**inputs)
            scores, loss = outputs
            # Filtered evaluation
            scores[batch[1] != 0] = 0.
            eval_loss += loss.mean().item()
        
        nb_eval_steps += 1
        triples = batch[0].detach().cpu()

        # Sorted predictions
        sorted_idxs = torch.argsort(scores, dim=1, descending=True)
        sorted_idxs = sorted_idxs.detach().cpu().numpy()
        
        for i in range(triples.size(0)):
            t = triples[i][2].item()
            rank = np.where(sorted_idxs[i] == t)[0][0]
            ranks.append(rank+1)
            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    eval_loss = eval_loss / nb_eval_steps
    metrics = {
        "H@1": np.mean(hits[0]),
        "H@3": np.mean(hits[2]),
        "H@10": np.mean(hits[9]), 
        "MRR": np.mean(1./np.array(ranks)), 
        "MR": np.mean(ranks)
    }
    for k, v in metrics.items():
        logger.info("{} = {}".format(k, v))
    
    return metrics


def run(args):
    run_dir = args.dataset + "-" + "_".join(str(datetime.datetime.now()).split())
    run_dir = run_dir.replace(":", "-")
    if args.run_name:
        run_dir = run_dir + "-" + args.run_name
    args.output_dir = os.path.join(args.output_dir, run_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    utils.add_logging_handlers(logger, args.output_dir)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    data_reader = utils.DataReader(args.dataset, reverse=args.reverse)
    model = models.LowFER(
        data_reader.ne, data_reader.nr, args.de, args.dr, 
        args.k, args.d_in, args.d_h1, args.d_h2, args.label_smoothing
    )
    logger.info("Starting training ...")
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train(model, data_reader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FB15k-237", type=str, 
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory.")
    parser.add_argument("--reverse", action="store_true",
                        help="Whether to add reverse relations.")
    parser.add_argument("--run_name", default="", type=str,
                        help="Name for the run.")
    
    # Training config parameters
    parser.add_argument("--epochs", default=500, type=int, 
                        help="Number of epochs.")
    parser.add_argument("--lr", default=0.0005, type=float,
                        help="Learning rate.")
    parser.add_argument("--decay_rate", default=0., type=float, 
                        help="Decay rate for exponential lr scheduler.")
    parser.add_argument("--label_smoothing", default=0., type=float,
                        help="Amount of label smoothing.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available.")
    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int, 
                        help="Train batch size per gpu/cpu.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, 
                        help="Eval batch size per gpu/cpu.")
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help="Run evaluation every this many epochs.")
    parser.add_argument('--seed', type=int, default=20,
                        help="Random seed for initialization")
    
    # Model parameters
    parser.add_argument("--de", default=200, type=int,
                        help="Entity embedding dimensionality.")
    parser.add_argument("--dr", default=200, type=int,
                        help="Relation embedding dimensionality.")
    parser.add_argument("--k", default=10, type=int, 
                        help="Latent dimension of LowFER.")
    parser.add_argument("--d_in", default=0.2, type=float, 
                        help="Input entity embedding dropout.")
    parser.add_argument("--d_h1", default=0.2, type=float, 
                        help="Dropout after the MFB layer.")
    parser.add_argument("--d_h2", default=0.3, type=float, 
                        help="Dropout before the final output.")
    
    args = parser.parse_args()
    run(args)
