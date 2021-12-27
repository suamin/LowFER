import os
from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import logging
import math

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


def add_logging_handlers(params, dir_name="logs"):
    os.makedirs(dir_name, exist_ok=True)
    log_file = os.path.join(dir_name, params + ".log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', '%m/%d/%Y %H:%M:%S'))
    global logger
    logger.addHandler(fh)

    
class Experiment:
    
    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., k=30, output_dir=None):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.n_gpu = torch.cuda.device_count() if cuda else None
        self.batch_size = batch_size * self.n_gpu if self.n_gpu > 1 else batch_size
        self.device = torch.device("cuda") if cuda else None
        self.output_dir = output_dir
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "k": k}
    
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    
    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.label_smoothing:
            targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
        if self.cuda:
            targets = targets.to(self.device)
        return np.array(batch), targets
    
    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        
        logger.info("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.to(self.device)
                r_idx = r_idx.to(self.device)
                e2_idx = e2_idx.to(self.device)
            predictions = model.forward(e1_idx, r_idx)
            
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            
            sort_values, sort_idxs = torch.sort(predictions.cpu(), dim=1, descending=True)
            
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)
                
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        
        metrics = {
            'h10': np.mean(hits[9]), 
            'h3': np.mean(hits[2]), 
            'h1': np.mean(hits[0]),
            'mr': np.mean(ranks),
            'mrr': np.mean(1./np.array(ranks))
        }
        logger.info('Hits @10: {0}'.format(metrics['h10']))
        logger.info('Hits @3: {0}'.format(metrics['h3']))
        logger.info('Hits @1: {0}'.format(metrics['h1']))
        logger.info('Mean rank: {0}'.format(metrics['mr']))
        logger.info('Mean reciprocal rank: {0}'.format(metrics['mrr']))
        
        return metrics
    
    def train_and_eval(self):
        logger.info("Training the LowFER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        
        train_data_idxs = self.get_data_idxs(d.train_data)
        logger.info("Number of training data points: %d" % len(train_data_idxs))
        
        model = LowFER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            if self.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            model.to(self.device)
        if hasattr(model, 'module'):
           model.module.init()
        else:
           model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)
        
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        
        logger.info("Starting training...")
        logger.info("Params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.to(self.device)
                    r_idx = r_idx.to(self.device)
                predictions = model.forward(e1_idx, r_idx)           
                if hasattr(model, 'module'):
                    loss = model.module.loss(predictions, targets)
                    loss = loss.mean()
                else:
                    loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            logger.info("Epoch %d / time %0.5f / loss %0.9f" % (it, time.time()-start_train, np.mean(losses)))
            model.eval()
            if it % 10 == 0 and it != 0:
                with torch.no_grad():
                    logger.info("Validation:")
                    valid_metrics = self.evaluate(model, d.valid_data)
                    torch.save(model.state_dict(), self.output_dir + "/%d.pt" % it)

        with torch.no_grad():
            logger.info("Final Validation:")
            valid_metrics = self.evaluate(model, d.valid_data)
            logger.info("Final Test:")
            test_metrics = self.evaluate(model, d.test_data)
            torch.save(model.state_dict(), self.output_dir + "/final.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--k", type=int, default=30, nargs="?",
                    help="Latent dimension of MFB.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    args = parser.parse_args()
    params = "{}_lr_{}_dr_{}_e_{}_r_{}_k_{}_id_{}_hd1_{}_hd2_{}_ls_{}".format(
        args.dataset, args.lr, args.dr, args.edim, args.rdim, 
        args.k, args.input_dropout, args.hidden_dropout1,
        args.hidden_dropout2, args.label_smoothing
    )
    add_logging_handlers(params)
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    output_dir = "output/%s/%s" % (dataset, params)
    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, k=args.k, 
                            output_dir=output_dir)
    experiment.train_and_eval()
