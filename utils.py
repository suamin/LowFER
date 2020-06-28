# -*- coding: utf-8 -*-

import os
import logging
import torch
import collections
import pickle
import itertools

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DataReader:
    
    def __init__(self, dataset, cache_dir="tmp/cache", reverse=True):
        self.data_dir = os.path.join("data", dataset)
        self.name = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.reverse = reverse
        self.read_data()
        self.to_dataset()
    
    def read_file(self, fname):
        triples = list()
        with open(fname, encoding="utf-8", errors="ignore") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                h, r, t = line.split("\t")
                triples.append((h, r, t))
                if self.reverse:
                    triples.append((t, r + "_reverse", h))
        return triples
    
    def read_data_split(self, split):
        triples = self.read_file(os.path.join(self.data_dir, "%s.txt" % split))
        entities = {j for i, _, k in triples for j in (i, k)}
        relations = {j for _, j, _ in triples}
        return triples, entities, relations
    
    def read_data(self):
        T = list()
        E = set()
        R = set()
        raw_data = dict()
        for split in ("train", "valid", "test"):
            triples, entities, relations = self.read_data_split(split)
            logger.info("***  `{}` data stats  ****".format(split))
            logger.info("# of triples: {}".format(len(triples)))
            logger.info("# of entities: {}".format(len(entities)))
            logger.info("# of relations: {}".format(len(relations)))
            raw_data[split] = {"T": triples, "E": entities, "R": relations}
            T += triples
            E.update(entities)
            R.update(relations)
        
        E = sorted(list(E))
        R = sorted(list(R))
        T = sorted(list(T))
        e2idx = {entity:idx for idx, entity in enumerate(E)}
        r2idx = {relation:idx for idx, relation in enumerate(R)}
        data = dict()
        for split in ("train", "valid", "test"):
            # subject, object, rel, rel dir
            data[split] = [(e2idx[i], r2idx[j], e2idx[k]) for i, j, k in raw_data[split]["T"]]
        
        # set as class attributes
        self.raw_data = raw_data
        self.data = data
        self.e2idx = e2idx
        self.r2idx = r2idx
        self.T, self.E, self.R = T, E, R
        self.ne = len(e2idx)
        self.nr = len(r2idx)
    
    def to_dataset(self):
        dataset = dict()
        hr = collections.defaultdict(list)
        for split in ("train", "valid", "test"):
            logger.info("Creating `{}` features ...".format(split))
            sp_hr = collections.defaultdict(list)
            for h, r, t in self.data[split]:
                sp_hr[(h, r)].append(t)
                hr[(h, r)].append(t)
            if split == "train":
                dataset[split] = TrainDataset(sp_hr, self.ne)
        
        # NOTE: For evaluation, we pass the (es, r) -> eo mapping of 
        # "train U valid U test" as pointed in Bordes et al. (2013)
        dataset["valid"] = EvalDataset(self.data["valid"], hr, self.ne)
        dataset["test"] = EvalDataset(self.data["test"], hr, self.ne)
        self.dataset = dataset
    
    def get_dataloader(self, split="train", batch_size=128):
        dataset = self.dataset[split]
        if split == "train":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader


class TrainDataset(Dataset):
    
    def __init__(self, hr_dict, ne):
        self.data = list(hr_dict.items())
        self.ne = ne
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        (h, r), targets = self.data[idx]
        # 1-N scoring
        y = torch.zeros(self.ne)
        y[targets] = 1
        return torch.tensor([h, r]).long(), y.float()


class EvalDataset(Dataset):
    
    def __init__(self, triples, hr_dict, ne):
        self.data = triples
        self.hr_dict = hr_dict
        self.ne = ne
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        y = torch.zeros(self.ne)
        all_t = self.hr_dict[(h, r)]
        # To allow batched data and fast evaluation. At test time,
        # for a given triple (h, r, t), we set score of all t'
        # (1-N scoring) of (h, r) to 0 except the target t.
        all_t = [i for i in all_t if i != t]
        y[all_t] = 1
        return torch.tensor([h, r, t]).long(), torch.tensor(y).float()


def add_logging_handlers(logger, dir_name):
    log_file = os.path.join(dir_name, "run.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', '%m/%d/%Y %H:%M:%S'
    ))
    logger.addHandler(fh)
