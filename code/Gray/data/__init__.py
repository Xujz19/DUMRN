from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

class TrainData():
    def __init__(self, args):
        dataset_name = args.dataset
        dataset = import_module('data.' + dataset_name.lower())
        dataset = getattr(dataset, dataset_name)(args)
        
        self.loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                )
            
    def get_loader(self):
        return self.loader