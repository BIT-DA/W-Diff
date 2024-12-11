import copy
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F

from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import prepare_data, forward_pass, get_collate_functions, MetricLogger


class BaseTrainer:
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = 2

        self.CL_dataset = copy.deepcopy(dataset)
        self.CL_dataset.mode = 2

        self.num_classes = dataset.num_classes
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(args, self.train_dataset)

        # Training hyperparameters
        self.args = args
        self.epochs = args.trainer.epochs
        self.mini_batch_size = args.data.mini_batch_size
        self.num_workers = args.data.num_workers
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.split_time = args.data.split_time
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = 'accuracy'

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'lr={self.args.trainer.lr}-' \
                                f'mini_batch_size={self.args.data.mini_batch_size}-seed={self.args.random_seed}'
        base_trainer_str += f'-eval_fix'
        return base_trainer_str

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.num_batches = {} // {} = {}".format(self.train_dataset.__len__(),
                                                                                 self.args.data.mini_batch_size,
                                                                                 self.train_dataset.__len__() // self.args.data.mini_batch_size))
        stop_iters = self.args.trainer.epochs * (self.train_dataset.__len__() // self.args.data.mini_batch_size) - 1
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            #-----------------print log infromation------------
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iters - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            meters.update(loss=(loss).item())
            if step % self.args.log.print_freq == 0:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "timestamp: {timestamp}",
                            f"[iter: {step}/{stop_iters}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        eta=eta_string,
                        timestamp=self.train_dataset.current_time,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )

            if step % (stop_iters // 5) == 0:
                timestamp = self.train_dataset.current_time
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t validation {} is {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, acc * 100.0))
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))

    def train_online(self):
        self.train_dataset.mode = 0
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if timestamp == (self.split_time + 1):
                break
            else:
                self.train_dataset.update_current_timestamp(timestamp)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader)

                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("ID timestamp = {}: \t validation {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                logits = self.network(x)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()
        pred_all = np.array(pred_all)
        y_all = np.array(y_all)
        correct = (pred_all == y_all).sum().item()
        metric = correct / float(y_all.shape[0])
        self.network.train()
        return metric

    def evaluate_offline(self):
        self.logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        self.logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric = self.network_evaluation(test_id_dataloader)
                self.logger.info("Merged ID validation data: {}: \t{:.3f}\n".format(self.eval_metric, id_metric * 100.0))
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
        if len(metrics) >= 2:
            self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        self.train_online()
        self.evaluate_offline()

    def run(self):
        torch.cuda.empty_cache()
        self.run_eval_fix()
