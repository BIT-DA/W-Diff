import math
import time
import numpy as np
import datetime
import gc
import torch
import copy
import torch.utils.data
from torch.nn import functional as F

from methods.base_trainer import BaseTrainer
from methods.utils import prepare_data, MetricLogger
from methods.dataloaders import FastDataLoader, InfiniteDataLoader




class WDiff(BaseTrainer):
    def __init__(self, args, logger, dataset, network, diffusion_model, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.diffusion_model = diffusion_model
        self.DM_optimizer = self.diffusion_model.configure_optimizers()
        self.logger = logger
        self.logger.info(f"DiffusionWrapper has {self.diffusion_model.total_params * 1.e-6:.2f} M params.")
        self.eps = 1e-6

    def __str__(self):
        str_all = f'WDiff-Lambda={self.args.trainer.Lambda}-{self.base_trainer_str}'
        return str_all

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        meters = MetricLogger(delimiter="  ")
        self.logger.info("self.train_dataset.num_batches = {} // {} = {}".format(self.train_dataset.__len__(), self.args.data.mini_batch_size, self.train_dataset.__len__() // self.args.data.mini_batch_size))
        timestamp = self.train_dataset.current_time
        stop_iters = self.args.trainer.epochs * (self.train_dataset.__len__() // self.args.data.mini_batch_size)
        warmup_iters = self.args.trainer.warm_up * stop_iters

        print("timestamp={}, init_timestamp={}".format(timestamp, self.args.data.init_timestamp))
        if timestamp - self.args.data.init_timestamp >= 1:
            previous_W = self.network.reference_point_queue.get_all_items()   # the oldest item at the start (left) of the quque, the newest item at the end (right) of the quque.
            previous_W = torch.stack(previous_W, dim=0)
            num_previous_items = self.network.reference_point_queue.len()
        else:
            previous_W = None

        best_val_acc = 0.0
        best_W = None
        end = time.time()
        global_feature_mean = None
        global_count = 0
        self.network.anchor_point_and_prototype_queue.init()  # clear the DM training sample queue at the beginning of training on the t-th domain
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            f, logits = self.network.foward(x)
            total_loss = 0

            if step > warmup_iters and timestamp - self.args.data.init_timestamp >= 1:
                with torch.no_grad():
                    softmax_logits = torch.softmax(logits, dim=1)
                    CxB_softmax_logits = softmax_logits.transpose(0, 1)
                    if global_feature_mean is None:
                        global_feature_mean = CxB_softmax_logits @ f / f.shape[0]
                    else:
                        global_feature_mean = global_feature_mean * (global_count / (global_count + f.shape[0])) + CxB_softmax_logits @ f / (global_count + f.shape[0])
                    global_count += f.shape[0]
                    item_to_save = torch.cat((self.network.classifier.weight.detach(), global_feature_mean.detach()), dim=1)  # item_to_save.shape=[C, 2*D]
                    self.network.anchor_point_and_prototype_queue.put_item(item_to_save.cpu())

            # --------cross-entropy loss--------
            loss_ce = self.criterion(logits, y)
            total_loss += loss_ce
            meters.update(loss_ce=(loss_ce).item())

            # ----------consistency loss--------
            if timestamp - self.args.data.init_timestamp > 0:
                LxBxD_f = f.unsqueeze(0).expand(num_previous_items, f.shape[0], f.shape[1])
                other_logits = torch.bmm(LxBxD_f, previous_W.transpose(1, 2).cuda())

                current_logits = f @ self.network.classifier.weight.detach().transpose(0, 1)
                all_logits = torch.cat((other_logits, current_logits.unsqueeze(0)), dim=0)
                all_softmax_outs = torch.softmax(all_logits, dim=2)
                mean_softmax_outs = torch.mean(all_softmax_outs, dim=0, keepdim=True)

                loss_con = torch.nn.functional.kl_div(torch.log(all_softmax_outs+1e-5), target=mean_softmax_outs.detach(), reduction='none')
                loss_con = torch.sum(loss_con) / (all_softmax_outs.shape[0] * all_softmax_outs.shape[1] * all_softmax_outs.shape[2])

                total_loss += self.args.trainer.Lambda * loss_con
                meters.update(loss_con=(loss_con).item())

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            meters.update(total_loss=(total_loss).item())


            # train diffusion model
            if self.network.anchor_point_and_prototype_queue.len() == self.args.trainer.M:
                DM_train_items = self.network.anchor_point_and_prototype_queue.get_all_items()
                DM_train_items = torch.stack(DM_train_items, dim=0)
                DM_W_batch = DM_train_items[:, :, :self.network.feature_dim]
                DM_prototype_batch = DM_train_items[:, :, self.network.feature_dim:]

                start_index = max(0, timestamp - self.args.data.init_timestamp - self.args.trainer.L)
                max_inner_iter = math.ceil(self.args.trainer.inner_iters_DM / (timestamp - self.args.data.init_timestamp - start_index))
                device = torch.device("cpu")
                self.network.to(device)
                torch.cuda.empty_cache()
                gc.collect()
                for inner_iter in range(max_inner_iter):
                    for t in range(start_index + 1, timestamp - self.args.data.init_timestamp + 1):
                        with torch.no_grad():
                            previous_W_at_t = previous_W[t - start_index - 1, :, :]
                            previous_W_at_t = previous_W_at_t.unsqueeze(0)   # [1, C, D]
                            delta_W_at_t = DM_W_batch - previous_W_at_t

                            delta_W_at_t = delta_W_at_t.unsqueeze(1)
                            delta_W_at_t = delta_W_at_t.cuda()  # [M, 1, C, D]

                            condition_at_t = torch.cat((previous_W_at_t.expand(DM_W_batch.shape).unsqueeze(1),
                                                        DM_prototype_batch.unsqueeze(1)), dim=1) # condition_at_t: [M, 2, C, D]
                            condition_at_t = condition_at_t.cuda()
                        loss_diff = self.diffusion_model.training_step(delta_W_at_t, y, weights_condition=[condition_at_t])
                        self.DM_optimizer.zero_grad()
                        loss_diff.backward()
                        self.DM_optimizer.step()
                        meters.update(loss_diff=(loss_diff).item())
                        torch.cuda.empty_cache()
                        gc.collect()
                self.network = self.network.cuda()
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iters - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
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
                val_acc = self.network_evaluation(test_id_dataloader, use_diffusion=False)
                if best_val_acc <= val_acc:
                    best_W = self.network.classifier.weight.detach().cpu()
                self.logger.info("[{}/{}]  ID timestamp = {}: \t validation {}: {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, val_acc * 100.0))

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))
        return best_W

    @torch.no_grad()
    def network_evaluation(self, test_time_dataloader, use_diffusion=False):
        print("evaluate_time={}".format(self.eval_dataset.current_time))
        self.network.eval()
        self.diffusion_model.eval()
        pred_all = []
        y_all = []

        if not use_diffusion:
            for _, sample in enumerate(test_time_dataloader):
                if len(sample) == 3:
                    x, y, _ = sample
                else:
                    x, y = sample
                x, y = prepare_data(x, y, str(self.eval_dataset))
                _, logits = self.network.foward(x)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()

            pred_all = np.array(pred_all)
            y_all = np.array(y_all)
            correct = (pred_all == y_all).sum().item()
            metric = correct / float(y_all.shape[0])
            self.network.train()
            self.diffusion_model.train()
            return metric
        else:
            previous_W = self.network.reference_point_queue.get_all_items()
            previous_W = torch.stack(previous_W, dim=0)
            num_previous_items = self.network.reference_point_queue.len()
            previous_W = previous_W.cuda()

            #-------------estimating prototypes----------
            prototype = None
            denominator = None
            sample_count = 0
            for _, sample in enumerate(test_time_dataloader):
                if len(sample) == 3:
                    x, y, _ = sample
                else:
                    x, y = sample
                x, y = prepare_data(x, y, str(self.eval_dataset))
                f, _ = self.network.foward(x)

                MxBxD_f = f.unsqueeze(0).expand(num_previous_items, f.shape[0], f.shape[1])
                logits = torch.bmm(MxBxD_f, previous_W.transpose(1, 2))
                softmax_outs = F.softmax(logits, dim=2)
                avg_softmax_outs = torch.mean(softmax_outs, dim=0)
                sample_count += f.shape[0]
                if prototype is None:
                    prototype = avg_softmax_outs.transpose(0, 1) @ f
                    denominator = torch.sum(avg_softmax_outs.transpose(0, 1), dim=1, keepdim=True)
                else:
                    prototype += avg_softmax_outs.transpose(0, 1) @ f
                    denominator += torch.sum(avg_softmax_outs.transpose(0, 1), dim=1, keepdim=True)
            prototype = prototype / sample_count
            self.logger.info("sample_count={}".format(sample_count))

            # -------------generating classifier weights----------
            prototype = prototype.unsqueeze(0)
            generated_weights = torch.zeros(self.args.trainer.Mg * num_previous_items, self.num_classes, self.network.feature_dim).cuda()
            for l in range(num_previous_items):
                condition = torch.cat((previous_W[l, :, :].unsqueeze(0), prototype), dim=0)
                condition = condition.expand(self.args.trainer.Mg, 2, self.num_classes, self.network.feature_dim)
                temp_generated_weights = self.diffusion_model.sample(weights_condition=[condition], batch_size=self.args.trainer.Mg)  # generated_weights.shape: [m, C, D]
                temp_generated_weights = temp_generated_weights.squeeze(1)
                temp_generated_weights = temp_generated_weights + previous_W[l, :, :].unsqueeze(0)
                generated_weights[l * self.args.trainer.Mg: (l + 1) * self.args.trainer.Mg, :, :] = temp_generated_weights

            #----------------evaluating-------------
            for _, sample in enumerate(test_time_dataloader):
                if len(sample) == 3:
                    x, y, _ = sample
                else:
                    x, y = sample
                x, y = prepare_data(x, y, str(self.eval_dataset))
                f, _ = self.network.foward(x)
                avg_generated_weights = torch.mean(generated_weights, dim=0)
                logits = torch.mm(f, avg_generated_weights.transpose(0, 1))
                softmax_outs = F.softmax(logits, dim=1)
                pred = softmax_outs.argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            y_all = np.array(y_all)
            correct = (pred_all == y_all).sum().item()
            metric = correct / float(y_all.shape[0])
            self.network.train()
            self.diffusion_model.train()
            return metric

    def train_online(self):
        self.train_dataset.mode = 0
        val_acc_list = []
        all_weights_list = []
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if timestamp == (self.split_time + 1):
                break
            else:
                self.train_dataset.update_current_timestamp(timestamp)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                best_W = self.train_step(train_dataloader)
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                if i > 0:
                    acc = self.network_evaluation(test_id_dataloader, use_diffusion=True)
                    self.logger.info("ID timestamp = {}: \t validation {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                else:
                    acc = self.network_evaluation(test_id_dataloader, use_diffusion=False)
                    self.logger.info("ID timestamp = {}: \t validation {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                self.network.memorize(best_W)
                all_weights_list.append(best_W)
                val_acc_list.append(acc * 100.0)
        mean_val_acc = np.mean(val_acc_list)
        self.logger.info("average of validation {} is {:.3f}".format(self.eval_metric, mean_val_acc))
        return all_weights_list

    def evaluate_offline(self):
        self.logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
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
                id_metric = self.network_evaluation(test_id_dataloader, use_diffusion=True)
                self.logger.info("Merged ID validation data: {}: \t{:.3f}\n".format(self.eval_metric, id_metric * 100.0))
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader, use_diffusion=True)
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
        if len(metrics) >= 2:
            self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))

    def save_checkpoint(self):
        save_dict = {
            "enc": self.network.cpu().enc.state_dict(),
            "reference_point_queue": self.network.reference_point_queue.get_all_items(),
            "diffusion_model": self.diffusion_model.cpu().state_dict(),
        }
        torch.save(save_dict, "./checkpoints/{}/{}_model.pkl".format(self.args.data.dataset.lower(), self.args.data.dataset.lower()))
        self.network.cuda()
        self.diffusion_model.cuda()

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        all_weights_list = self.train_online()
        self.save_checkpoint()
        self.evaluate_offline()
