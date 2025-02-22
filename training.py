import sys
import numpy as np
import sklearn.metrics
from copy import deepcopy

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import settings
from reproduction import lib_data
import mlflow


class Training(object):

    def __init__(self, model=None, lr=5e-3, weight_decay=1e-2,
                 milestones=None, lr_d=0.1, balanced=False):
        self.times = 0
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        if milestones is None:
            self.milestones = [1000]
        else:
            self.milestones = milestones
        self.lr_d = lr_d
        self.balanced = balanced
        self.min_vloss = float('inf')
        self.final_model = None
        self.test_accuracies = []

    def init_optimizer(self, optimizer):
        if optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise RuntimeError("unknown optimizer")
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=self.lr_d)

    def validation(self, *args):
        _, validation_loss = self.compute_loss(*args, validation=True)
        print('Validation Loss:', validation_loss.item(), flush=True)
        if validation_loss.item() < self.min_vloss:
            self.min_vloss = validation_loss.item()
            self.final_model = deepcopy(self.model)
        return validation_loss

    @staticmethod
    def feed_together(model, *args):
        split_sizes = []
        for x in args:
            split_sizes.append(x.size(0))
        x_cat = torch.cat(args).type(settings.dtype)
        fx = model(x_cat)
        return torch.split(fx, split_sizes)

    @staticmethod
    def feed_in_batches(model, x, batch_size=None):
        if batch_size is None:
            batch_size = settings.test_batch_size
        model.eval()
        if len(x) <= batch_size:
            with torch.no_grad():
                fx = model(x.type(settings.dtype))
            return fx
        fxs = []
        for i in range(0, x.size(0), batch_size):
            x_batch = x[i: min(i+batch_size, x.size(0))]
            with torch.no_grad():
                fxs.append(model(x_batch.type(settings.dtype)))
        return torch.cat(fxs)[:x.size(0)]

    def compute_classification_metrics(
            self, labels, pred, output=None, to_print=True, epoch=None, is_final_model=False):
        target = labels.numpy().reshape(-1).copy()
        target[target == -1] = 0
        pred = pred.cpu().numpy().reshape(-1)
        pred[pred == -1] = 0
        accuracy = sklearn.metrics.accuracy_score(target, pred)
        self.test_accuracies.append(accuracy)
        # If pred is binary input this is correct
        balanced_accuracy =\
            sklearn.metrics.roc_auc_score(target, pred)
        if output is not None:
            output = output.cpu().numpy().reshape(-1)
            auc_score = sklearn.metrics.roc_auc_score(target, output)
        # Return result for each class
        precision, recall, f1_score, _ =\
            sklearn.metrics.precision_recall_fscore_support(target, pred)
        n_negative = np.sum(target == 0)
        n_false_positive = np.sum(np.logical_and(pred == 1, target == 0))

        prefix = "heur_" if is_final_model else ""
        # First add the metrics as they were computed in the original code.
        metrics = {
            f"{prefix}orig_test_acc": accuracy,
            f"{prefix}orig_test_tpr": np.sum(np.logical_and(pred == 1, target == 1)) / np.sum(target == 1),
            f"{prefix}orig_test_tnr": np.sum(np.logical_and(pred == 0, target == 0)) / np.sum(target == 0),
            f"{prefix}orig_auroc": auc_score,
        }

        # Then we add our own metrics. For the sake of consistency with the other OOD detection methods, we will label
        # the training set as negative and the outliers as positives (so the opposite of how PU learning labels them).
        our_metrics = lib_data.get_eval_metrics(
            id_y_true=1 - target[target == 1],
            id_y_pred=1 - pred[target == 1],
            id_test_statistic=1 - output[target == 1],
            ood_y_true=1 - target[target == 0],
            ood_y_pred=1 - pred[target == 0],
            ood_test_statistic=1 - output[target == 0],
        )
        if is_final_model:
            our_metrics = {f"heur_{k}": v for k, v in our_metrics.items()}
        metrics.update(our_metrics)

        lib_data.retry(lambda: mlflow.log_metrics(metrics, step=epoch))

        if to_print:
            print('Test set: Accuracy: {:.2f}%'
                  .format(accuracy*100), flush=True)
            print('Test set: Balanced Accuracy: {:.2f}%'
                  .format(balanced_accuracy*100), flush=True)
            if output is not None:
                print('Test set: Auc Score: {:.2f}%'
                      .format(auc_score*100), flush=True)
            print('Test set: Precision: {:.2f}%'
                  .format(precision[1]*100), flush=True)
            print('Test set: Recall Score: {:.2f}%'
                  .format(recall[1]*100), flush=True)
            print('Test set: F1 Score: {:.2f}%'
                  .format(f1_score[1]*100), flush=True)
            print('Test set: False Positive Rate: {:.2f}%'
                  .format(n_false_positive/n_negative*100), flush=True)

    def last_layer_activation(self, x):
        features = []

        def hook(module, input, output):
            features.append(output)

        getattr(self.model, 'fc1').register_forward_hook(hook)
        self.feed_in_batches(
            self.model, x, settings.test_batch_size)
        print(torch.cat(features, dim=0))
        return torch.cat(features, dim=0)

    def train(self, *args):
        raise NotImplementedError

    def test(self, *args):
        raise NotImplementedError

    def compute_loss(self, *args):
        raise NotImplementedError


class Classifier(Training):

    def test(self, test_set, to_print=True, epoch=None, is_final_model=False):
        x = test_set.tensors[0]
        labels = test_set.tensors[1]
        output = self.feed_in_batches(self.model, x, settings.test_batch_size)
        pred = torch.sign(output)
        self.compute_classification_metrics(labels, pred, output, to_print, epoch, is_final_model)

    def basic_loss(self, fx, convex):
        if convex:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)


class ClassifierFrom2(Classifier):

    def __init__(self, model, pi=0.5, *args, **kwargs):
        self.pi = pi
        super().__init__(model, *args, **kwargs)
        if self.balanced:
            self.p_weight = 1
            self.n_weight = 1
        else:
            self.p_weight = pi
            self.n_weight = 1-pi

    def train(self, p_set, n_set, test_set,
              p_batch_size, n_batch_size,
              p_validation, n_validation,
              num_epochs, convex_epochs=5,
              test_interval=1, print_interval=1, optimizer="adam"):

        self.init_optimizer(optimizer)
        self.test(test_set, True)

        p_loader = torch.utils.data.DataLoader(
            p_set, batch_size=p_batch_size,
            shuffle=True, num_workers=0)

        n_loader = torch.utils.data.DataLoader(
            n_set, batch_size=n_batch_size,
            shuffle=True, num_workers=0)

        for epoch in range(num_epochs):

            # XXX: warm-up with log sigmoid before training with sigmoid?
            convex = True if epoch < convex_epochs else False
            average_loss = self.train_step(
                p_loader, n_loader,
                p_validation, n_validation, convex)

            validation_loss = self.validation(p_validation, n_validation, convex)
            if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:

                p_output = self.feed_in_batches(self.model, p_set.tensors[0], settings.test_batch_size)
                n_output = self.feed_in_batches(self.model, n_set.tensors[0], settings.test_batch_size)
                p_pred = torch.sign(p_output)
                n_pred = torch.sign(n_output)
                train_acc = (p_pred[p_pred > 0].shape[0] + n_pred[n_pred < 0].shape[0]) / (p_pred.shape[0] + n_pred.shape[0])

                lib_data.retry(lambda: mlflow.log_metrics({
                    "lr": self.scheduler.get_lr()[0],
                    "val_loss": validation_loss.item(),
                    "train_loss": average_loss,
                    "train_acc": train_acc,
                    "train_p_acc": p_pred[p_pred > 0].shape[0] / p_pred.shape[0],
                    "train_u_acc": n_pred[n_pred > 0].shape[0] / n_pred.shape[0],
                }, step=epoch))

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                    print('Train Loss: {:.6f}; Train Acc: {:.6f}'.format(average_loss, train_acc))
                self.test(test_set, to_print, epoch=epoch)

        if self.final_model is not None:
            self.model = self.final_model
            print('Final error:')
            self.test(test_set, True, is_final_model=True)

    def train_step(self, p_loader, n_loader,
                   p_validation, n_validation, convex):
        self.scheduler.step()
        self.times += 1
        losses = []
        for i, x in enumerate(p_loader):
            self.model.train()
            self.optimizer.zero_grad()
            nx = next(iter(n_loader))
            loss, true_loss = self.compute_loss(x, nx, convex)
            losses.append(true_loss.item())
            loss.backward()
            self.optimizer.step()
            # if (i+1) % settings.validation_interval == 0:
            #     # print(p_validation[0].shape)
            #     # print(n_validation[0].shape)
            #     self.validation(p_validation, n_validation, convex)
        self.optimizer.zero_grad()
        return np.mean(np.array(losses))


class PNClassifier(ClassifierFrom2):

    def __init__(self, model, pu_model=None,
                 adjust_p=False, adjust_n=False, *args, **kwargs):
        if pu_model is not None:
            self.pu_model = pu_model
            self.test = self.test_two_stage
        self.adjust_p = adjust_p
        self.adjust_n = adjust_n
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, nx, convex, validation=False):
        if validation:
            fpx = self.feed_in_batches(self.model, px[0])
            fnx = self.feed_in_batches(self.model, nx[0])
            convex = False
        else:
            fpx, fnx = self.feed_together(self.model, px[0], nx[0])
        if self.adjust_p:
            fpx_prob = px[1].type(settings.dtype)
            p_loss = self.p_weight * torch.mean(
                self.basic_loss(fpx, convex) / fpx_prob)
        else:
            p_loss = self.p_weight * torch.mean(self.basic_loss(fpx, convex))
        if self.adjust_n:
            fnx_prob = nx[1].type(settings.dtype)
            n_loss = self.n_weight * torch.mean(
                self.basic_loss(-fnx, convex) / fnx_prob)
        else:
            n_loss = self.n_weight * torch.mean(self.basic_loss(-fnx, convex))
        loss = p_loss + n_loss
        return loss.cpu(), loss.cpu()

    def test_two_stage(self, test_set, to_print=True):
        x = test_set.tensors[0]
        labels = test_set.tensors[1]
        output1 = self.feed_in_batches(
            self.pu_model, x, settings.test_batch_size)
        pred1 = torch.sign(output1)
        output2 = self.feed_in_batches(
            self.model, x, settings.test_batch_size)
        pred = torch.sign(output2)
        pred[pred1 == -1] = -1
        self.compute_classification_metrics(labels, pred, to_print=to_print)


# XXX: nnPU classifier.
class PUClassifier(ClassifierFrom2):

    def __init__(self, model, pn_model=None,
                 nn=True, nn_threshold=0, nn_rate=1/100, prob_est=False,
                 *args, **kwargs):
        self.nn_rate = nn_rate
        self.nn_threshold = nn_threshold
        self.nn = nn
        if pn_model is not None:
            self.pn_model = pn_model
            self.test = self.test_two_stage
        if prob_est:
            self.test = self.test_prob_est
        super().__init__(model, *args, **kwargs)

    def compute_loss(self, px, ux, convex, validation=False):
        # XXX: ignore completely px[1] and ux[1] which contain the posterior probability for each sample
        px, ux = px[0], ux[0]
        if validation:
            fpx = self.feed_in_batches(self.model, px)
            fux = self.feed_in_batches(self.model, ux)
            convex = False
        else:
            fpx, fux = self.feed_together(self.model, px, ux)
        p_loss = self.p_weight * torch.mean(self.basic_loss(fpx, convex))
        n_loss = (torch.mean(self.basic_loss(-fux, convex))
                  - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
        true_loss = p_loss + n_loss/(1-self.pi)*self.n_weight
        loss = true_loss
        # if not validation:
        #     print(n_loss.item())
        if self.nn and n_loss < self.nn_threshold:
            loss = -n_loss * self.nn_rate
        return loss.cpu(), true_loss.cpu()

    def test_prob_est(self, test_set, to_print=True):
        x = test_set.tensors[0]
        target = test_set.tensors[2]
        target = target/torch.mean(target)*self.pi
        output = torch.sigmoid(self.feed_in_batches(
            self.model, x, settings.test_batch_size)).cpu()
        output = output/torch.mean(output)*self.pi
        error = torch.mean((target-output)**2).item()
        if to_print:
            print('Test set: Error: {}'.format(error), flush=True)

    def test_two_stage(self, test_set, to_print=True):
        x = test_set.tensors[0]
        labels = test_set.tensors[1]
        output1 = self.feed_in_batches(
            self.pn_model, x, settings.test_batch_size)
        pred1 = torch.sign(output1)
        output2 = self.feed_in_batches(
            self.model, x, settings.test_batch_size)
        pred = torch.sign(output2)
        pred[pred1 == -1] = -1
        self.compute_classification_metrics(labels, pred, to_print=to_print)


# class ClassifierFrom3(Classifier):
#
#     def __init__(self, model, pi=0.5, rho=0.1, *args, **kwargs):
#         self.pi = pi
#         self.rho = rho
#         super().__init__(model, *args, **kwargs)
#
#     def train(self, p_set, sn_set, u_set, test_set,
#               p_batch_size, sn_batch_size, u_batch_size,
#               p_validation, sn_validation, u_validation,
#               num_epochs, convex_epochs=5,
#               test_interval=1, print_interval=1):
#
#         self.init_optimizer()
#         self.test(test_set, True)
#
#         p_loader = torch.utils.data.DataLoader(
#             p_set, batch_size=p_batch_size,
#             shuffle=True, num_workers=0)
#
#         if self.rho != 0:
#             sn_loader = torch.utils.data.DataLoader(
#                 sn_set, batch_size=sn_batch_size,
#                 shuffle=True, num_workers=0)
#         else:
#             sn_loader = None
#
#         u_loader = torch.utils.data.DataLoader(
#             u_set, batch_size=u_batch_size,
#             shuffle=True, num_workers=0)
#
#         for epoch in range(num_epochs):
#
#             convex = True if epoch < convex_epochs else False
#             avg_loss = self.train_step(
#                 p_loader, sn_loader, u_loader,
#                 p_validation, sn_validation, u_validation, convex)
#
#             if (epoch+1) % test_interval == 0 or epoch+1 == num_epochs:
#
#                 to_print = (epoch+1) % print_interval == 0
#                 if to_print:
#                     sys.stdout.write('Epoch: {}  '.format(epoch))
#                     print('Train Loss: {:.6f}'.format(avg_loss))
#                 self.test(test_set, to_print)
#
#         if self.final_model is not None:
#             self.model = self.final_model
#             print('Final error:')
#             self.test(test_set, True)
#
#     def train_step(self, p_loader, sn_loader, u_loader,
#                    p_validation, sn_validation, u_validation, convex):
#         self.scheduler.step()
#         self.times += 1
#         losses = []
#         for i, x in enumerate(p_loader):
#             self.model.train()
#             if sn_loader is not None:
#                 snx = next(iter(sn_loader))
#             else:
#                 snx = None
#             ux = next(iter(u_loader))
#             loss, true_loss = self.compute_loss(x, snx, ux, convex)
#             losses.append(true_loss.item())
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             if (i+1) % settings.validation_interval == 0:
#                 self.validation(
#                     p_validation, sn_validation, u_validation, convex)
#         print('Training CE loss', true_loss.item())
#         return np.mean(np.array(losses))
#
#     def compute_pu_loss(self, px, nx, ux):
#         px, nx, ux = px[0], nx[0], ux[0]
#         fpx = self.feed_in_batches(self.model, px)
#         fux = self.feed_in_batches(self.model, ux)
#         p_loss = self.pi * torch.mean(self.basic_loss(fpx, False))
#         # p_loss = self.pi * torch.mean((fpx <= 0).float())
#         n_loss = (torch.mean(self.basic_loss(-fux, False))
#                   - self.pi * torch.mean(self.basic_loss(-fpx, False)))
#         # n_loss = (torch.mean((fux > 0).float())
#         #           - self.pi * torch.mean((fpx > 0).float()))
#         loss = p_loss + n_loss
#         return loss.cpu(), loss.cpu()
#
#
# class PUClassifier3(ClassifierFrom3):
#     """This classifier is used in the first step of PUbN learning"""
#
#     def __init__(self, model,
#                  nn=True, nn_threshold=0, nn_rate=1/2,
#                  prob_est=False, *args, **kwargs):
#         self.nn_rate = nn_rate
#         self.nn_threshold = nn_threshold
#         self.nn = nn
#         self.prob_est = prob_est
#         if prob_est:
#             self.test = self.test_prob_est
#             self.validation = self.validation_prob_est
#         super().__init__(model, *args, **kwargs)
#
#     def compute_loss(self, px, snx, ux, convex, validation=False):
#         if self.rho != 0:
#             px, snx, ux = px[0], snx[0], ux[0]
#         else:
#             px, ux = px[0], ux[0]
#         if validation:
#             fpx = self.feed_in_batches(self.model, px)
#             if self.rho != 0:
#                 fsnx = self.feed_in_batches(self.model, snx)
#             fux = self.feed_in_batches(self.model, ux)
#             convex = False
#         else:
#             if self.rho != 0:
#                 fpx, fsnx, fux = self.feed_together(self.model, px, snx, ux)
#             else:
#                 fpx, fux = self.feed_together(self.model, px, ux)
#         p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
#         # fux_mean = torch.mean(torch.sigmoid(fux))
#         if self.rho != 0:
#             sn_loss = self.rho * torch.mean(self.basic_loss(fsnx, convex))
#             n_loss = (torch.mean(self.basic_loss(-fux, convex))
#                       - self.rho * torch.mean(self.basic_loss(-fsnx, convex))
#                       - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
#         else:
#             n_loss = (torch.mean(self.basic_loss(-fux, convex))
#                       - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
#         if self.rho != 0:
#             true_loss = p_loss + sn_loss + n_loss
#         else:
#             true_loss = p_loss + n_loss
#         loss = true_loss
#         # print(n_loss.item())
#         if self.nn and n_loss < self.nn_threshold:
#             loss = -n_loss * self.nn_rate
#         # loss += (fux_mean - self.pi - self.rho)**2
#         # true_loss += (fux_mean - self.pi - self.rho)**2
#         return loss.cpu(), true_loss.cpu()
#
#     def average_loss(self, fx):
#         negative_logistic = nn.LogSigmoid()
#         logistic_loss = torch.mean(-negative_logistic(fx))
#         sigmoid = nn.Sigmoid()
#         sigmoid_loss = torch.mean(sigmoid(-fx))
#         return torch.tensor([logistic_loss, sigmoid_loss])
#
#     def validation_prob_est(self, p_val, sn_val, u_val, convex):
#         p_val, sn_val, u_val = p_val[0], sn_val[0], u_val[0]
#         fpx = self.feed_in_batches(self.model, p_val)
#         fsnx = self.feed_in_batches(self.model, sn_val)
#         fux = self.feed_in_batches(self.model, u_val)
#         ls_loss = (torch.mean(torch.sigmoid(fux)**2)
#                    - 2 * torch.mean(torch.sigmoid(fpx)) * self.pi
#                    - 2 * torch.mean(torch.sigmoid(fsnx)) * self.rho)
#         print('Validation Ls Loss:', ls_loss.cpu().item(), flush=True)
#         p_loss = self.pi * self.average_loss(fpx)
#         sn_loss = self.rho * self.average_loss(fsnx)
#         n_loss = (self.average_loss(-fux)
#                   - self.rho * self.average_loss(-fsnx)
#                   - self.pi * self.average_loss(-fpx))
#         logistic_loss = p_loss[0] + sn_loss[0] + n_loss[0]
#         print('Validation Log Loss:', logistic_loss.cpu().item(), flush=True)
#         sigmoid_loss = p_loss[1] + sn_loss[1] + n_loss[1]
#         print('Validation Sig Loss:', sigmoid_loss.cpu().item(), flush=True)
#         if ls_loss.item() < self.min_vloss:
#             self.min_vloss = ls_loss.item()
#             self.final_model = deepcopy(self.model)
#         return ls_loss
#
#     def test(self, test_set, to_print=True):
#         x = test_set.tensors[0]
#         target = test_set.tensors[2]
#         labels = torch.ones_like(target)
#         labels[target < 1/2] = -1
#         output = self.feed_in_batches(self.model, x, settings.test_batch_size)
#         pred = torch.sign(output)
#         self.compute_classification_metrics(labels, pred, output, to_print)
#
#     def test_prob_est(self, test_set, to_print=True):
#         x = test_set.tensors[0]
#         target = test_set.tensors[2]
#         output = self.feed_in_batches(
#                 self.model, x, settings.test_batch_size).cpu()
#         output_prob = torch.sigmoid(output)
#         error = torch.mean((target-output_prob)**2).item()
#         error_std = torch.std((target-output_prob)**2).item()
#         if to_print:
#             print('Test set: Error: {}'.format(error), flush=True)
#             print('Test set: Error Std: {}'.format(error_std), flush=True)
#         target_normalized = target/torch.mean(target)
#         output_prob = output_prob/torch.mean(output_prob)
#         error = torch.mean((target_normalized-output_prob)**2).item()
#         error_std = torch.std((target_normalized-output_prob)**2).item()
#         if to_print:
#             print('Test set: Normalized Error: {}'.format(error))
#             print('Test set: Normalized Error Std: {}'.format(error_std))
#         pred = torch.sign(output)
#         labels = torch.ones_like(target)
#         labels[target < 1/2] = -1
#         self.compute_classification_metrics(labels, pred, output, to_print)
#
#
# class PNUClassifier(ClassifierFrom3):
#
#     def __init__(self, model, pn_fraction=0.5,
#                  nn=True, nn_threshold=0, nn_rate=1/2,
#                  *args, **kwargs):
#         self.nn_rate = nn_rate
#         self.nn_threshold = nn_threshold
#         self.nn = nn
#         self.pn_fraction = pn_fraction
#         super().__init__(model, *args, **kwargs)
#
#     def compute_loss(self, px, nx, ux, convex, validation=False):
#         if validation:
#             return self.compute_pu_loss(px, nx, ux)
#         px, nx, ux = px[0], nx[0], ux[0]
#         fpx, fnx, fux = self.feed_together(self.model, px, nx, ux)
#         p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
#         n_loss = (1-self.pi) * torch.mean(self.basic_loss(-fnx, convex))
#         n_loss2 = (torch.mean(self.basic_loss(-fux, convex))
#                    - self.pi * torch.mean(self.basic_loss(-fpx, convex)))
#         n_loss = n_loss * self.pn_fraction + n_loss2 * (1-self.pn_fraction)
#         true_loss = p_loss + n_loss
#         loss = true_loss
#         if self.nn and n_loss < self.nn_threshold:
#             loss = -n_loss * self.nn_rate
#         return loss.cpu(), true_loss.cpu()
#
#
# class PUbNClassifier(ClassifierFrom3):
#
#     def __init__(self, model, sep_value=0.3,
#                  adjust_p=True, adjust_sn=True, hard_label=False,
#                  *args, **kwargs):
#         self.sep_value = sep_value
#         self.adjust_p = adjust_p
#         self.adjust_sn = adjust_sn
#         self.hard_label = hard_label
#         self.a = torch.tensor(20., requires_grad=True, device='cuda')
#         self.b = torch.tensor(10., requires_grad=True, device='cuda')
#         super().__init__(model, *args, **kwargs)
#
#     def train(self, p_set, sn_set, u_set, *args, **kwargs):
#         fux_prob = u_set.tensors[1]
#         print('Number of used unlabeled samples:',
#               torch.sum(fux_prob <= self.sep_value).item())
#         super().train(p_set, sn_set, u_set, *args, **kwargs)
#         # print(self.a, self.b)
#
#     def distribute(self, px):
#         return 1/(1+torch.exp(self.a*px-self.b))
#
#     def compute_loss(self, px, snx, ux, convex, validation=False):
#
#         if validation:
#             return self.compute_pu_loss(px, snx, ux)
#
#         if self.rho != 0:
#             fpx, fsnx, fux = self.feed_together(
#                 self.model, px[0], snx[0], ux[0])
#         else:
#             fpx, fux = self.feed_together(self.model, px[0], ux[0])
#
#         # Divide into two parts according to the value of p(s=1|x)
#         fux_prob = ux[1].type(settings.dtype)
#         fux = fux[fux_prob <= self.sep_value]
#         fux_prob = fux_prob[fux_prob <= self.sep_value]
#
#         if self.adjust_p:
#             fpx_prob = px[1].type(settings.dtype)
#             fpx_prob[fpx_prob <= self.sep_value] = 1
#             p_loss = self.pi * torch.mean(
#                 self.basic_loss(fpx, convex)
#                 + self.basic_loss(-fpx, convex) * (1-fpx_prob)/fpx_prob)
#             # * (1-fpx_prob)/fpx_prob*(1-self.distribute(fpx_prob)))
#         else:
#             p_loss = self.pi * torch.mean(self.basic_loss(fpx, convex))
#
#         if self.rho != 0:
#             if self.adjust_sn:
#                 fsnx_prob = snx[1].type(settings.dtype)
#                 fsnx_prob[fsnx_prob <= self.sep_value] = 1
#                 sn_loss = self.rho * torch.mean(
#                     self.basic_loss(-fsnx, convex) / fsnx_prob)
#                 # sn_loss = self.rho * torch.mean(
#                 #     self.basic_loss(-fsnx, convex)
#                 #     * (1 + (1-fsnx_prob)
#                 #         / fsnx_prob*(1-self.distribute(fsnx_prob))))
#             else:
#                 sn_loss = self.rho * torch.mean(self.basic_loss(-fsnx, convex))
#
#         if self.hard_label:
#             u_loss = torch.sum(
#                 self.basic_loss(-fux, convex)) / len(ux[0])
#         else:
#             u_loss = torch.sum(
#                 self.basic_loss(-fux, convex)*(1-fux_prob)/len(ux[0]))
#             # * self.distribute(fux_prob)) / len(ux[0])
#
#         if self.balanced:
#             p_loss = p_loss/self.pi
#             u_loss = u_loss/(1-self.pi)
#             if self.rho != 0:
#                 sn_loss = sn_loss/(1-self.pi)
#         if self.rho == 0:
#             loss = p_loss + u_loss
#         else:
#             loss = p_loss + sn_loss + u_loss
#         return loss.cpu(), loss.cpu()
#
#
# class ThreeClassifier(ClassifierFrom3):
#
#     def __init__(self, model,
#                  nn=True, nn_threshold=0, nn_rate=1/2,
#                  prob_pred=False, *args, **kwargs):
#         self.nn_rate = nn_rate
#         self.nn_threshold = nn_threshold
#         self.nn = nn
#         self.prob_pred = prob_pred
#         super().__init__(model, *args, **kwargs)
#
#     def compute_loss(self, px, snx, ux, convex, validation=False):
#         if len(px) <= 2:
#             px, snx, ux = px[0], snx[0], ux[0]
#         if validation:
#             fpx = self.feed_in_batches(self.model, px)
#             fsnx = self.feed_in_batches(self.model, snx)
#             fux = self.feed_in_batches(self.model, ux)
#         fpx, fsnx, fux = self.feed_together(self.model, px, snx, ux)
#         cross_entropy = nn.CrossEntropyLoss()
#         p_loss = self.pi * cross_entropy(
#             fpx, torch.zeros(len(px), dtype=torch.long).cuda())
#         sn_loss = self.rho * cross_entropy(
#             fsnx, torch.ones(len(snx), dtype=torch.long).cuda())
#         n_loss = (cross_entropy(
#                     fux, 2*torch.ones(len(ux), dtype=torch.long).cuda())
#                   - self.rho * cross_entropy(
#                       fsnx, 2*torch.ones(len(snx), dtype=torch.long).cuda())
#                   - self.pi * cross_entropy(
#                       fpx, 2*torch.ones(len(px), dtype=torch.long).cuda()))
#         true_loss = p_loss + sn_loss + n_loss
#         loss = true_loss
#         print(n_loss.item())
#         if self.nn and n_loss < self.nn_threshold:
#             loss = -n_loss * self.nn_rate
#         return loss.cpu(), true_loss.cpu()
#
#     def test(self, test_set, to_print=True):
#         x = test_set.tensors[0]
#         output = self.feed_in_batches(
#                 self.model, x, settings.test_batch_size).cpu()
#         output_prob = F.softmax(output, dim=1)
#         target = test_set.tensors[2]
#         est_posteriors = output_prob[:, 0:1] + output_prob[:, 1:2]
#         error = torch.mean((target - est_posteriors)**2).item()
#         error_std = torch.std((target - est_posteriors)**2).item()
#         if to_print:
#             print('Test set: Error: {}'.format(error), flush=True)
#             print('Test set: Error Std: {}'.format(error_std), flush=True)
#         target_normalized = target/torch.mean(target)
#         est_posteriors = est_posteriors/torch.mean(est_posteriors)
#         error = torch.mean((target_normalized - est_posteriors)**2).item()
#         error_std = torch.std((target_normalized - est_posteriors)**2).item()
#         if to_print:
#             print('Test set: Normalized Error: {}'.format(error))
#             print('Test set: Normalized Error Std: {}'.format(error_std))
#         labels = test_set.tensors[1]
#         if self.prob_pred:
#             pred = torch.sign(output_prob[:, 0:1] - 0.5)
#         else:
#             pred = -torch.ones(labels.size())
#             pred[torch.argmax(output, dim=1) == 0] = 1
#         self.compute_classification_metrics(labels, pred, output, to_print)
