import torch
from torch.autograd import Variable
from torchvision import models, utils
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import matplotlib.pyplot as plt
import pdb
import time

# this code was not training the convolution filters previously i.e. only training the classifier stage, now it is.

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.resnet18(pretrained=True)
        self.features = model.features
        self.imagenet_classifier = model.classifier

        # change param.requires_grad = False to freeze all the convolution layers
        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2))

    def forward(self, x, imagenet=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if not imagenet:
            x = self.classifier(x)
        else:
            x = self.imagenet_classifier(x)
        return x


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        # print "Hook executed"
        # pdb.set_trace()
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = torch.sum((activation * grad), dim=-1).sum(dim=-1).sum(dim=0).data

        # Normalize the rank by the filter dimensions
        values = values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, imagenet_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)
        self.imagenet_val_loader = dataset.test_loader(imagenet_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self, imagenet=False):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            output = model(Variable(batch), imagenet=False)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print "Accuracy on custom data:", float(correct) / total

        if imagenet:
            correct = 0
            total = 0

            for i, (batch, label) in enumerate(self.imagenet_val_loader):
                # grid = utils.make_grid(batch)
                # plt.imshow(grid.numpy().transpose((1, 2, 0)))

                batch = batch.cuda()
                output = model(Variable(batch), imagenet=True)
                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(label).sum()
                total += label.size(0)

            print "Accuracy on ILSVRC2012 subset of 5000 images:", float(correct) / total

        self.model.train()

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = \
                optim.SGD(model.parameters(),
                          # changed model.classifier.parameters() to model.parameters() to train all layers and not just classifier layers
                          lr=0.0001, momentum=0.9)
        self.test(imagenet=True)
        for i in range(epoches):
            print "Epoch: ", i
            self.train_epoch(optimizer)
            self.test(imagenet=True)
        print "Finished fine tuning."

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()

        self.train_epoch(rank_filters=True)

        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        self.test(imagenet=True)
        self.model.train()

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print "Number of prunning iterations to reduce 67% filters", iterations

        for _ in range(iterations):
            print "Ranking filters.. "
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print "Layers that will be prunned", layers_prunned
            print "Prunning filters.. "
            model = self.model.cpu()
            # pdb.set_trace()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index)

            self.model = model.cuda()

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print "Filters prunned", str(message)
            print "Now testing with the pruned model before fine tuning"
            self.test(imagenet=True)
            print "Fine tuning to recover from prunning iteration."
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches=4)

        print "Finished. Going to fine tune the model a bit more"
        self.train(optimizer, epoches=4)
        print "Saving pruned model"
        torch.save(model, "model_prunned")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--imagenet_path", type=str, default="/data1/ImageNet_Fall2011/ILSVRC2012_grouped")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    t1 = time.time()
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model().cuda()
    elif args.prune:
        model = torch.load("model").cuda()

    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, args.imagenet_path, model)

    if args.train:
        fine_tuner.train(epoches=8)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()
    print "Time elapsed: ", time.time() - t1, " seconds"
