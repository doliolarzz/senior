import os, shutil
import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from generators import DataGenerator

def k_train(k_fold, model, optimizer, loss_func, lr_scheduler, 
            batch_size, max_iterations, eval_every=1):

    data_gen = DataGenerator()
    for k in range(k_fold):

        k_model = model()
        train_loss = 0

        for i in tqdm(range(1, max_iterations)):

            for b in data_gen.size():

                train_data, train_label = data_gen.get(b)
                k_model.train()
                optimizer.zero_grad()
                output = k_model(train_data)
                loss = loss_func(output, train_label)
                loss.backward()
                torch.nn.utils.clip_grad_value_(k_model.parameters(), clip_value=50.0)
                optimizer.step()
                train_loss += loss.item()

                if (i * b) % eval_every == 0:
                    train_csi = 0
                    train_loss = 0

        data_gen.shuffle()