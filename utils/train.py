import os, shutil
import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from generators import DataGenerator
from evaluators import fp_fn_image_csi

def k_train(k_fold, model, optimizer, loss_func, lr_scheduler, 
            batch_size, max_iterations, eval_every=16):

    data_gen = DataGenerator(k_fold)
    writer = SummaryWriter(log_dir)

    for k in range(k_fold):

        k_model = model()
        data_gen.set_k(k)
        train_loss = 0.0
        train_csi = 0.0

        for i in tqdm(range(1, max_iterations)):

            for b in range(data_gen.n_train_batch()):

                train_data, train_label = data_gen.get_train(b)
                k_model.train()
                optimizer.zero_grad()
                output = k_model(train_data)
                loss = loss_func(output, train_label)
                loss.backward()
                torch.nn.utils.clip_grad_value_(k_model.parameters(), clip_value=50.0)
                optimizer.step()
                train_loss += loss.item()
                train_csi += fp_fn_image_csi(output, train_label)

                if (i * b) % eval_every == 0:
                    val_loss = 0.0
                    val_csi = 0.0
                    with torch.no_grad():
                        k_model.eval()
                        for b_val in range(data_gen.n_val_batch()):
                            val_data, val_label = data_gen.get_val(b_val)
                            output = k_model(val_data)
                            loss = loss_func(output, val_label)
                            val_loss += loss.item()
                            val_csi += fp_fn_image_csi(output, val_label)

                    train_loss /= eval_every
                    train_csi /= data_gen.n_train()
                    val_loss /= data_gen.n_val_batch()
                    val_csi /= data_gen.n_val()

                    # writer.add_scalars('loss', {
                    #     'train': train_loss,
                    #     'valid': valid_loss
                    # }, i * b)

                    train_loss = 0.0
                    train_csi = 0.0

        data_gen.shuffle()