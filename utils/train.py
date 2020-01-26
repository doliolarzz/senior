import os, shutil
import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from ..models.config import config
from tensorboardX import SummaryWriter
from generators import DataGenerator
from evaluators import fp_fn_image_csi

def k_train(k_fold, model, optimizer, loss_func, lr_scheduler, 
            batch_size, max_iterations, save_dir, eval_every=50, checkpoint_every=1000):

    data_gen = DataGenerator(config['DATA_PATH'], k_fold, 
        batch_size, config['IN_LEN'], config['OUT_LEN'])
    writer = SummaryWriter(os.path.join(save_dir, 'train_logs'))

    for k in range(k_fold):

        k_model = model()
        data_gen.set_k(k)
        train_loss = 0.0
        train_csi = 0.0
        train_count = 0
        i_batch = 0

        for i in tqdm(range(1, max_iterations)):

            for b, bs in range(data_gen.n_train_batch()):

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
                train_count += bs

                if i_batch % eval_every == 0:

                    val_loss = 0.0
                    val_csi = 0.0
                    val_count = 0

                    with torch.no_grad():
                        k_model.eval()
                        for b_val, bs_val in range(data_gen.n_val_batch()):
                            val_data, val_label = data_gen.get_val(b_val)
                            output = k_model(val_data)
                            loss = loss_func(output, val_label)
                            val_loss += loss.item()
                            val_csi += fp_fn_image_csi(output, val_label)
                            val_count += bs_val

                    train_loss /= train_count
                    train_csi /= train_count
                    val_loss /= val_count
                    val_csi /= val_count

                    writer.add_scalars('loss', {
                        'train': train_loss,
                        'valid': val_loss
                    }, i * b)

                    writer.add_scalars('csi', {
                        'train': train_csi,
                        'valid': val_csi
                    }, i * b)

                    train_loss = 0.0
                    train_count = 0
                    train_csi = 0.0
                
                if i_batch % checkpoint_every == 0:
                    torch.save(k_model.state_dict(), os.path.join(save_dir, 
                        'model_f{}__i{}.pth'.format(k, i_batch)))

                i_batch += 1
        
        data_gen.shuffle()
        
    writer.close()