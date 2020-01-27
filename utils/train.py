import os, shutil
import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from config import config
from tensorboardX import SummaryWriter
from utils.generators import DataGenerator
from utils.evaluators import fp_fn_image_csi
from datetime import datetime

def k_train(k_fold, model, loss_func,
            batch_size, max_iterations, save_dir='./logs', eval_every=500, checkpoint_every=1000):

    save_dir += datetime.now().strftime("_%m_%d_%H_%M")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_gen = DataGenerator(config['DATA_PATH'], k_fold, 
        batch_size, config['IN_LEN'], config['OUT_LEN'], config['IN_LEN'] + config['OUT_LEN'])
    writer = SummaryWriter(os.path.join(save_dir, 'train_logs'))

    for k in range(1, k_fold + 1):

        k_model, optimizer, lr_scheduler = model()
        data_gen.set_k(k)
        train_loss = 0.0
        train_csi = 0.0
        train_count = 0
        i_batch = 1
        
        pbar = tqdm(range(1, max_iterations))
        for itera in pbar:
            
            n_train_batch = data_gen.n_train_batch()
            pbar_b = tqdm(range(data_gen.n_train_batch()))
            for b in pbar_b:
                
                pbar.set_description("Training at batch %d / %d" % (i_batch - 1, n_train_batch))

                train_data, train_label = data_gen.get_train(b)
                k_model.train()
                optimizer.zero_grad()
                output = k_model(train_data)
                loss = loss_func(output, train_label)
                loss.backward()
                torch.nn.utils.clip_grad_value_(k_model.parameters(), clip_value=50.0)
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()
                train_csi += fp_fn_image_csi(output.cpu().detach().numpy(), train_label.cpu().numpy())
                train_count += train_data.shape[1]

                if i_batch % eval_every == 0:

                    val_loss = 0.0
                    val_csi = 0.0
                    val_count = 0

                    with torch.no_grad():
                        k_model.eval()
                        n_val_batch = data_gen.n_val_batch()
                        for b_val in range(n_val_batch):
                            val_data, val_label = data_gen.get_val(b_val)
                            output = k_model(val_data)
                            loss = loss_func(output, val_label)
                            val_loss += loss.item()
                            val_csi += fp_fn_image_csi(output.cpu().detach().numpy(), train_label.cpu().numpy())
                            val_count += val_data.shape[1]
                            pbar.set_description("Validating at batch %d / %d" % (i_batch - 1, n_val_batch))

                    train_loss /= train_count
                    train_csi /= train_count
                    val_loss /= val_count
                    val_csi /= val_count

                    writer.add_scalars('loss', {
                        'train': train_loss,
                        'valid': val_loss
                    }, i_batch)

                    writer.add_scalars('csi', {
                        'train': train_csi,
                        'valid': val_csi
                    }, i_batch)

                    train_loss = 0.0
                    train_count = 0
                    train_csi = 0.0
                
                if i_batch % checkpoint_every == 0:
                    torch.save(k_model.state_dict(), os.path.join(save_dir, 
                        'model_f{}__i{}.pth'.format(k, i_batch)))

                i_batch += 1
        
    writer.close()