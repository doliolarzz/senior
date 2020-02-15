import os, shutil
import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from global_config import global_config
from tensorboardX import SummaryWriter
from utils.generators import DataGenerator
from utils.evaluators import fp_fn_image_csi
from datetime import datetime
from utils.units import dbz_mm

def k_train(k_fold, model, loss_func,
            batch_size, max_iterations, save_dir='./logs', eval_every=100, 
            checkpoint_every=1000, multitask=False, config=None):

    cel_cri = None
    if multitask:
        cel_cri = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([99])).to(config['DEVICE'])

    save_dir += datetime.now().strftime("_%m_%d_%H_%M")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_gen = DataGenerator(global_config['DATA_PATH'], k_fold, 
        batch_size, config['IN_LEN'], config['OUT_LEN'], config['IN_LEN'] + config['OUT_LEN'], config=config)
    writer = SummaryWriter(os.path.join(save_dir, 'train_logs'))

    for k in range(1, k_fold + 1):

        k_model, optimizer, lr_scheduler = model()
        
        data_gen.set_k(k)
        train_loss = 0.0
        train_csi = 0.0
        train_count = 0
        i_batch = 0
        best_val_loss = np.inf
        
        pbar = tqdm(range(1, max_iterations + 1))
        for itera in pbar:
            
            n_train_batch = data_gen.n_train_batch()
            pbar_b = tqdm(np.random.choice(data_gen.n_train_batch(), 2000))#range(data_gen.n_train_batch()))
            for b in pbar_b:
                
                pbar.set_description("Fold %d Training at batch %d / %d" % (k, i_batch, n_train_batch))

                train_data, train_label = data_gen.get_train(b)
                k_model.train()
                optimizer.zero_grad()
                output = k_model(train_data)
                loss = loss_func(output, train_label)
                n_t, n_b, n_c, n_h, n_w = output.size()
                if multitask:
                    loss += cel_cri((output>=0.2).float().view(n_t*n_b,-1), (train_label>=0.2).float().view(n_t*n_b,-1))
                loss.backward()
                torch.nn.utils.clip_grad_value_(k_model.parameters(), clip_value=50.0)
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()
                train_csi += fp_fn_image_csi(dbz_mm(output.cpu().detach().numpy()), dbz_mm(train_label.cpu().numpy()))
                train_count += 1

                if i_batch % eval_every == 0:

                    val_loss = 0.0
                    val_csi = 0.0
                    val_count = 0

                    with torch.no_grad():
                        k_model.eval()
                        n_val_batch = data_gen.n_val_batch()

                        for ib_val, b_val in enumerate(np.random.choice(n_val_batch, 20)): #range(n_val_batch)
                            val_data, val_label = data_gen.get_val(b_val)
                            output = k_model(val_data)
                            loss = loss_func(output, val_label)
                            n_t, n_b, n_c, n_h, n_w = output.size()
                            if multitask:
                                loss += cel_cri((output>=0.2).float().view(n_t*n_b,-1), (val_label>=0.2).float().view(n_t*n_b,-1))
                            val_loss += loss.item()
                            val_csi += fp_fn_image_csi(dbz_mm(output.cpu().detach().numpy()), dbz_mm(val_label.cpu().numpy()))
                            val_count += 1
                            pbar.set_description("Fold %d Validating at batch %d / %d" % (k, ib_val, 20))

                    train_loss /= train_count
                    train_csi /= train_count
                    val_loss /= val_count
                    val_csi /= val_count

                    writer.add_scalars('loss/'+str(k), {
                        'train': train_loss,
                        'valid': val_loss
                    }, i_batch)

                    writer.add_scalars('csi/'+str(k), {
                        'train': train_csi,
                        'valid': val_csi
                    }, i_batch)

                    train_loss = 0.0
                    train_count = 0
                    train_csi = 0.0
                    
                    if val_loss <= best_val_loss:
                        torch.save(k_model.state_dict(), os.path.join(save_dir, 
                            'model_f{}_i{}_best.pth'.format(k, i_batch)))
                        best_val_loss = val_loss
                
                if i_batch % checkpoint_every == 0:
                    torch.save(k_model.state_dict(), os.path.join(save_dir, 
                        'model_f{}_i{}.pth'.format(k, i_batch)))

                i_batch += 1
        try:
            torch.save(k_model.state_dict(), os.path.join(save_dir, 
                            'model_f{}_i{}.pth'.format(k, i_batch)))
        except:
            print('cannot save model')
        
    writer.close()