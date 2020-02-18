import os, shutil
import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from global_config import global_config
from tensorboardX import SummaryWriter
from utils.losses import cross_entropy2d
from utils.generators import DataGenerator_FCN
from utils.evaluators import fp_fn_image_csi_muti
from datetime import datetime
from utils.units import dbz_mm
from sklearn.metrics import accuracy_score, f1_score

def k_train_fcn(k_fold, model,
            batch_size, max_iterations, save_dir='./logs', eval_every=100, 
            checkpoint_every=1000, mode='reg', config=None):
    
    mse_loss = torch.nn.MSELoss().to(config['DEVICE'])
    cls_loss = cross_entropy2d

    save_dir += datetime.now().strftime("_%m_%d_%H_%M")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_gen = DataGenerator_FCN(global_config['DATA_PATH'], k_fold, 
        batch_size, config['IN_LEN'], config['OUT_LEN'], config['IN_LEN'] + config['OUT_LEN'], config=config)
    writer = SummaryWriter(os.path.join(save_dir, 'train_logs'))

    for k in range(1, k_fold + 1):

        k_model, optimizer, lr_scheduler = model()
        
        data_gen.set_k(k)
        train_loss = 0.0
        train_acc = 0.0
        train_f1 = 0.0
        train_csi = np.zeros((len(global_config['LEVEL_BUCKET']) + 1, ), dtype=np.float32)
        train_count = 0
        i_batch = 0
        best_val_loss = np.inf
        
        pbar = tqdm(range(1, max_iterations + 1))
        for itera in pbar:
            
            n_train_batch = data_gen.n_train_batch()
            pbar_b = tqdm(np.random.choice(data_gen.n_train_batch(), 5000))#range(data_gen.n_train_batch()))
            for b in pbar_b:
                
                pbar.set_description("Fold %d Training at batch %d / %d" % (k, i_batch, n_train_batch))

                train_data, train_label, train_label_cat = data_gen.get_train(b)
                k_model.train()
                optimizer.zero_grad()
                output = k_model(train_data)

                loss = None
                if mode == 'reg':
                    loss = mse_loss(output, train_label)
                elif mode == 'seg':
                    loss = cls_loss(output, train_label_cat)
                elif mode == 'reg_multi':
                    loss = mse_loss(output, train_label)
                    loss += cls_loss(output, train_label_cat)
                else:
                    raise Exception('wrong mode')
                
                loss.backward()
                # torch.nn.utils.clip_grad_value_(k_model.parameters(), clip_value=50.0)
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()

                pred_numpy = np.argmax(output.cpu().detach().numpy(), axis=1).flatten()
                label_numpy = train_label_cat.cpu().numpy().flatten()
                
                train_acc += accuracy_score(label_numpy, pred_numpy)
                train_f1 += f1_score(label_numpy, pred_numpy, average='macro', zero_division=1)
                train_csi += fp_fn_image_csi_muti(dbz_mm(output.cpu().detach().numpy()), dbz_mm(train_label.cpu().numpy()))
                train_count += 1

                if i_batch % eval_every == 0:

                    val_loss = 0.0
                    val_acc = 0.0
                    val_f1 = 0.0
                    val_csi = np.zeros((len(global_config['LEVEL_BUCKET']) + 1, ), dtype=np.float32)
                    val_count = 0

                    with torch.no_grad():
                        k_model.eval()
                        n_val_batch = data_gen.n_val_batch()

                        for ib_val, b_val in enumerate(np.random.choice(n_val_batch, 20)): #range(n_val_batch)
                            val_data, val_label, val_label_cat = data_gen.get_val(b_val)
                            output = k_model(val_data)

                            loss = None
                            if mode == 'reg':
                                loss = mse_loss(output, val_label)
                            elif mode == 'seg':
                                loss = cls_loss(output, val_label_cat)
                            elif mode == 'reg_multi':
                                loss = mse_loss(output, val_label)
                                loss += cls_loss(output, val_label_cat)

                            val_loss += loss.item()

                            pred_numpy = np.argmax(output.cpu().detach().numpy(), axis=1).flatten()
                            label_numpy = val_label_cat.cpu().numpy().flatten()
                            
                            val_acc += accuracy_score(label_numpy, pred_numpy)
                            val_f1 += f1_score(label_numpy, pred_numpy, average='macro', zero_division=1)
                            val_csi += fp_fn_image_csi_muti(dbz_mm(output.cpu().detach().numpy()), dbz_mm(val_label.cpu().numpy()))
                            val_count += 1
                            pbar.set_description("Fold %d Validating at batch %d / %d" % (k, ib_val, 20))

                    train_loss /= train_count
                    train_f1 /= train_count
                    train_acc /= train_count
                    train_csi /= train_count
                    val_loss /= val_count
                    val_f1 /= val_count
                    val_acc /= val_count
                    val_csi /= val_count

                    writer.add_scalars('loss/'+str(k), {
                        'train': train_loss,
                        'valid': val_loss
                    }, i_batch)

                    writer.add_scalars('f1/'+str(k), {
                        'train': train_f1,
                        'valid': val_f1
                    }, i_batch)

                    writer.add_scalars('acc/'+str(k), {
                        'train': train_acc,
                        'valid': val_acc
                    }, i_batch)

                    for i in range(train_csi.shape[0]):
                        writer.add_scalars('csi_'+str(i)+'/'+str(k), {
                            'train': train_csi[i],
                            'valid': val_csi[i]
                        }, i_batch)

                    train_loss = 0.0
                    train_acc = 0.0
                    train_f1 = 0.0
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