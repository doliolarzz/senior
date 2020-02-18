import os
from tqdm import tqdm

device = 'cuda:1'
in_len = [5]
out_len = [1, 18]
modes = ['seg']
# modes = ['reg', 'seg', 'reg_multi']
batch_size = 4

# with tqdm(total=len(in_len)*len(out_len)*len(multitask)) as pbar:
for il in in_len:
    for ol in out_len:
        for mt in modes:
            cmdStr = 'python3 fcn.py --name={} --device={} --in={} --out={} --batchsize={} --mode={}'.format(
                '_'.join([str(il), str(ol), str(batch_size), str(mt)]),
                device,
                il,
                ol,
                batch_size,
                mt
            )
            # pbar.set_description('Executing:' + cmdStr)
            os.system(cmdStr)
            # pbar.update(1)
            break
        break
    break
