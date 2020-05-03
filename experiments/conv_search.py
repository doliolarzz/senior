import os
from tqdm import tqdm

device = 'cuda:0'
in_len = [4]
out_len = [10]
multitask = [False]
batch_size = 2

# with tqdm(total=len(in_len)*len(out_len)*len(multitask)) as pbar:
for il in in_len:
    for ol in out_len:
        for mt in multitask:
            cmdStr = 'python3 conv.py --name={} --device={} --in={} --out={} --batchsize={} --multitask={}'.format(
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

