from __future__ import print_function
# from builtins import range

import json
import character_set
import sys
import hw_dataset
from hw_dataset import HwDataset
import crnn
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import error_rates
import string_utils

def main():
    config_path = sys.argv[1]
    cont = False
    if len(sys.argv) > 2:
        cont = True

    with open(config_path) as f:
        config = json.load(f)

    idx_to_char, char_to_idx = character_set.load_char_set(config['character_set_path'])

    def train_epoch():

        train_dataset = HwDataset(config['training_set_path'], char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config['image_root_directory'], augmentation=True,
            categories=config['categories'], maxwidths=config['maxwidths'],
            wide=config['wide'], writeout=config['error_log'])
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
            num_workers=2, collate_fn=hw_dataset.collate)

        sum_loss = 0.0
        steps = 0.0
        hw.train()
        for i, x in enumerate(train_dataloader):
            if x is None:
                continue
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels =  Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

            # if line_imgs.size()[-1] > 400:
            #     print(line_imgs.size())
            #     print(x['idx'])

            try:
                preds = hw(line_imgs).cpu()
            except Exception as e:
                print('error on {}'.format(x))
                continue

            preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))

            output_batch = preds.permute(1,0,2)
            out = output_batch.data.cpu().numpy()

            loss = criterion(preds, labels, preds_size, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i == 0:
            #    for i in xrange(out.shape[0]):
            #        pred, pred_raw = string_utils.naive_decode(out[i,...])
            #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
            #        print(pred_str)

            for j in range(out.shape[0]):
                logits = out[j,...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str(pred, idx_to_char, False)
                gt_str = x['gt'][j]
                cer = error_rates.cer(gt_str, pred_str)
                sum_loss += cer
                steps += 1

            # if i % 1000 == 0:
                # torch.save(hw.state_dict(), os.path.join(config['model_save_path']))
                # with open(config['error_rates'], 'a') as f:
                #     f.write(str(sum_loss / steps) + ',')
                # print("numeration: {}".format(i))
                # print("Training CER", sum_loss / steps)

        return sum_loss / steps

    def val_epoch():

        test_dataset = HwDataset(config['validation_set_path'], char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config['image_root_directory'], augmentation=True,
            categories=config['categories'], maxwidths=config['maxwidths'],
            wide=config['wide'], writeout=config['error_log'])
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
            num_workers=0, collate_fn=hw_dataset.collate)

        sum_loss = 0.0
        steps = 0.0
        hw.eval()
        for x in test_dataloader:
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
            labels =  Variable(x['labels'], requires_grad=False, volatile=True)
            label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)

            try:
                preds = hw(line_imgs).cpu()
            except Exception as e:
                print('error on {}'.format(x))
                continue

            output_batch = preds.permute(1,0,2)
            out = output_batch.data.cpu().numpy()

            for i, gt_line in enumerate(x['gt']):
                logits = out[i,...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str(pred, idx_to_char, False)
                cer = error_rates.cer(gt_line, pred_str)
                sum_loss += cer
                steps += 1

        print("Test CER", sum_loss / steps)

        return sum_loss / steps


    hw = crnn.create_model({
        'cnn_out_size': config['network']['cnn_out_size'],
        'num_of_channels': 3,
        'num_of_outputs': len(idx_to_char)+1
    })

    if cont:
        hw.load_state_dict(torch.load(config['model_save_path']))
    else:
        with open(config['error_rates'], 'w') as f:
            f.write('')
        with open(config['error_log'], 'w') as f:
            f.write('')

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")


    optimizer = torch.optim.Adam(hw.parameters(), lr=config['network']['learning_rate'])
    criterion = CTCLoss()
    lowest_loss = float('inf')
    for epoch in range(1000):

        loss = train_epoch()
        with open(config['error_rates'], 'a') as f:
            f.write(str(loss) + ',')

        loss = val_epoch()
        with open(config['error_rates'], 'a') as f:
            f.write(str(loss) + '\n')

        if lowest_loss > loss:
            lowest_loss = loss
            print("Saving Best")
            dirname = os.path.dirname(config['model_save_path'])
            if len(dirname) > 0 and not os.path.exists(dirname):
                os.makedirs(dirname)

            torch.save(hw.state_dict(), os.path.join(config['model_save_path']))


if __name__ == "__main__":
    main()
