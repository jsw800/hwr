from __future__ import print_function
from builtins import range

import json
import sys
import hw_dataset
from hw_dataset import HwDataset
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import datasets
import cnn as model
import numpy as np

def main():
    # load config
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    # prepare training datasets and test datasets
    train_dataset, test_dataset = datasets.get_training_and_validation_datasets(config['image_root_directory'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=hw_dataset.collate)
    
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=hw_dataset.collate)

    #hw = Net()
    # initialize an untrained network
    hw = model.create_model(config)

    # check if we can use the GPU
    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")

    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(hw.parameters(), lr=config['network']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    lowest_loss = 0.0
    # 1000 epochs is way too many, just Ctrl+C when it seems to be trained
    for epoch in range(1000):
        # start epoch
        sum_loss = 0.0
        steps = 0.0
        # put the net in training mode
        hw.train()
        # for each BATCH in the training set
        for i, x in enumerate(train_dataloader):
            # Get the batch of training imgs
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            # prepare it for the net
            gts = Variable(torch.from_numpy(np.array(x['gt'])))

            # run it through the net
            preds = hw(line_imgs).cpu()

            # How bad did we do?
            loss = criterion(preds, gts)

            optimizer.zero_grad()
            # Based on how badly we did, adjust the weights of the net
            loss.backward()
            optimizer.step()
            prs = preds.data.numpy()
            prs = np.argmax(prs, axis=1)
            # Accumulate error count in sum_loss
            out = [1 for num in range(len(prs)) if prs[num] == gts.data.numpy()[num]]
            sum_loss += sum(out)
        
        # print training accuracy for the epoch
        print("Training accuracy: %.2f" % (sum_loss * 100 / (len(train_dataloader) * 8)) + "%")
        # zero accumulators
        sum_loss = 0.0
        steps = 0.0
        hw.eval()
        # for each BATCH in test dataset
        for x in test_dataloader:
            # get batch of test imgs, prepare for net
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
            gts = Variable(torch.from_numpy(np.array(x['gt'])))

            # send them through the net
            preds = hw(line_imgs).cpu()

            #output_batch = preds.permute(1,0,2)
            out = preds.data.cpu().numpy()

            # get error rates on the batch, add to sum_loss
            for i, gt_line in enumerate(x['gt']):
                logits = out[i,...]
                max = np.argmax(logits)
                sum_loss += 1 if max == gt_line else 0
                steps += 1

        # log test accuracy for the epoch
        print("Test accuracy: %.2f" % (sum_loss * 100 / steps) + '%')

        # if this epoch is the best so far, save the net weights to the weight file.
        if lowest_loss < sum_loss/steps:
            lowest_loss = sum_loss/steps
            print("Saving Best")
            dirname = os.path.dirname(config['model_save_path'])
            if len(dirname) > 0 and not os.path.exists(dirname):
                os.makedirs(dirname)

            torch.save(hw.state_dict(), os.path.join(config['model_save_path']))
    # end epoch

if __name__ == "__main__":
    main()
