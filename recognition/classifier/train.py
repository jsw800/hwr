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
import cnn as model
import numpy as np

def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)
#Replace next line with initializing the dataset object for both training and validation. Create two HWDataset class objects ...
    #Beginning edits of this code.
    train_dataset, test_dataset = datasets.get_training_and_validation_datasets(config['image_root_directory'])
    train_dataset = HwDataset(config['image_root_directory'])#This doesn't work yet as is--I think it needs to be passed something else
    test_dataset = HwDataset(config)#


    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=hw_dataset.collate)
    
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=hw_dataset.collate)

    hw = model.create_model(config)

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")

    optimizer = torch.optim.Adam(hw.parameters(), lr=config['network']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    lowest_loss = 0.0
    for epoch in range(1000):
        sum_loss = 0.0
        steps = 0.0
        hw.train()
        for i, x in enumerate(train_dataloader):
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            gts = Variable(torch.from_numpy(np.array(x['gt'])))

            preds = hw(line_imgs).cpu()

            loss = criterion(preds, gts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prs = preds.data.numpy()
            prs = np.argmax(prs, axis=1)
            out = [1 for num in range(len(prs)) if prs[num] == gts.data.numpy()[num]]
            sum_loss += sum(out)

        print("Training accuracy: %.2f" % (sum_loss * 100 / (len(train_dataloader) * 8)) + "%")
        sum_loss = 0.0
        steps = 0.0
        hw.eval()
        for x in test_dataloader:
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
            gts = Variable(torch.from_numpy(np.array(x['gt'])))

            preds = hw(line_imgs).cpu()

            #output_batch = preds.permute(1,0,2)
            out = preds.data.cpu().numpy()

            for i, gt_line in enumerate(x['gt']):
                logits = out[i,...]
                max = np.argmax(logits)
                sum_loss += 1 if max == gt_line else 0
                steps += 1

        print("Test accuracy: %.2f" % (sum_loss * 100 / steps) + '%')

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
