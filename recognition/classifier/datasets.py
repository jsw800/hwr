import os
import hw_dataset
from os.path import join
import random
import json

PERCENTAGE_TRAIN = .75

# NOTE: This code is a bit hacky and gross. I'm imposing some weird things here for the purpose of
# having good training data. I would recommend refactoring it and making it pretty. it evolved
# over time and ended up being gross, but it currently works

def get_training_and_validation_datasets(root_dir):

    # get all possible labels (ignore 'other' directory, it is just junk)
    labels = []
    for root, dirs, files in os.walk(str(root_dir)):
        for dir in dirs:
            if dir != 'other':
                labels.append(dir)
        break
    
    train_sets = {}
    val_sets = {}

    train_set = []
    val_set = []

    for label in labels:
        train_sets[label] = set()
        val_sets[label] = set()
        for root, dirs, files in os.walk(join(str(root_dir), str(label))):
            random.shuffle(files)
            num_train = int(float(len(files)) * float(PERCENTAGE_TRAIN))
            if num_train > 450:
                num_train = 450

            # for each file in this label dir,
            for i in range(len(files)):
                if i < num_train:
                    train_sets[label].add(join(root_dir, label, files[i]))
                else:
                    # don't put more than 600 blank examples in training data, or the net
                    # will ALWAYS guess that we aren't a vet, and that's stupid.
                    if label == "blank" and i > 600:
                        break
                    val_sets[label].add(join(root_dir, label, files[i]))

    # we need these specific images to be in the train set for blanks, because the net has had problems with them.
    # We're leaving this out for now
    #with open('wrong.json') as f:
        #wrong_blanks = json.load(f)

#    num_train = int(len(wrong_blanks) * PERCENTAGE_TRAIN)
#    wrong_train = wrong_blanks[:num_train]
#    wrong_val = wrong_blanks[num_train:]
#    train_sets['blank'].update(wrong_train)
#    val_sets['blank'].update(wrong_val)

    # for each label, make the actual dict objects
    for i, label in enumerate(labels):
        for image in train_sets[label]:
            ob = {
                'im': image,
                'gt': 1 if label == "WW" else 0
            }
            train_set.append(ob)
        for image in val_sets[label]:
            ob = {
                'im': image,
                'gt': 1 if label == "WW" else 0
            }
            val_set.append(ob)

    # Now that we have the filenames of the images and the labels,
    # we can pass those to the Dataset object constructors and we're done
    train = hw_dataset.HwDataset(train_set, augmentation=True)
    val = hw_dataset.HwDataset(val_set)
    return train, val

# Just like the above ones, except we don't split train/validate
def get_full_set(root_dir):
    # get all labels from the label dir
    labels = [dirs for root, dirs, files in os.walk(root_dir)][0]
    # ignore 'other' dir, it is junk
    labels = [label for label in labels if label != 'other']
    data = []
    for label in labels:
        for root, dirs, files in os.walk(join(root_dir, label)):
            random.shuffle(files)
            for file in files:
                file = join(root_dir, label, file)
                # Everything that isn't the "WW" dir is not_WW
                data.append({
                    'im' : file,
                    'gt' : 1 if label == "WW" else 0
                })
    # pass the data to the dataset constructor
    return hw_dataset.HwDataset(data)

if __name__ == '__main__':
    print(get_full_set('war_images'))
