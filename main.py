import pandas as pd
import torch
import timm
import numpy as np
import os
import pickle
from pathlib import Path
import traceback
from ast import literal_eval
from ribophene.file_io import read_metadata, get_training_data, cache_data
from ribophene.trainer import Trainer

# device
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# assign paths/folders
package_root = Path(__file__).resolve().parents[0]
data_dir = package_root / "data"
SAVE_DIR = data_dir
MODEL_FOLDER_NAME = "PhenotypeClassification"

# define parameters
image_size = (64,64)
resize = False
ratio_train = 0.9
val_test_split = 0.5
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 10
AUGMENT = True
model_backbone = 'densenet121'
model_backbone = 'efficientnet_b0'


task_dict = {"task1": "metadata_clinical_train_2strain_1X_test_6strain_ribophene.csv",
             "task2": "metadata_clinical_SRmodel_ribophene.csv",
             "task3": "metadata_MG1655_cam_ribophene.csv",
             "task4": "metadata_MG1655_cip_ribophene.csv",
             "task5": "metadata_MG1655_gent_ribophene.csv"}

task_name = "task1"
metadata, class_labels, channel_list = read_metadata(task_dict[task_name], data_dir)

if __name__ == '__main__':

    cached_data = cache_data(
        metadata,
        class_labels,
        image_size,
        import_limit = 'None',
        mask_background=True,
        resize=resize)

    # with open('cacheddata.pickle', 'wb') as handle:
    #     pickle.dump(cached_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('cacheddata.pickle', 'rb') as handle:
    #     cached_data = pickle.load(handle)

    num_classes = len(np.unique(cached_data["labels"]))

    print(f"num_classes: {num_classes}, num_images: {len(cached_data['images'])}")

    train_data, val_data, test_data = get_training_data(cached_data,
                                                          shuffle=True,
                                                          ratio_train = 0.8,
                                                          val_test_split=0.5,
                                                          label_limit = 'None',
                                                          balance = True,)

    model = timm.create_model(model_backbone, pretrained=False, num_classes=num_classes).to(device)
    # 'timm.list_models()' to list available models

    trainer = Trainer(model=model,
                      task_name = task_name,
                      num_classes=num_classes,
                      augmentation=AUGMENT,
                      device=device,
                      learning_rate=LEARNING_RATE,
                      train_data=train_data,
                      val_data=val_data,
                      test_data=test_data,
                      tensorboard=True,
                      class_labels = class_labels,
                      channel_list = channel_list,
                      epochs=EPOCHS,
                      batch_size = BATCH_SIZE,
                      save_dir = SAVE_DIR,
                      model_folder_name = MODEL_FOLDER_NAME)

    # trainer.visualise_augmentations(n_examples=10, show_plots=True, save_plots=True)
    # trainer.tune_hyperparameters(num_trials=10, num_images = 10, num_epochs = 2)
    model_path = trainer.train()
    trainer.evaluate(model_path)
