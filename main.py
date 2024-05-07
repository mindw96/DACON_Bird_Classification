import wandb
import pandas as pd
import pytorch_lightning as L
from sklearn.model_selection import train_test_split
from utils import data_setting, finetune_large, finetune_small, kfold_inference, test_data_setting


def main(CFG):
    wandb.login()
    L.seed_everything(CFG['SEED'])
    df = pd.read_csv('./train.csv')

    df, le, train_transform, test_transform = data_setting(df, CFG)

    finetune_large(df, CFG, train_transform, test_transform)
    finetune_small(df, CFG, train_transform, test_transform)

    test_df = pd.read_csv('./test.csv')
    test_loader = test_data_setting(test_df, CFG, test_transform)

    kfold_inference(le, test_loader)


if __name__ == "__main__":
    CFG = {
        'IMG_SIZE': 256,
        'EPOCHS': 5,
        'LEARNING_RATE': 3e-4,
        'BATCH_SIZE': 24,
        'SEED': 9608
    }

    main(CFG)
