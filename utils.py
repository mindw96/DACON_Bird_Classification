from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import timm
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import numpy as np
import albumentations as A
from sklearn import preprocessing
from lightning.pytorch import Trainer
from sklearn.model_selection import train_test_split, StratifiedKFold
from albumentations.pytorch.transforms import ToTensorV2
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from models import LitModel


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        image = cv2.imread(img_path)

        if self.label_list is not None:
            label = self.label_list[index]
            return {'image': image, 'label': label}
        else:
            return {'image': image}

    def __len__(self):
        return len(self.img_path_list)


class CustomCollateFn:
    def __init__(self, transform, mode):
        self.mode = mode
        self.transform = transform

    def __call__(self, batch):
        if self.mode == 'train':
            pixel_values = torch.stack([self.transform(image=data['image'])['image'] for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values': pixel_values,
                'label': label,
            }
        elif self.mode == 'val':
            pixel_values = torch.stack([self.transform(image=data['image'])['image'] for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values': pixel_values,
                'label': label,
            }
        elif self.mode == 'inference':
            pixel_values = torch.stack([self.transform(image=data['image'])['image'] for data in batch])
            return {
                'pixel_values': pixel_values,
            }


class RandomAugMix(ImageOnlyTransform):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )

        return image


def int_parameter(level, maxval):
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def normalize(image):
    return image - 127


def apply_op(image, op, severity):
    pil_img = Image.fromarray(image)
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)

    augmentations = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y
    ]

    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)

        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix

    return mixed.astype(np.float64)


def data_setting(df, CFG):
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    train_transform = A.Compose([
        RandomAugMix(severity=3, width=3, alpha=1., p=1.),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    tset_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    train_collate_fn = CustomCollateFn(train_transform, 'train')
    val_collate_fn = CustomCollateFn(tset_transform, 'val')

    return df, le, train_collate_fn, val_collate_fn


def test_data_setting(df, CFG, transform):
    test_dataset = CustomDataset(df['img_path'].values, None, transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    return test_loader


def finetune_large(df, CFG, train_fn, test_fn):
    skf = StratifiedKFold(n_splits=5, random_state=CFG['SEED'], shuffle=True)

    for fold_idx, (train_index, val_index) in enumerate(skf.split(df, df['label'])):
        train = df.loc[train_index, :]
        val = df.loc[val_index, :]

        train_dataset = CustomDataset(train['upscale_img_path'].values, train['label'].values)
        train_loader = DataLoader(train_dataset, collate_fn=train_fn, batch_size=CFG['BATCH_SIZE'], shuffle=True,
                                  num_workers=0)

        val_dataset = CustomDataset(val['upscale_img_path'].values, val['label'].values)
        val_loader = DataLoader(val_dataset, collate_fn=test_fn, batch_size=CFG['BATCH_SIZE'], shuffle=False,
                                num_workers=0)

        run = wandb.init(
            project="Bird-Classification",
            notes="SwinV2-large",
            tags=["SwinV2-large"],
        )
        wandb_logger = WandbLogger()

        model = timm.create_model('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', pretrained=True,
                                  num_classes=25)
        litmodel = LitModel(model)

        early_stop_callback = EarlyStopping("val_f1", patience=5, verbose=True, mode="max")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            dirpath='./checkpoints/',
            filename=f'SwinV2_large_augmix-fold_idx={fold_idx}.ckpt',
            save_top_k=1,
            save_weights_only=True,
            verbose=True
        )
        trainer = Trainer(max_epochs=100, precision=32,
                          callbacks=[early_stop_callback, checkpoint_callback],
                          logger=wandb_logger)
        trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()


def finetune_small(df, CFG, train_transform, test_transform):
    skf = StratifiedKFold(n_splits=5, random_state=CFG['SEED'], shuffle=True)

    for fold_idx, (train_index, val_index) in enumerate(skf.split(df, df['label'])):
        train = df.loc[train_index, :]
        val = df.loc[val_index, :]

        train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

        val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

        run = wandb.init(
            project="Bird-Classification",
            notes="SwinV2-large",
            tags=["SwinV2-large"],
        )
        wandb_logger = WandbLogger()

        model = timm.create_model('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', pretrained=True,
                                  num_classes=25)
        checkpoint_path = f'SwinV2_large_augmix-fold_idx={fold_idx}.ckpt'
        litmodel = LitModel.load_from_checkpoint(checkpoint_path, model=model)

        early_stop_callback = EarlyStopping("val_f1", patience=5, verbose=True, mode="max")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            dirpath='./checkpoints/',
            filename=f'SwinV2_large_augmix_finetune_idx={fold_idx}.ckpt',
            save_top_k=1,
            save_weights_only=True,
            verbose=True
        )
        trainer = Trainer(max_epochs=100, precision=32,
                          callbacks=[early_stop_callback, checkpoint_callback],
                          logger=wandb_logger,
                          default_root_dir="./checkpoints/")
        trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()


def kfold_inference(le, test_loader):
    fold_preds = []
    for checkpoint_path in glob.glob('./checkpoints/SwinV2_large_augmix_finetune_idx*.ckpt'):
        model = timm.create_model('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', pretrained=True,
                                  num_classes=25)
        lit_model = LitModel.load_from_checkpoint(checkpoint_path, model=model)
        trainer = Trainer(accelerator='auto', precision=32)
        preds = trainer.predict(lit_model, test_loader)
        preds = torch.cat(preds, dim=0).detach().cpu().numpy().argmax(1)
        fold_preds.append(preds)
    pred_ensemble = list(map(lambda x: np.bincount(x).argmax(), np.stack(fold_preds, axis=1)))

    submission = pd.read_csv('./sample_submission.csv')
    submission['label'] = le.inverse_transform(pred_ensemble)
    submission.to_csv('./SwinV2-large-augmix-finetune-5fold.csv', index=False)
