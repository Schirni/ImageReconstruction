import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
import hyper
from Dataset import GregorDatasetGBandLevel1_10, GregorDatasetGBandLevel1_5, GregorDatasetGBandLevel2, StorageDataset, \
    GregorDatasetGBandLevel1, SimulationDataset, GregorDatasetGBandLevel1_50, GregorDatasetGBandLevel1_25, \
    GregorDatasetGBandLevel1_75
from Trainer import Trainer
from Model import DiscriminatorMode
from Editor import RandomPatchEditor, DarkestPixelPatchEditor
import argparse
import logging
import os

parser = argparse.ArgumentParser(description='Train mitigation of atmospheric degradations from GREGOR observations')
parser.add_argument('--base_dir', type=str, help='path to the results directory.')

parser.add_argument('--hq_path', type=str, help='path to the high-quality GREGOR data.')
parser.add_argument('--lq_path', type=str, help='path to the low-quality GREGOR data.')
parser.add_argument('--hq_converted_path', type=str, help='path to store the converted high-quality GREGOR data.')
parser.add_argument('--lq_converted_path', type=str, help='path to store the converted low-quality GREGOR data.')

args = parser.parse_args()
base_dir = args.base_dir
low_path = args.lq_path
low_converted_path = args.lq_converted_path
high_path = args.hq_path
high_converted_path = args.hq_converted_path

log_iteration = 1000
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("{0}/{1}.log".format(base_dir, "info_log")),
        logging.StreamHandler()
    ])

#base_dir = '/gpfs/data/fs71254/schirni/Training/Training2'
#low_path = '/gpfs/data/fs71254/schirni/Level1_Files/Level1_Files_GBand'
#high_path = '/gpfs/data/fs71254/schirni/Level2_Files/Level2_Files_GBand'
#low_converted_path = '/gpfs/data/fs71254/schirni/Lvl1GBand_converter'
#high_converted_path = '/gpfs/data/fs71254/schirni/Lvl2Gband_converter'

trainer = Trainer(input_dim_a=10, input_dim_b=1, norm='in_rs_aff', discriminator_mode=DiscriminatorMode.RANDOM,
                  lambda_diversity=0, shuffle=True)
trainer.cuda()

gregor_train_low_dataset = GregorDatasetGBandLevel1_10(low_path, months=list(range(10)))
gregor_train_low_storage = StorageDataset(gregor_train_low_dataset, low_converted_path,
                                          ext_editors=[RandomPatchEditor((512, 512))])

gregor_train_high_dataset = GregorDatasetGBandLevel2(high_path, months=list(range(9)))
gregor_train_high_storage = StorageDataset(gregor_train_high_dataset, high_converted_path,
                                           ext_editors=[RandomPatchEditor((512, 512))])

gregor_valid_low_dataset = GregorDatasetGBandLevel1_10(low_path, months=[10, 11, 12], limit=50)
gregor_valid_low_storage = StorageDataset(gregor_valid_low_dataset, low_converted_path,
                                          ext_editors=[RandomPatchEditor((1024, 1024))])

gregor_valid_high_dataset = GregorDatasetGBandLevel2(high_path, months=[9, 10, 11, 12], limit=50)
gregor_valid_high_storage = StorageDataset(gregor_valid_high_dataset, high_converted_path,
                                           ext_editors=[RandomPatchEditor((1024, 1024))])

plot_settings_A = {"cmap": "yohkohsxtal", "title": "GREGOR Level1", 'vmin': -1, 'vmax': 1}
plot_settings_B = {"cmap": "yohkohsxtal", "title": "GREGOR Level2", 'vmin': -1, 'vmax': 1}

trainer.startBasicTraining(base_dir, gregor_train_low_storage, gregor_train_high_storage,
                           gregor_valid_low_storage, gregor_valid_high_storage,
                           plot_settings_A, plot_settings_B, num_workers=8)


