import gc
import glob
import logging
import os
import random
import warnings
from collections import Iterable
from enum import Enum
from typing import List, Union
import numpy as np
from astropy.visualization import AsinhStretch
from dateutil.parser import parse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from Editor import Editor, MapToDataEditor, NanEditor, NormalizeEditor, \
    ExpandDimsEditor, DistributeEditor, LoadGregorGBandLevel1Editor, LoadGregorGBandLevel2Editor, \
    LoadGregorContinuumLevel1Editor, LoadGregorContinuumLevel2Editor, \
    gregor_norms_gband, BrightestPixelPatchEditor, gregor_norms_continuum, \
    RandomPatchEditor, WhiteningEditor, StackEditor, GregorGBandPrepEditor, \
    GregorContinuumPrepEditor, LoadDotLowEditor, ReshapeEditor, PaddingEditor, \
    dot_norms, LoadDotHighEditor, dot_norms_lvl2, DarkestPixelPatchEditor, \
    BlockReduceEditor, ContrastNormalizeEditor, NormalizeExposureEditor, \
    ImageNormalizeEditor, MinMaxNormalizeEditor, LoadGregorGBandLevel1Editor_1, \
    MinMaxQuantileNormalizeEditor, LoadMapEditor, ScaleEditor, hinode_norms, \
    LoadGregorContinuumLevel1Editor_1, LoadDKISTEditor, LoadGregorGBandLevel1Editor_50, \
    LoadGregorGBandLevel1Editor_10, ShiftMeanEditor, StretchPixelEditor, ReadSimulationEditor, \
    GregorPaddingEditor, SimulationPaddingEditor, LoadGregorContinuumLevel1Editor_new, LoadGregorGBandLevel1Editor_new, \
    LoadGregorGBandLevel1Editor_5, LoadGregorGBandLevel1Editor_25, LimbEditor, LoadGregorGBandLevel1Editor_75, \
    LoadFITSEditor, RemoveBackgroundEditor, CropSimulationEditor, LoadGregorContinuumLevel1Editor_5, \
    LoadGregorContinuumLevel1Editor_10, LoadGregorContinuumLevel1Editor_25, LoadGregorContinuumLevel1Editor_50, \
    LoadGregorContinuumLevel1Editor_75, DataToMapEditor


class BaseDataset(Dataset):
    def __init__(self, data: Union[str, list], editors: List[Editor], ext: str = None, limit: int = None,
                 months: list = None, date_parser=None, **kwargs):
        if isinstance(data, str):
            pattern = '*' if ext is None else '*' + ext
            data = sorted(glob.glob(os.path.join(data, '**', pattern), recursive=True))
        assert isinstance(data, Iterable), 'Dataset requires list of samples or path of files!'
        if months: #Assuming filename is parsable datetime
            if date_parser is None:
                date_parser = lambda f: parse(os.path.basename(f).split('_')[1])
            data = [d for d in data if date_parser(d).month in months]

        if limit is not None:
            data = random.sample(list(data), limit)
        self.data = data
        self.editors = editors

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, _ = self.getIndex(idx)
        return data

    def sample(self, n_samples):
        it = DataLoader(self, batch_size=1, shuffle=True, num_workers=4).__iter__()
        samples = []
        while len(samples) < n_samples:
            try:
                samples.append(next(it).detach().numpy()[0])
            except Exception as ex:
                logging.error(str(ex))
                continue
        del it
        return np.array(samples)

    def getIndex(self, idx):
        try:
            return self.convertData(self.data[idx])
        except Exception as ex:
            logging.error('Unable to convert %s: %s' % (self.data[idx], ex))
            raise ex

    def getId(self, idx):
       #return os.path.basename(self.data[idx]).split('.')[1]
        return os.path.basename(self.data[idx])

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data, kwargs

    def addEditor(self, editor):
        self.editors.append(editor)



class StorageDataset(Dataset):
    def __init__(self, dataset: BaseDataset, store_dir, ext_editors=[]):
        self.dataset = dataset
        self.store_dir = store_dir
        self.ext_editors = ext_editors
        os.makedirs(store_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset.getId(idx)
        store_path = os.path.join(self.store_dir, '%s.npy' % id)
        if os.path.exists(store_path):
            data = np.load(store_path, mmap_mode='r+')
            data = self.convertData(data)
            return data
        data = self.dataset[idx]
        np.save(store_path, data)
        data = self.convertData(data)
        return data

    def convertData(self, data):
        kwargs = {}
        for editor in self.ext_editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data

    def sample(self, n_samples):
        it = DataLoader(self, batch_size=1, shuffle=True, num_workers=4).__iter__()
        samples = []
        while len(samples) < n_samples:
            try:
                samples.append(next(it).detach().numpy())
            except Exception as ex:
                logging.error(str(ex))
                continue
        del it
        return np.concatenate(samples)

    def convert(self, n_worker):
        it = DataLoader(self, batch_size=1, shuffle=False, num_workers=n_worker).__iter__()
        for i in tqdm(range(len(self.dataset))):
            try:
                next(it)
                gc.collect()
            except StopIteration:
                return
            except Exception as ex:
                logging.error('Invalid data: %s' % self.dataset.data[i])
                logging.error(str(ex))
                continue


def get_intersecting_files(path, dirs, months=None, years=None, n_samples=None, ext=None, basenames=None, **kwargs):
    pattern = '*' if ext is None else '*' + ext
    if basenames is None:
        basenames = [[os.path.basename(path) for path in glob.glob(os.path.join(path, str(d), '**', pattern), recursive=True)] for d in dirs]
        basenames = list(set(basenames[0]).intersection(*basenames))
    if months:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('_')[1]).month in months]
    if years:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('_')[1]).year in years]
    basenames = sorted(list(basenames))
    if n_samples:
        basenames = basenames[::len(basenames) // n_samples]
    return [[os.path.join(path, str(dir), b) for b in basenames] for dir in dirs]



class StackDataset(BaseDataset):

    def __init__(self, data_sets, limit=None, **kwargs):
        self.data_sets = data_sets

        editors = [StackEditor(data_sets)]
        super().__init__(list(range(len(data_sets[0]))), editors, limit=limit)

    def getId(self, idx):
        return os.path.basename(self.data_sets[0].data[idx]).split('.')[0]




class DKISTDataset(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.0253, patch_shape=None, ext='.fits', **kwargs):
        editors = [LoadDKISTEditor(),
                   ScaleEditor(scale),
                   MapToDataEditor(),
                   #ContrastNormalizeEditor(),
                   MinMaxQuantileNormalizeEditor(),
                   NanEditor(),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))



class DotDatasetLow(BaseDataset):

    def __init__(self, data: Union[str, list], resolution=512, patch_shape=None, ext='.fits', **kwargs):
        norm = dot_norms['gband']
        editors = [LoadDotLowEditor(),
                   MapToDataEditor(),
                   WhiteningEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ReshapeEditor((resolution, resolution)),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class DotDatasetHigh(BaseDataset):

    def __init__(self, data: Union[str, list], resolution=512, patch_shape=None, ext='.fits', **kwargs):
        #norm = dot_norms_lvl2['gband']
        editors = [LoadDotHighEditor(),
                   MapToDataEditor(),
                   #WhiteningEditor(),
                   NanEditor(),
                   MinMaxNormalizeEditor(),
                   #NormalizeEditor(norm),
                   #ReshapeEditor((resolution, resolution)),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))



class GregorDatasetGBandLevel1(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce number of pixels by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(DarkestPixelPatchEditor(patch_shape))

class GregorDatasetGBandLevel1_1(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor_1(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetGBandLevel1_5(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor_5(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetGBandLevel1_10(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor_10(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetGBandLevel1_25(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor_25(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetGBandLevel1_50(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor_50(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetGBandLevel1_75(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel1Editor_75(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetGBandLevel2(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)),
                       NanEditor(),
                       MinMaxQuantileNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorGBandLevel2Editor(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(DarkestPixelPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_continuum['continuum']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((2, 2)),
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_1(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_continuum['continuum']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)),
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_1(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_5(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_5(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_10(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_10(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_25(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_25(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_50(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_50(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_75(BaseDataset):

    def __init__(self, data: Union[str, list], scale = 0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_gband['gband']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)), # Reduce pixel by 8 (2650, 2160) --> (320, 270)
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_75(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel1_new(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_continuum['continuum']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       #BlockReduceEditor((2, 2)),
                       NanEditor(),
                       MinMaxNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel1Editor_new(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class GregorDatasetContinuumLevel2(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, ext='.fts', **kwargs):
        norm = gregor_norms_continuum['continuum']
        sub_editors = [ScaleEditor(scale),
                       MapToDataEditor(),
                       GregorPaddingEditor(),
                       #BlockReduceEditor((8, 8)),
                       NanEditor(),
                       MinMaxQuantileNormalizeEditor(),
                       ContrastNormalizeEditor(),
                       ExpandDimsEditor()]
        editors = [LoadGregorContinuumLevel2Editor(), DistributeEditor(sub_editors)]

        super().__init__(data, editors, ext=ext, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))


class HinodeDataset(BaseDataset):

    def __init__(self, data, scale=0.0253, wavelength='continuum', **kwargs):
        norm = hinode_norms[wavelength]

        editors = [LoadMapEditor(),
                   #ScaleEditor(scale),
                   NormalizeExposureEditor(),
                   MapToDataEditor(),
                   NanEditor(),
                   NormalizeEditor(norm),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, **kwargs)


class SimulationDataset(BaseDataset):

    def __init__(self, data: Union[str, list], scale=0.02761, patch_shape=None, **kwargs):

        editors = [ReadSimulationEditor(),
                   ScaleEditor(scale),
                   MapToDataEditor(),
                   #SimulationPaddingEditor(),
                   NanEditor(),
                   MinMaxNormalizeEditor(),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))




class LimbSimulationDataset(BaseDataset):

    def __init__(self, data: Union[str, list], patch_shape=None, **kwargs):

        editors = [LoadFITSEditor(),
                   StretchPixelEditor(),
                   ExpandDimsEditor()]
        super().__init__(data, editors=editors, **kwargs)
        if patch_shape is not None:
            self.addEditor(RandomPatchEditor(patch_shape))
