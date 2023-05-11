import os
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint
from urllib import request

import astropy.io.ascii
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, AsinhStretch
from dateutil.parser import parse
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.transform import pyramid_reduce
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header, all_coordinates_from_map, header_helper
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from scipy.interpolate import interp2d


class Editor(ABC):

    def convert(self, data, **kwargs):
        result = self.call(data, **kwargs)
        if isinstance(result, tuple):
            data, add_kwargs = result
            kwargs.update(add_kwargs)
        else:
            data = result
        return data, kwargs

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()

gregor_norms_gband = {'gband': ImageNormalize(stretch=LinearStretch(), clip=True)}
gregor_norms_continuum = {'continuum': ImageNormalize(stretch=LinearStretch(), clip=True)}
dot_norms = {'gband': ImageNormalize(vmin=-1, vmax=1, stretch=LinearStretch(), clip=True)}
dot_norms_lvl2 = {'gband': ImageNormalize(vmin=-3, vmax=3, stretch=LinearStretch(), clip=True)}
hinode_norms = {'continuum': ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True),
                'gband': ImageNormalize(vmin=0, vmax=25000, stretch=LinearStretch(), clip=True), }


class MapToDataEditor(Editor):
    def call(self, s_map, **kwargs):
        return s_map.data, {"header": s_map.meta}



class BlockReduceEditor(Editor):

    def __init__(self, block_size, func=np.mean):
        self.block_size = block_size
        self.func = func

    def call(self, data, **kwargs):
        return block_reduce(data, self.block_size, func=self.func)


class NanEditor(Editor):
    def __init__(self, nan=0):
        self.nan = nan

    def call(self, data, **kwargs):
        data = np.nan_to_num(data, nan=self.nan)
        return data



class NormalizeEditor(Editor):
    def __init__(self, norm, **kwargs):
        self.norm = norm

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class ImageNormalizeEditor(Editor):

    def __init__(self, vmin=None, vmax=None, stretch=LinearStretch()):
        self.norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class MinMaxQuantileNormalizeEditor(Editor):
    def call(self, data, **kwargs):

        vmin = np.quantile(data, 0.001)
        vmax = np.quantile(data, 0.999)
        #vmax = np.max(data)

        data = (data - vmin) / (vmax - vmin) * 2 - 1
        data = np.clip(data, -1, 1)
        return data


class MinMaxNormalizeEditor(Editor):
    def call(self, data, **kwargs):

        vmin = np.min(data)
        vmax = np.max(data)

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        data = norm(data).data * 2 - 1
        return data


class StretchPixelEditor(Editor):
    def call(self, data, **kwargs):

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        data = (data - vmin) / (vmax - vmin) * 2 - 1
        return data


class WhiteningEditor(Editor):
    """ Mean value is set to 0 (remove contrast) and std is set to 1"""

    def call(self, data, **kwargs):
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        data = (data - data_mean) / (data_std + 1e-6)
        return data



class ExpandDimsEditor(Editor):
    def __init__(self, axis=0):
        self.axis = axis

    def call(self, data, **kwargs):
        return np.expand_dims(data, axis=self.axis).astype(np.float32)



class DistributeEditor(Editor):
    def __init__(self, editors):
        self.editors = editors

    def call(self, data, **kwargs):
        return np.concatenate([self.convertData(d, **kwargs) for d in data], 0)

    def convertData(self, data, **kwargs):
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data


class BrightestPixelPatchEditor(Editor):
    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == np.nanmax(smoothed))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

            x = pixel_pos[0]
            y = pixel_pos[1]
            patch = data[:,
                    x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                    y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class DarkestPixelPatchEditor(Editor):
    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == (np.nanmin(smoothed)))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

            x = pixel_pos[0]
            y = pixel_pos[1]
            patch = data[:,
                    x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                    y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class LoadFITSEditor(Editor):

    def call(self, map_path, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            hdul = fits.open(map_path)
            hdul.verify("fix")
            data, header = hdul[0].data, hdul[0].header
            hdul.close()
        return data, {"header": header}


class LoadSimulationFITSEditor(Editor):

    def call(self, map_path, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            hdul = fits.open(map_path)
            hdul.verify("fix")
            data, header = hdul[0].data, hdul[0].header
            primary_header = hdul[0].header

            hdul.close()
        return data, {"header": header}


class DataToMapEditor(Editor):

    def call(self, data, **kwargs):
        return Map(data[0], kwargs['header'])


class LoadDKISTEditor(Editor):

    def call(self, file, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            hdul = fits.open(file)
            data, header = hdul[1].data, hdul[1].header
            data = data[48:4048, 48:4048] #remove padding
            smap = Map(data, header)

            return smap, {'path': file}



class LoadDotLowEditor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        primary_header = hdul[0].header
        primary_header["WAVELNTH"] = primary_header["LAMBDA"]
        primary_header["CUNIT1"] = 'arcsec'
        primary_header["CUNIT2"] = 'arcsec'
        primary_header["CDELT1"] = primary_header["ARCS_PP"]
        primary_header["CDELT2"] = primary_header["ARCS_PP"]
        primary_header["INSTRUME"] = "DOT"
        #
        data = hdul[0].data
        data_low = data[2] # select the temporal average of the whole speckle burst
        #
        dot_maps = Map(data_low, primary_header)

        return dot_maps, {'path': file}


class LoadDotHighEditor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        primary_header = hdul[0].header
        primary_header["WAVELNTH"] = primary_header["LAMBDA"]
        primary_header["CUNIT1"] = 'arcsec'
        primary_header["CUNIT2"] = 'arcsec'
        primary_header["CDELT1"] = primary_header["ARCS_PP"]
        primary_header["CDELT2"] = primary_header["ARCS_PP"]
        primary_header["INSTRUME"] = "DOT"
        #
        data = hdul[0].data
        data_high = data[0] # select speckled image
        #
        dot_maps = Map(data_high, primary_header)

        return dot_maps, {'path': file}


class LoadGregorGBandLevel1Editor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header or 'FILTNOTE' in hdul[1].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['DATE'][0:4] == '2022':
            index = 2
        elif hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        g_band = hdul[index::2]
        #g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_1(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        g_band = hdul[index]
        #g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(g_band.data, primary_header)]
        #gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_5(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        g_band = hdul[index::40]
        g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        #gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_new(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        #assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[1].header['FILTNOTE'] == 'G-band':
            index = 1
        elif hdul[2].header['FILTNOTE'] == 'G-band':
            index = 2
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #primary_header['cdelt1'] = 0.0253
        #primary_header['cdelt2'] = 0.0253
        #
        g_band = hdul[index::2]
        #g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_10(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[64:68]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[64:68]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[64:68]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[64:68]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        g_band = hdul[index::20]
        g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_25(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        g_band = hdul[index::8]
        g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_50(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        g_band = hdul[index::4]
        g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel1Editor_75(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        g_band = hdul[index::4] + hdul[index::8]
        g_band = sorted(g_band, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorGBandLevel2Editor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter('ignore')
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 430.7 or hdul[0].header['WAVELNTH'] == '      430.700' or hdul[0].header['WAVELNTH'] == '       430.70000':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 430.7 or hdul[1].header['WAVELNTH'] == '      430.700' or hdul[1].header['WAVELNTH'] == '       430.70000':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[56:60]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[56:60]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[56:60]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[56:60]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        g_band = hdul[index]
        gregor_maps = [Map(g_band.data, primary_header)]
        return gregor_maps, {'path': file}



class LoadGregorContinuumLevel1Editor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter('ignore')
        hdul = fits.open(file)

        assert 'WAVELNTH' in hdul[0].header or 'FILTNOTE' in hdul[1].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['DATE'][0:4] == '2022':
            index = 1
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)

        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)

        continuum = hdul[index::2]
        continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        gregor_maps = [Map(hdu.data, primary_header) for hdu in continuum]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_5(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        continuum = hdul[index::40]
        continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in continuum]
        #gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_10(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        continuum = hdul[index::20]
        continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in continuum]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_25(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        continuum = hdul[index::8]
        continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in continuum]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_50(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        continuum = hdul[index::4]
        continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in continuum]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_75(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        #random_idx = np.random.randint(0, 100)
        continuum = hdul[index::4] + hdul[index::8]
        continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in continuum]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_new(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter("ignore")
        hdul = fits.open(file)
        #
        assert 'FILTNOTE' in hdul[1].header, 'Invalid GREGOR file %s' % file
        if hdul[1].header['FILTNOTE'] == 'blue continuum':
            index = 1
        elif hdul[2].header['FILTNOTE'] == 'blue continuum':
            index = 2
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[65:69]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[65:69]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[65:69]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[65:69]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #primary_header['cdelt1'] = 0.0253
        #primary_header['cdelt2'] = 0.0253
        #
        g_band = hdul[index::2]
        #
        gregor_maps = [Map(hdu.data, primary_header) for hdu in g_band]
        return gregor_maps, {'path': file}


class LoadGregorContinuumLevel1Editor_1(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter('ignore')
        hdul = fits.open(file)

        assert 'WAVELNTH' in hdul[0].header or 'FILTNOTE' in hdul[1].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['DATE'][0:4] == '2022':
            index = 1
        elif  hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)

        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[60:64]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[60:64]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[60:64]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[60:64]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        continuum = hdul[index]
        #continuum = sorted(continuum, key=lambda hdu: hdu.header['TIMEOFFS'])
        gregor_maps = [Map(continuum.data, primary_header)]
        return gregor_maps, {'path': file}



class LoadGregorContinuumLevel2Editor(Editor):

    def call(self, file, **kwargs):
        warnings.simplefilter('ignore')
        hdul = fits.open(file)
        #
        assert 'WAVELNTH' in hdul[0].header, 'Invalid GREGOR file %s' % file
        if hdul[0].header['WAVELNTH'] == 450.55 or hdul[0].header['WAVELNTH'] == '      450.550' or hdul[0].header['WAVELNTH'] == '       450.55000':
            index = 0
        elif hdul[1].header['WAVELNTH'] == 450.55 or hdul[1].header['WAVELNTH'] == '      450.550' or hdul[1].header['WAVELNTH'] == '       450.55000':
            index = 1
        else:
            raise Exception('Invalid GREGOR file %s' % file)
        #
        primary_header = hdul[0].header
        primary_header['cunit1'] = 'arcsec'
        primary_header['cunit2'] = 'arcsec'
        if int(file[68:72]) < 2019:
            primary_header['cdelt1'] = 0.0253
            primary_header['cdelt2'] = 0.0253
        elif int(file[68:72]) == 2019:
            primary_header['cdelt1'] = 0.0286
            primary_header['cdelt2'] = 0.0286
        elif int(file[68:72]) == 2020:
            primary_header['cdelt1'] = 0.0271
            primary_header['cdelt2'] = 0.0271
        elif int(file[68:72]) > 2020:
            primary_header['cdelt1'] = 0.02761
            primary_header['cdelt2'] = 0.02761
        else:
            raise Exception("Invalid Resolution for file %s" % file)
        #
        continuum = hdul[index]
        gregor_maps = [Map(continuum.data, primary_header)]
        return gregor_maps, primary_header


class ReadSimulationEditor(Editor):

    def call(self, filename, **kwargs):

        f = np.fromfile(filename, dtype='float32')
        nvars = f[0].astype('int')
        ny = f[1].astype('int')
        nx = f[2].astype('int')
        t_iteration = f[3].astype('int')
        arr = f[4:]
        arr = arr.reshape((nvars, ny, nx))
        index_ic = 0
        data = arr[index_ic, :, :]
        scale = (0.12144, 0.12144)
        my_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2012-01-01",
                            observer='earth', frame=frames.Helioprojective)
        header = make_fitswcs_header(data, my_coord, scale=scale * u.arcsec / u.pix)
        sim_map = Map(data, header)

        return sim_map


class LimbEditor(Editor):

    def call(self, smap, **kwargs):
        observer_coordinate = smap.observer_coordinate
        obs = SkyCoord(lon=60 * u.deg, lat=0 * u.deg, radius=observer_coordinate.radius,
                       frame=frames.HeliographicStonyhurst, obstime=observer_coordinate.obstime)
        obs_ref_coord = SkyCoord(-830 * u.arcsec, -50 * u.arcsec,
                                 obstime=smap.reference_coordinate.obstime,
                                 observer=obs,
                                 rsun=smap.reference_coordinate.rsun,
                                 frame="helioprojective")

        obs_header = make_fitswcs_header(
            smap.data.shape,
            obs_ref_coord,
            scale=u.Quantity(smap.scale),
            rotation_matrix=smap.rotation_matrix,
            instrument="Simulation",
            wavelength=smap.wavelength
        )

        return smap.reproject_to(obs_header)



class GregorGBandPrepEditor(Editor):

    def call(self, gregor_maps, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore warnings

            gregor_maps.meta["waveunit"] = "nm"
            if gregor_maps.meta["wavelnth"] != 430.7:
                gregor_maps.meta["wavelnth"] = 430.7

        return gregor_maps

class GregorContinuumPrepEditor(Editor):

    def call(self, gregor_maps, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            gregor_maps.meta["waveunit"] = "nm"
            if gregor_maps.meta["wavelnth"] != 450.55:
                gregor_maps.meta["wavelnth"] = 450.55

        return gregor_maps


class PaddingEditor(Editor):
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.target_shape
        x_pad = (p[0] - s[-2]) / 2
        y_pad = (p[1] - s[-1]) / 2
        pad = [(int(np.floor(x_pad)), int(np.ceil(x_pad))),
               (int(np.floor(y_pad)), int(np.ceil(y_pad)))]
        if len(s) == 3:
            pad.insert(0, (0, 0))
        return np.pad(data, pad, 'constant', constant_values=np.nan)


class UnpaddingEditor(Editor):
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.target_shape
        x_unpad = (s[-2] - p[0]) / 2
        y_unpad = (s[-1] - p[1]) / 2
        #
        unpad = [(None if int(np.floor(y_unpad)) == 0 else int(np.floor(y_unpad)),
                 None if int(np.ceil(y_unpad)) == 0 else -int(np.ceil(y_unpad))),
                 (None if int(np.floor(x_unpad)) == 0 else int(np.floor(x_unpad)),
                  None if int(np.ceil(x_unpad)) == 0 else -int(np.ceil(x_unpad)))]
        data = data[:, unpad[0][0]:unpad[0][1], unpad[1][0]:unpad[1][1]]
        return data


class GregorPaddingEditor(Editor):


    def call(self, data, **kwargs):

        #data = data[78: 1782, 118: 2254]
        data = data[228: 1932, 118: 2254]
        return data


class SimulationPaddingEditor(Editor):

    def call(self, data, **kwargs):
        data = data[70:1010, 50:500]
        return data


class ReshapeEditor(Editor):

    def __init__(self, shape):
        self.shape = shape

    def call(self, data, **kwargs):
        data = data[:self.shape[0], :self.shape[1]]
        return np.reshape(data, self.shape).astype(np.float32)


class ImagePatch(Editor):

    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[0] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[1] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        x = randint(0, data.shape[0] - self.patch_shape[0])
        y = randint(0, data.shape[1] - self.patch_shape[1])
        patch = data[x:x + self.patch_shape[0], y:y + self.patch_shape[1]]

        return patch


class RandomPatchEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)
        x = randint(0, data.shape[1] - self.patch_shape[0])
        y = randint(0, data.shape[2] - self.patch_shape[1])
        patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        patch = np.copy(patch)  # copy from mmep
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        assert not np.any(np.isnan(patch)), 'NaN found'
        return patch


class StackEditor(Editor):

    def __init__(self, data_sets):
        self.data_sets = data_sets

    def call(self, idx, **kwargs):
        results = [dp.getIndex(idx) for dp in self.data_sets]
        return np.concatenate([img for img, kwargs in results], 0), {'kwargs_list': [kwargs for img, kwargs in results]}


class ContrastNormalizeEditor(Editor):

    def __init__(self, use_median=False, shift=None, normalization=None):
        self.use_median = use_median
        self.shift = shift
        self.normalization = normalization

    def call(self, data, **kwargs):
        shift = np.mean(data)
        data = (data - shift)
        data = np.clip(data, -1, 1)
        return data


class NormalizeExposureEditor(Editor):
    def __init__(self, target=1 * u.s):
        self.target = target
        super().__init__()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        data = s_map.data
        data = data / s_map.exposure_time.to(u.s).value * self.target.to(u.s).value
        return Map(data.astype(np.float32), s_map.meta)

class LoadMapEditor(Editor):

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_map = Map(data)
            s_map.meta['timesys'] = 'tai'  # fix leap seconds
            return s_map, {'path': data}


class ScaleEditor(Editor):
    def __init__(self, arcspp):
        self.arcspp = arcspp
        super(ScaleEditor, self).__init__()

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            scale_factor = s_map.scale[0].value / self.arcspp
            new_dimensions = [int(s_map.data.shape[1] * scale_factor),
                              int(s_map.data.shape[0] * scale_factor)] * u.pixel
            s_map = s_map.resample(new_dimensions)

            return Map(s_map.data.astype(np.float32), s_map.meta)


class CropEditor(Editor):
    def __init__(self, start_x, end_x, start_y, end_y):
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y

    def call(self, data, **kwargs):

        crop = data[self.start_x: self.end_x, self.start_y:self.end_y]
        return crop


class ShiftMeanEditor(Editor):
    def call(self, data, **kwargs):
        mean = np.mean(data)
        data = (data - mean)
        data = np.clip(data, -1, 1)
        return data


class RemoveBackgroundEditor(Editor):
    def call(self, data, **kwargs):
        dummy = np.argwhere(data != -1)
        max_y = dummy[:, 0].max()
        min_y = dummy[:, 0].min()
        max_x = dummy[:, 1].max()
        min_x = dummy[:, 1].min()
        crop_image = data[min_y:max_y, min_x: max_x]
        return crop_image


class CropSimulationEditor(Editor):
    def call(self, data, **kwargs):
        fr = data[0]
        lr = data[data.shape[0]-1]
        find_fr = np.argwhere(fr != -1)
        min_fr = find_fr.min()
        max_fr = find_fr.max()
        find_lr = np.argwhere(lr != -1)
        min_lr = find_lr.min()
        max_lr = find_lr.max()

        crop = data[0:data.shape[0]-1, min_lr:max_fr]
        return crop
