import os
import time
from deepliif.options.test_options import TestOptions
from deepliif.options import read_model_params, Options, print_options
from deepliif.data import create_dataset
from deepliif.models import create_model, infer_modalities
from deepliif.util.visualizer import save_images
from deepliif.util import html
from deepliif.data import base_dataset
import torch
import click
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


import sys
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
from scipy import ndimage

warnings.filterwarnings("ignore")
import os
from pathlib import Path
from math import ceil
import glob






def transform(img):
    return transforms.Compose([
        transforms.Lambda(lambda i: base_dataset.__make_power_2(i, base=4, method=Image.BICUBIC)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(img).unsqueeze(0)

def do_mask(img, lvl_resize):
    ''' create tissue mask '''
    # get he image and find tissue mask
    he = np.array(img)
    he = he[:, :, 0:3]
    heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagem = cv2.bitwise_not(thresh1)
    tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=5))
    tissue_mask = morphology.remove_small_objects(tissue_mask, 1000)
    tissue_mask = ndimage.binary_fill_holes(tissue_mask)

    # create polygons for faster tiling in cancer detection step
    polygons = []
    contours, hier = cv2.findContours(tissue_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cvals = contour.transpose(0, 2, 1)
        cvals = np.reshape(cvals, (cvals.shape[0], 2))
        cvals = cvals.astype('float64')
        for i in range(len(cvals)):
            cvals[i][0] = np.round(cvals[i][0] * lvl_resize, 2)
            cvals[i][1] = np.round(cvals[i][1] * lvl_resize, 2)
        try:
            poly = Polygon(cvals)
            if poly.length > 0:
                # polygons.append(Polygon(poly.exterior))
                rp = Polygon(poly.exterior)
                if rp.is_valid == False:
                    rp = rp.buffer(0)
                if rp.is_valid == True:
                    polygons.append(rp)
        except:
            pass
    tissue = unary_union(polygons)
    while not tissue.is_valid:
        print('pred_union is invalid, buffering...')
        tissue = tissue.buffer(0)

    return tissue, tissue_mask




def check_tissue(tile_starts, tile_ends, roi):
    ''' checks if tile in tissue '''
    tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [tile_ends[0], tile_starts[1]], [
        tile_ends[0], tile_ends[1]]
    tile_box = list(tile_box)
    tile_box = MultiPoint(tile_box).convex_hull
    ov = 0  # initialize
    if tile_box.intersects(roi):
        ov_reg = tile_box.intersection(roi)
        ov += ov_reg.area / tile_box.area

    return ov

def whitespace_check(im):
    ''' checks if meets whitespace requirement'''
    bw = im.convert('L')
    bw = np.array(bw)
    bw = bw.astype('float')
    bw = bw / 255
    prop_ws = (bw > 0.8).sum() / (bw > 0).sum()
    return prop_ws

def slide_ROIS(polygons, edges, mpp, savename, labels, ref, roi_color):
    ''' generate geojson from polygons '''
    # edges = edges.buffer(0)
    edge_merge = unary_union(edges)
    all_polys = polygons + [edge_merge]
    # all_polys = unary_union(polygons)
    final_polys = []
    for poly in all_polys:
        if poly.type == 'Polygon':
            polypoints = poly.exterior.xy
            polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
            polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
            newpoly = Polygon(zip(polyx, polyy))
            if newpoly.area * mpp * mpp > 0.1:
                final_polys.append(newpoly)
        if poly.type == 'MultiPolygon':
            for roii in poly.geoms:
                polypoints = roii.exterior.xy
                polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                newpoly = Polygon(zip(polyx, polyy))
                if newpoly.area * mpp * mpp > 0.1:
                    final_polys.append(newpoly)
    # note to python 3.10 users, please comment line directly below and uncomment line after that
    # final_shape = unary_union(final_polys)
    final_shape = final_polys
    try:
        trythis = '['
        for i in range(0, len(final_shape)):
            trythis += json.dumps(
                {"type": "Feature", "id": "PathAnnotationObject",
                 "geometry": shapely.geometry.mapping(final_shape[i]),
                 "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                "measurements": []}}, indent=4)
            if i < len(final_shape) - 1:
                trythis += ','
        trythis += ']'
    except:
        trythis = '['
        trythis += json.dumps(
            {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape),
             "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                            "measurements": []}}, indent=4)
        trythis += ']'

    with open(savename, 'w') as outfile:
        outfile.write(trythis)

    print('done')
    return

def tile_ROIS(imgname, mask_arr, tissue, tile_starts, tile_ends,tile_size):
    tile_box = [tile_starts[0] + 5, tile_starts[1] + 5], [tile_starts[0] + 5, tile_ends[1] - 5], [tile_ends[0] - 5,tile_starts[1] + 5], [tile_ends[0] - 5, tile_ends[1] - 5]
    tile_box = list(tile_box)
    tile_box = MultiPoint(tile_box).convex_hull
    # print(np.max(mask_arr))

    polygons = []
    edge_polygons = []
    nameparts = str.split(imgname, '_')
    pos = str.split(nameparts[0], '-')
    sz = str.split(nameparts[1], '-')
    radj = max([int(sz[0]), int(sz[1])]) / (tile_size - 1)
    start1 = int(pos[0])
    start2 = int(pos[1])
    # c = morphology.remove_small_objects(mask_arr.astype(bool), 10, connectivity=2)
    # c = morphology.binary_closing(c)
    # c = morphology.remove_small_holes(c, 100)
    # contours, hier = cv2.findContours(c.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hier = cv2.findContours(mask_arr.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
    # for celli in range(1,np.max(mask_arr)):
    #     cellmask = np.zeros_like(mask_arr)
    #     cellmask[mask_arr==celli]=1
    #     contour, hier = cv2.findContours(cellmask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contour = contour[0]
        cvals = contour.transpose(0, 2, 1)
        cvals = np.reshape(cvals, (cvals.shape[0], 2))
        cvals = cvals.astype('float64')
        for i in range(len(cvals)):
            cvals[i][0] = start1 + radj * (cvals[i][0])
            cvals[i][1] = start2 + radj * (cvals[i][1])
        try:
            poly = Polygon(cvals)
            if poly.length > 0:
                if poly.is_valid == False:
                    poly = poly.buffer(0)
                if poly.intersects(tissue):
                    ov_reg = poly.intersection(tile_box)
                    ov = ov_reg.area / poly.area
                    if ov < 1:
                        rp = Polygon(poly.exterior)
                        if rp.is_valid == False:
                            rp = rp.buffer(0)
                        if rp.is_valid == True:
                            edge_polygons.append(rp)
                    else:
                        rp = Polygon(poly.exterior)
                        if rp.is_valid == False:
                            rp = rp.buffer(0)
                        if rp.is_valid == True:
                            polygons.append(rp)
        except:
            pass
    return polygons, edge_polygons


def test(checkpoints_dir,name,file_location,mag_extract,tile_size,json_location,save_location):

    # LOAD AND INITIALIZE MODEL


    # retrieve options used in training setting, similar to cli.py test
    # checkpoints_dir = '/data/MIP/harmon-lab/mpca/codes/DeepLIIF/model-server'
    # name = 'DeepLIIF_Latest_Model'
    model_dir = os.path.join(checkpoints_dir, name)
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')

    # overwrite/supply unseen options using the values from the options provided in the command
    setattr(opt,'checkpoints_dir',checkpoints_dir)
    # setattr(opt,'dataroot','./Sample_Ki67')
    setattr(opt,'name',name)
    # setattr(opt,'results_dir',results_dir)
    setattr(opt,'num_test',1)

    if not hasattr(opt,'seg_gen'): # old settings for DeepLIIF models
        opt.seg_gen = True

    gpu_ids=[]
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        number_of_gpus = 0
        gpu_ids = [-1]
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))

    opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus

    # hard-code some parameters for test.py
    opt.aspect_ratio = 1.0 # from previous default setting
    opt.display_winsize = 512 # from previous default setting
    opt.use_dp = True # whether to initialize model in DataParallel setting (all models to one gpu, then pytorch controls the usage of specified set of GPUs for inference)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    print_options(opt)
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    torch.backends.cudnn.benchmark = False
    model.eval()
    # if opt.eval:
    #     model.eval()


    ## now start loop over WSI

    oslide = openslide.OpenSlide(file_location)
    savnm = os.path.basename(file_location)
    save_name = str(Path(savnm).with_suffix(''))

    # define offset
    try:
        offset = [int(oslide.properties[openslide.PROPERTY_NAME_BOUNDS_X]),
                  int(oslide.properties[openslide.PROPERTY_NAME_BOUNDS_Y])]
        print('no offset')
    except:
        offset = [0, 0]

    # this is physical microns per pixel
    acq_mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

    # this is nearest multiple of 20 for base layer
    base_mag = int(20 * round(float(acq_mag) / 20))

    # this is how much we need to resample our physical patches for uniformity across studies
    physSize = round(tile_size * (base_mag / mag_extract))

    if json_location == 'none':
        # intermeadiate level for probability map
        lvl_img = oslide.read_region((0, 0), 5, oslide.level_dimensions[5])
        lvl_resize = oslide.level_downsamples[5]
        # send to get tissue polygons
        print('no json supplied, detecting tissue')
        tissue, he_mask = do_mask(lvl_img, lvl_resize)
    elif os.path.exists(json_location):
        with open(json_location) as f:
            allobjects = geojson.load(f)
        allshapes = [shape(obj["geometry"]) for obj in allobjects]
        # adjust for offset
        final_polys = []
        for poly in allshapes:
            # print(poly)
            if poly.type == 'Polygon':
                polypoints = poly.exterior.xy
                polyx = [np.round(number + offset[0], 1) for number in polypoints[0]]
                polyy = [np.round(number + offset[1], 1) for number in polypoints[1]]
                # newpoly = Polygon(zip(np.round(polypoints[0],1),np.round(polypoints[1],1)))
                newpoly = Polygon(zip(polyx, polyy))
                final_polys.append(newpoly)
            if poly.type == 'MultiPolygon':
                for roii in poly.geoms:
                    polypoints = roii.exterior.xy
                    polyx = [np.round(number + offset[0], 1) for number in polypoints[0]]
                    polyy = [np.round(number + offset[1], 1) for number in polypoints[1]]
                    # newpoly = Polygon(zip(np.round(polypoints[0], 1), np.round(polypoints[1], 1)))
                    newpoly = Polygon(zip(polyx, polyy))
                    final_polys.append(newpoly)
        tissue = unary_union(final_polys)
    else:
        print('failure to correctly specify json location, please enter "none" or full json path')

    nrow, ncol = oslide.level_dimensions[0]
    x_tiles = ceil(nrow / physSize)
    y_tiles = ceil(ncol / physSize)

    polygons_pos = []
    edge_polygons_pos = []
    polygons_neg = []
    edge_polygons_neg = []

    for y in range(0, y_tiles):
        for x in range(0, x_tiles):

            # grab tile coordinates
            tile_coords = [x * physSize, y * physSize]
            save_coords = str(tile_coords[0]) + "-" + str(tile_coords[1]) + "_" + '%.0f' % (
                physSize) + "-" + '%.0f' % (physSize)
            tile_ends = (x * physSize + physSize, y * physSize + physSize)

            # check for tissue membership
            tile_tiss = check_tissue(tile_starts=tile_coords, tile_ends=tile_ends, roi=tissue)
            if tile_tiss > 0.05:
                tile_pull = oslide.read_region((x * physSize, y * physSize), 0, (physSize, physSize))
                tile_pull = tile_pull.resize(size=(tile_size, tile_size), resample=Image.LANCZOS)

                #run inference
                data = {}
                # img = Image.open('/data/MIP/harmon-lab/mpca/codes/DeepLIIF/Sample_Ki67/test/1.png')
                img = Image.fromarray(np.asarray(tile_pull)[:, :, 0:3])
                _img = transform(img)
                data['A'] = _img
                data['B'] = [_img, _img, _img, _img, _img]
                data['A_paths'] = './'
                data['B_paths'] = './'
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                # images = {'Hema': visuals['fake_B_1'],'DAPI': visuals['fake_B_2'],'Lap2': visuals['fake_B_3'],'Marker': visuals['fake_B_4'],'Seg': visuals['fake_B_5']}
                seg = visuals['fake_B_5'][0].cpu().float().numpy()
                seg = (np.transpose(seg, (1, 2, 0)) + 1) / 2.0 * 255.0
                seg = seg.astype(np.uint8)
                cell = np.logical_and(np.add(seg[:, :, 0], seg[:, :, 2], dtype=np.uint16) > 150, seg[:, :, 1] <= 80)
                pos = np.logical_and(cell, seg[:, :, 0] >= seg[:, :, 2])
                neg = np.logical_xor(cell, pos)
                mask_pos = np.full(seg.shape[0:2], 0, dtype=np.uint8)
                mask_neg = np.full(seg.shape[0:2], 0, dtype=np.uint8)
                mask_pos[pos] = 1
                mask_neg[neg] = 1

                # convert positive detections to ROIs
                pred_polys, edge_preds = tile_ROIS(imgname=save_coords, mask_arr=mask_pos, tissue=tissue,
                                                        tile_starts=tile_coords, tile_ends=tile_ends,tile_size=tile_size)
                polygons_pos += pred_polys
                edge_polygons_pos += edge_preds

                # convert negative detections to ROIs
                pred_polys, edge_preds = tile_ROIS(imgname=save_coords, mask_arr=mask_neg, tissue=tissue,
                                                        tile_starts=tile_coords, tile_ends=tile_ends,tile_size=tile_size)
                polygons_neg += pred_polys
                edge_polygons_neg += edge_preds

    slide_ROIS(polygons=polygons_pos, edges=edge_polygons_pos,
                    mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                    savename=os.path.join(save_location, save_name + '_deepliif_pos.json'),
                    labels='pos', ref=offset, roi_color=-16711936)

    slide_ROIS(polygons=polygons_neg, edges=edge_polygons_neg,
                    mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                    savename=os.path.join(save_location, save_name + '_deepliif_neg.json'),
                    labels='neg', ref=offset, roi_color=-16711936)




if __name__ == '__main__':
    test(checkpoints_dir='./model-server',
         name = 'DeepLIIF_Latest_Model',
         file_location='path/to/WSI.tif',
         mag_extract=40,
         tile_size=512,
         json_location='path/to/WSI_roi.json',
         save_location='/path/to/output')
    
