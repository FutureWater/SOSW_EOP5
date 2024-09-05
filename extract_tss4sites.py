import json
import os
from os import path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib as plt

import datetime as datetime 
from datetime import date
from osgeo import ogr
import xarray as xr
import hvplot
import holoviews as hv
import hvplot.pandas
import hvplot.xarray  
import rioxarray
import timeit
import geopandas
import seaborn as sns
import requests
import configparser
import shapefile as shp
from rasterio import features
from matplotlib import pyplot as plt
import rasterio
import netCDF4

os.chdir('/SOS-WATER/')
demo_site = 'juc_roi_test_a5030-01'
area = demo_site.split('_')[-1] 

## Open shapefile with points
valsites = ogr.Open(os.getcwd() + '/' + demo_site  + '/' + area + '_valsites.shp')
shape = valsites.GetLayer(0)

landsat_files =  glob(os.getcwd() + '/' + demo_site + '/LANDSAT'+ '/*.nc',recursive=True)

final_df = pd.DataFrame()
for file in landsat_files:
    xr_dataset = xr.open_mfdataset(file)
    
    for feat in shape:
        x, y = json.loads(feat.ExportToJson())['geometry']['coordinates']
        id = json.loads(feat.ExportToJson())['properties']['id']
        luc_id = json.loads(feat.ExportToJson())['properties']['LUC_ID']
        df=xr_dataset.sel({'y':y,'x':x}, method='nearest').to_dataframe()
        df['id'] = id
        df['luc_id'] = luc_id
        df['sensor'] = file.split('/')[-1].split('_')[0]
        variable = file.split('/')[-1].split('_')[-1][0:2]
        df['variable'] = variable

        if variable == 'SR':
            var = 'ndvi'
        elif variable == 'ST':
            var = 'lst'

        df = df[['id','luc_id','sensor','variable',var,'y','x']]
        final_df = final_df.append(df)

final_df.to_excel(os.getcwd() + '/' + demo_site + '/' + area + '_valsites.xlsx', merge_cells=False)

