# %% imports
import os, datetime, json, glob, configparser
import numpy as np
import pandas as pd
import xarray as xr
import shapefile as shp
import hvplot.xarray

from osgeo import gdal
print("Using gdal version", gdal.__version__)
from functools import partial

# pywapor
import pywapor
print("Using pywapor version:", pywapor.__version__)
"""
Generates input data for `pywapor.et_look`.
"""
import pywapor.general.pre_defaults as defaults
import pywapor.se_root as se_root
import pywapor.general.levels as levels

from pywapor.collect import downloader
from pywapor.general.logger import log, adjust_logger
from pywapor.general import compositer
from pywapor.general.variables import fill_attrs
from pywapor.enhancers.temperature import lapse_rate as _lapse_rate
from pywapor.general.processing_functions import remove_temp_files


# %% --------------------- Setup accounts ----------------------------------
# for more details, check README.txt
# pywapor.collect.accounts.setup('NASA')
# pywapor.collect.accounts.setup('EARTHEXPLORER')
# pywapor.collect.accounts.setup('COPERNICUS_DATA_SPACE')
# pywapor.collect.accounts.setup('TERRA')
# pywapor.collect.accounts.setup('ECMWF')
# %% ------------- CONFIGURATION ------------------------
# folder in which this script is located
os.chdir('/SOS-WATER/')

#%%
cfg = configparser.ConfigParser()
path = os.getcwd()

config_filename = "./config.cfg" 
cfg.read(config_filename)
CONFIG_DATA = {}
print(cfg.sections())
for section_name in cfg.sections():
    CONFIG_DATA[section_name] = {}
    for item_name in cfg.items(section_name):
        CONFIG_DATA[section_name][item_name[0]] = cfg.get(
            section_name, item_name[0])
        globals()[item_name[0]] = cfg.get(section_name, item_name[0])

project_folder = r"" + shp_name + "" + version
# Time period of analysis
timelim = [date_ini,date_end]
# ----------------------- END OF CONFIGURATION -------------------------------------

# %% Functions 
def rename_vars(ds, *args):
    """Rename some variables in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset whose variables will be renamed.

    Returns
    -------
    xr.Dataset
        Dataset with renamed variables.
    """
    varis = ["p", "ra", "t_air", "t_air_min", "t_air_max", "u", "vp",
            "u2m", "v2m", "qv", "p_air", "p_air_0", "wv", "t_dew"]
    present_vars = [x for x in varis if x in ds.variables]
    ds = ds.rename({k: k + "_24" for k in present_vars})
    return ds

def lapse_rate(ds, *args):
    """Applies lapse rate correction to variables whose name contains `"t_air"`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset on whose variables containing `"t_air"` a lapse rate correction will be applied.

    Returns
    -------
    xr.Dataset
        Dataset on whose variables containing `"t_air"` a lapse rate correction has been applied.
    """
    present_vars = [x for x in ds.variables if "t_air" in x]
    for var in present_vars:
        ds = _lapse_rate(ds, var)
    return ds

def calc_doys(ds, *args, bins = None):
    """Calculate the day-of-the-year (doy) in the middle of a timebin and assign the results to a new
    variable `doy`.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    bins : list, optional
        List of boundaries of the timebins, by default None

    Returns
    -------
    xr.Dataset
        Dataset with a new variable containing the `doy` per timebin.
    """
    bin_doys = [int(pd.Timestamp(x).strftime("%j")) for x in bins]
    doy = np.mean([bin_doys[:-1], bin_doys[1:]], axis=0, dtype = int)
    if "time_bins" in list(ds.variables):
        ds["doy"] = xr.DataArray(doy, coords = ds["time_bins"].coords).chunk("auto")
    return ds

def add_constants(ds, *args):
    """Adds default dimensionless constants to a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to which the constants will be added.

    Returns
    -------
    xr.Dataset
        Dataset with extra variables.
    """
    ds = ds.assign(defaults.constants_defaults())
    return ds

def remove_empty_statics(ds, *args):
    for var in ds.data_vars:
        if "time" not in ds[var].coords and "time_bins" not in ds[var].coords:
            if ds[var].isnull().all():
                ds = ds.drop(var)
                log.info(f"--> Removing `{var}` from dataset since it is empty.")
    return ds

def add_constants_new(ds, *args):
    c = defaults.constants_defaults()
    for var, value in c.items():
        if var not in ds.data_vars:
            ds[var] = value
    return ds

# We need to know the corners of the shapefile in order to run the algorithm 
sf = shp.Reader(os.getcwd() + '/GIS/jucar/' + shp_name + '.shp')
sf.bbox
lonmin = sf.bbox[0]
latmin = sf.bbox[1]
lonmax = sf.bbox[2]
latmax = sf.bbox[3]

latlim = [latmin - 0.11, latmax + 0.11]
lonlim = [lonmin - 0.11, lonmax + 0.11]
enhancers = [lapse_rate]

# Open SIGPAC dataset
lulc = xr.open_dataset(os.getcwd() + '/GIS/jucar/' + lulc_map)

#Some adjustments to convert the tiff to xarray
lulc =lulc.rename_vars({'band_data':'lulc'})
land_mask = xr.Dataset(
    {
        "land_mask": (["y", "x"], lulc.lulc.values[0,:,:]),
    },
    coords={
        "x": (["x"], lulc.x.values),
        "y": (["y"], lulc.y.values),
        "spatial_ref": lulc.spatial_ref
        
    },
)

land_mask.land_mask.attrs = {'grid_mapping':'spatial_ref'}
land_mask = land_mask.fillna(0)

with open(os.getcwd() + "/" + downloads_file, "r") as fp:
    custom_downloads = json.load(fp)

bins = compositer.time_bins(timelim, bin_length)
general_enhancers = enhancers + [rename_vars, fill_attrs, partial(calc_doys, bins = bins), remove_empty_statics, add_constants_new]
adjusted_timelim = [bins[0], bins[-1]]
buffered_timelim = [adjusted_timelim[0] - np.timedelta64(3, "D"), 
                    adjusted_timelim[1] + np.timedelta64(3, "D")]


# %% ---------------- Download custom configs ------------------------
dss, sources = downloader.collect_sources(project_folder, custom_downloads, latlim, lonlim, buffered_timelim)

# %% ---------------- Conversion from rh to qv -----------------------
# SH=RH*e/(100-RH); e = A*exp(B*T/(T+C)), T temperature in ºC, A=611.21 Pa, B=17.502, C=240.97 ºC
# take agrometeorological indicators and open rh, then apply the formula, using t_air on the same file

sis_agro_indicators = xr.open_dataset(project_folder + '/ERA5/' + 'sis-agrometeorological-indicators.nc')

def compute_qv(ds):

    a = 611.21
    b = 17.502
    c = 240.97
    e = a*np.exp((b*ds['t_air'])/(ds['t_air']+c))

    ds['qv'] = (ds['rh']*0.01 * e)/(100-ds['rh']*0.01)/100

    return ds[['qv','spatial_ref']]
    #return ds['qv']

qv = compute_qv(sis_agro_indicators)

qv['qv'].attrs = sis_agro_indicators['rh'].attrs
qv.to_netcdf(project_folder + '/ERA5/' + 'qv.nc')

# ------------ Crop files to save memory and resources ------------------------------------
# If enough computational resources, this piece of code can be commented
#  -------------- Filter nc files to be cropped --------------------------------------------- 

# The ones with 30m resolution, the problematic ones 
nc_filenames = ['30M','LC09_SR','LC08_SR','LE07_SR','LC05_SR','LC09_ST','LC08_ST','LE07_ST','LC05_ST']

nc_list = glob.glob('**/*.nc',recursive=True)

filtered_files = [file for file in nc_list if os.path.basename(file).split('.')[0] in nc_filenames]
filtered_files = [file for file in filtered_files if project_folder in file]

print(filtered_files)
for file in filtered_files:
    filename = file.split('/')[-1]
    print(filename)
    ds = xr.open_dataset(file)

    cropped_land_mask = land_mask.sel(y=slice(ds.y.values[0],ds.y.values[-1]), x=slice(ds.x.values[0],ds.x.values[-1]))

    ds_int = ds.interp(y=cropped_land_mask.y.values,
                        x=cropped_land_mask.x.values,
                        method='nearest')
    ds_int = ds_int.assign_coords({'y':cropped_land_mask.y.values,'x': cropped_land_mask.x.values})

    ds_cropped = ds_int.sel(y=slice(latmax,latmin), x=slice(lonmin,lonmax))
    geotransform = ds_cropped.spatial_ref.attrs['GeoTransform'].split(' ')
    xsize = ds_cropped.x.values[1]-ds_cropped.x.values[0]
    ysize = ds_cropped.y.values[1]-ds_cropped.y.values[0]

    x_geo = ds_cropped.x.values[0] - xsize/2
    y_geo = ds_cropped.y.values[0] - ysize/2

    new_geotransform = str(x_geo) + ' ' + str(xsize) + ' ' + geotransform[2] + ' ' + str(y_geo) + ' ' + geotransform[4] + ' ' + str(ysize)

    ds_cropped.spatial_ref.attrs['GeoTransform'] = new_geotransform

    ds.close()
    ds_cropped.to_netcdf(file,mode='w')
    del(ds)
    del(ds_cropped)

    print(file + 'done')

# %% ----------- Configuration to RUN PRE-ET-Look ---------------------------------------

with open(os.getcwd() + "/" + se_downloads_file, "r") as fp:
    se_root_custom_products = json.load(fp)

def qv_sideload(**kwargs):
    fh = project_folder + '/' + 'ERA5' + '/qv.nc'
    ds = xr.open_dataset(fh)
    return ds

# Join the era5 
qv_config = {'qv': {'products': [{'source': qv_sideload,
    'product_name': 'qv',
    'enhancers': 'default'}],
    'composite_type': 'mean',
  'temporal_interp': 'linear',
  'spatial_interp': 'bilinear'}}

se_root_custom_config = {**se_root_custom_products,**qv_config}

se_root_def_config = pywapor.general.levels.pre_et_look_levels(level = "level_3", bin_length = 1)["se_root"]
se_root_dler = partial(se_root.se_root, sources = se_root_custom_config)
se_root_def_config["products"][0]["source"] = se_root_dler

se_root_custom_config = {'se_root':se_root_def_config}

custom_downloads = {**custom_downloads,**qv_config}
pre_et_look_custom_config =  {**custom_downloads,**se_root_custom_config} 

# %% -------------------------- RUN PRE ET-Look -------------------------------------------------
# Generates et_look_in.nc file, containing all inputs except landuse 
# dependent ones, which will be ingested from SIGPAC
latitudes = [latmin,latmax]
longitudes = [lonmin,lonmax]

ds  = pywapor.pre_et_look.main(project_folder, latitudes, longitudes, timelim, sources = pre_et_look_custom_config,bin_length=bin_length)

# %% ------------------------- SIGPAC ingestion -------------------------------------------------
pre_etlook_ds = xr.open_dataset(project_folder + '/et_look_in.nc')
land_mask = land_mask.sel(y=slice(pre_etlook_ds.y.values[0],pre_etlook_ds.y.values[-1]), x=slice(pre_etlook_ds.x.values[0],pre_etlook_ds.x.values[-1]))

# Land mask and et_look_in don't have exactly the same resolution
# we have to slightly resample
land_mask = land_mask.interp(y=pre_etlook_ds.y.values,
                    x=pre_etlook_ds.x.values,
                    method='nearest')
land_mask = land_mask.assign_coords({'y':pre_etlook_ds.y.values,'x': pre_etlook_ds.x.values})

# Get landuse specific parameters
class_values_dict = pd.read_excel(os.getcwd() + '/' + lulc_table,index_col=0).to_dict()
print(class_values_dict)

rs_min = land_mask
for i in range(1,len(class_values_dict)+1):
    rs_min = rs_min.where(land_mask!=i,class_values_dict['rs_min'][i])

rs_min = rs_min.rename({'land_mask':'rs_min'})

z_obst_max = land_mask
for i in range(1,len(class_values_dict)+1):
    z_obst_max = z_obst_max.where(land_mask!=i,class_values_dict['z_obst_max'][i])

z_obst_max = z_obst_max.rename({'land_mask':'z_obst_max'})

phot_eff = land_mask
for i in range(1,len(class_values_dict)+1):
    phot_eff = phot_eff.where(land_mask!=i,class_values_dict['phot_eff'][i])

phot_eff = phot_eff.rename({'land_mask':'phot_eff'})

# % Include these variables in the ETLook in file
pre_etlook_ds['land_mask']= land_mask.assign_coords({"x": pre_etlook_ds.x,"y":pre_etlook_ds.y})['land_mask'].drop_vars('spatial_ref')
pre_etlook_ds['rs_min']= rs_min.assign_coords({"x": pre_etlook_ds.x,"y":pre_etlook_ds.y})['rs_min'].drop_vars('spatial_ref')
pre_etlook_ds['z_obst_max']= z_obst_max.assign_coords({"x": pre_etlook_ds.x,"y":pre_etlook_ds.y})['z_obst_max'].drop_vars('spatial_ref')
pre_etlook_ds['phot_eff']= phot_eff.assign_coords({"x": pre_etlook_ds.x,"y":pre_etlook_ds.y})['phot_eff'].drop_vars('spatial_ref')

os.rename(project_folder + '/et_look_in.nc',project_folder + '/et_look_in_tmp.nc')
pre_etlook_ds.to_netcdf(project_folder + '/et_look_in.nc',mode='w')
os.remove(project_folder + '/et_look_in_tmp.nc')
# %% ---------------------- RUN ETLook ------------------------------------
# Generates etlook_out.nc file 
et_look_out = pywapor.et_look.main(pre_etlook_ds,export_vars='all')

# %% ---------------- Reference ETLook out ----------------------
# et_look_out file is not well referenced by default, need to introduce spatial reference

et_look_out = xr.open_dataset(project_folder +'/et_look_out.nc')
et_look_in = xr.open_dataset(project_folder +'/et_look_in.nc')

et_look_out['spatial_ref'] = et_look_in['spatial_ref']
for var in et_look_out:
    if var != 'spatial_ref':
        et_look_out[var].attrs = {'grid_mapping':'spatial_ref'}

os.rename(project_folder + '/et_look_out.nc',project_folder + '/et_look_out_tmp.nc')
et_look_out.to_netcdf(project_folder + '/et_look_out.nc',mode='w')
os.remove(project_folder + '/et_look_out_tmp.nc')

# %% ------------------ OTHER FILES OF INTEREST ----------------------
# Mean annual evapotranspiration (mm)
ma_et = et_look_out[['et_24_mm']].groupby('time_bins.year').mean()*365
ma_et['spatial_ref'] = et_look_out['spatial_ref']
ma_et['et_24_mm'].attrs = {'grid_mapping':'spatial_ref'}
ma_et.to_netcdf(project_folder + '/ma_et.nc')

# Dekadal ndvi
ndvi_dk = et_look_out[['ndvi','spatial_ref']]
ndvi_dk.to_netcdf(project_folder + '/dk_ndvi.nc')

