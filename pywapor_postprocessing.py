# %% imports
import os
from osgeo import gdal
print("Using gdal version", gdal.__version__)
import pywapor
print("Using pywapor version:", pywapor.__version__)
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import hvplot
import hvplot.pandas
import hvplot.xarray 
import holoviews as hv 
hv.extension('bokeh')  # to generate interactive plots 
import openpyxl
import hvplot.pandas # use hvplot directly with pandas
import pandas as pd
import seaborn as sns
# %% 
def simple_graph(ds,time_ini,time_end,x,y,width,height,title,kind='line'):

    if y == 'allvar':
        return ds.sel(date=slice(time_ini,time_end)).hvplot(x=x,width=width, height=height,title=title,kind=kind)
    else:
        return ds[y].sel(date=slice(time_ini,time_end)).hvplot(x=x,width=width, height=height,title=title,kind=kind) 
    

# %% ET_look input and output
os.chdir('/SOS-WATER/')

# dictionary with predominant crops per area
pred_crops = {'a5030-01': ['50','51','40','70','71'], 'a5030-03': ['50','51','20','21'],'a5150-01': ['51','31','11'],'a5125-01': ['20','70','71'] }

# Name of the shapefile, which is asumed to be placed in SOS-WATER/GIS/jucar
shp_name = 'juc_roi_test_a5030-01'
# Name of the version, if any (if not, leave it like "")
version = 'v1'
project_folder= shp_name + '/' + version
aoi = shp_name.split('_')[-1]
# new: categories is only the predominant crops per area
categories = pred_crops[aoi]

et_look_out = xr.open_mfdataset('./' + project_folder + '/et_look_out*.nc')
et_look_out = et_look_out.rename_vars({'et_24_mm':'et_mm', 'p_24':'prcp', 't_24_mm':'t_mm','int_mm':'i_mm','et_ref_24_mm':'et_ref_mm'})
et_look_out = et_look_out.rename({'time_bins':'date'})
et_look_in = xr.open_mfdataset('./' + project_folder + '/et_look_in*.nc')

# Open SIGPAC dataset
lulc = xr.open_dataset('/SOS-WATER/GIS/juc_sigpac_monocrops_30m_v2_epsg4326.tif')
lulc =lulc.rename_vars({'band_data':'lulc'})

land_mask = xr.Dataset(
    {
        "land_mask": (["y", "x"], lulc.lulc.values[0,:,:]),
    },
    coords={
        "x": (["x"], lulc.x.values),
        "y": (["y"], lulc.y.values),
        
    },
)

land_mask = land_mask.fillna(0)
# Crop to the study area only 
land_mask = land_mask.sel(y=slice(et_look_out.y.values[0],et_look_out.y.values[-1]), x=slice(et_look_out.x.values[0],et_look_out.x.values[-1]))

# Sometimes y values are flipped, dont know why
# land_mask = land_mask.sel(y=slice(et_look_out.y.values[-1],et_look_out.y.values[0]), x=slice(et_look_out.x.values[0],et_look_out.x.values[-1]))

# if they don't have the same resolution, we have to resample
if  et_look_out.x.values[0] - land_mask.x.values[0] != 0:
    land_mask = land_mask.interp(y=et_look_out.y.values,
                        x=et_look_out.x.values,
                        method='nearest')
    land_mask = land_mask.assign_coords({'y':et_look_out.y.values,'x': et_look_out.x.values})

et_look_out = et_look_out.where(land_mask['land_mask']!=0,np.NaN)
et_look_in = et_look_in.where(land_mask['land_mask']!=0,np.NaN)

et_look_out = et_look_out.sel(date=slice('2018','2022'))
et_look_in = et_look_in.sel(time_bins=slice('2018','2022'))

# %% Count pixels by category 
# Create a dictionary using a list comprehension
sres = 30
npixels = {
    value:
        int(land_mask['land_mask'].where(land_mask['land_mask']==int(value),np.NaN).count().values)*sres*sres/1000000
    
    for value in categories
}

# %% Takes al variables of interest from et_look_out and creates separate xarrays and dataframes with each variable name
# for instance, each variable has a xarray.Dataset with each variable one category, still didn't aggregate over time 
variables = ['ndvi','et_mm','prcp','npp','npp_max','se_root','t_mm','i_mm','et_ref_mm']
dk_table= pd.DataFrame(list(npixels.items()), columns=['LUC', 'area'])

for variable in variables:
    locals()[variable] = xr.Dataset()
    print(variable)
    for value in categories:
        locals()[variable][value]= et_look_out[variable].where(land_mask['land_mask']==int(value),np.NaN)
    locals()[variable+'_df'] = locals()[variable].median(dim=["x", "y"],skipna=True).to_dataframe()
    locals()[variable+'_df_final'] = locals()[variable+'_df'].reset_index().melt(id_vars='date',var_name='LUC',value_name=variable,ignore_index=False)
    dk_table = dk_table.merge(locals()[variable+'_df_final'])
# %%  ------------------------------------- DEKADAL TABLE ---------------------------------------------------------------------

dk_table['aoi'] = aoi
dk_table['dmp'] = dk_table['npp']*22.222
dk_table['t_m3'] = dk_table['t_mm']*10
dk_table['i_m3'] = dk_table['i_mm']*10
dk_table['et_m3'] = dk_table['et_mm']*10
dk_table = dk_table[['aoi','LUC','date','ndvi', 'prcp', 'se_root', 't_mm','i_mm','et_mm','et_ref_mm','t_m3','i_m3','et_m3','npp','npp_max','dmp']]
#dk_table = dk_table.round(2)
dk_table = dk_table.set_index(['aoi','LUC','date'])
#dk_table.to_excel(project_folder + '/' + aoi + '_dk.xlsx')

# %%  ------------------------------------- ANNUAL TABLE ---------------------------------------------------------------------
# Compute annual ET, T : averaging over time and median over space 
# generates numbers per category corresponding to the final value for the year
a_et = et_mm.median(dim=["x", "y"],skipna=True).groupby('date.year').mean()*365
a_et_m3 = a_et * 10
a_et_df = a_et.to_dataframe()
a_et_m3_df = a_et_m3.to_dataframe()

a_t = t_mm.median(dim=["x", "y"],skipna=True).groupby('date.year').mean()*365
a_t_m3 = a_t * 10

a_t_df = a_t.to_dataframe()
a_t_m3_df = a_t_m3.to_dataframe()

# Compute annual NPP: averaging over time and over median over space 
# generates numbers per category corresponding to the final value for the year
dmp = npp * 22.222
a_tbp = dmp.median(dim=["x", "y"],skipna=True).groupby('date.year').mean()*365
a_tbp_df = a_tbp.to_dataframe()

a_npp_ = npp.median(dim=["x", "y"],skipna=True).groupby('date.year').mean()*365
a_npp_df = a_npp_.to_dataframe() 
# %% Compute annual NBWP (TBP/T) and annual WPET (TBP/ET)
a_nbwpt_df = a_tbp_df/a_t_m3_df
a_nbwpet_df = a_tbp_df/a_et_m3_df

# %% Construct final annual table

final_table= pd.DataFrame(list(npixels.items()), columns=['LUC', 'area'])

t_df = a_t_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='t_mm',ignore_index=False).set_index(["year","LUC"])
et_df = a_et_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='et_mm',ignore_index=False).set_index(["year","LUC"])
t_m3_df = a_t_m3_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='t_m3',ignore_index=False).set_index(["year","LUC"])
et_m3_df = (a_et_df * 10).reset_index().melt(id_vars='year',var_name='LUC',value_name='et_m3',ignore_index=False).set_index(["year","LUC"])

tbp_df = a_tbp_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='tbp',ignore_index=False).set_index(["year","LUC"])
npp_df = a_npp_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='npp',ignore_index=False).set_index(["year","LUC"])

nbwpt_df = a_nbwpt_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='nbwp_t',ignore_index=False).set_index(["year","LUC"])
nbwpet_df = a_nbwpet_df.reset_index().melt(id_vars='year',var_name='LUC',value_name='nbwp_et',ignore_index=False).set_index(["year","LUC"])

final_table = pd.concat([t_m3_df,et_m3_df,tbp_df,nbwpt_df,nbwpet_df,t_df,et_df,npp_df],axis=1)

# %% Export both tables 
with pd.ExcelWriter(project_folder + '/' + aoi + '_' + version  + '.xlsx', engine='openpyxl') as writer:
   dk_table.to_excel(writer, sheet_name='dk',merge_cells=False)
   final_table.to_excel(writer, sheet_name='year',merge_cells=False)
    

# %% --------------------------- GENERATION OF NETCDFS  --------------------------------------

a_nbwpt = (dmp.groupby('date.year').mean()*365)/(t_mm.groupby('date.year').mean()*365*10)
a_nbwpet = (dmp.groupby('date.year').mean()*365)/(et_mm.groupby('date.year').mean()*365*10)

ma_nbwpt = a_nbwpt.mean(dim='year')
ma_nbwpet = a_nbwpt.mean(dim='year')

ma_nbwpt.to_netcdf(project_folder + '/ma_nbwpt_'+ aoi +'_' +version +'.nc') 
ma_nbwpet.to_netcdf(project_folder + '/ma_nbwpet_'+ aoi +'_' +version +'.nc')


# %% --------------------- PLOTS -----------------------------------------------------------

# %% --------- First plot: boxplot + stripplot ----------------------------------------------
if aoi == 'all':
    files = os.listdir(project_folder) 
    final_table = pd.DataFrame()
    for file in files:
     if file.endswith('.xlsx'):
         final_table = final_table.append(pd.read_excel(project_folder +'/'+ file,sheet_name='year'), ignore_index=False) 
    # this is the one that goes to the report
    final_table.groupby('LUC').mean().to_excel(project_folder + '/' + 'results_all.xlsx')

else:
    final_table = pd.read_excel(project_folder + '/'+ aoi + '_' +version +'.xlsx', sheet_name='year')

final_table = final_table.reset_index()
final_table = final_table[final_table['year']>=2018]
final_table = final_table[final_table['year']<=2022]
final_table['id'] = final_table['LUC'].astype(str).str[0]
final_table['id_irr'] = final_table['LUC'].astype(str).str[1]


dict_LUC = {"1":"Citrus Trees","2":"Nuts","3":"Fruit trees","4":"Olive Trees","5":"Arable Lands","6":"Orchards","7":"Vineyards"}
dict_Irr = {"0":"Rainfed","1":"Irrigated"}
final_table["LUC_name"] = final_table['id'].map(dict_LUC) 
final_table["Irr_name"] = final_table['id_irr'].map(dict_Irr) 

sns.set_theme(style="whitegrid")

f, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(
    final_table, x="LUC_name", y="nbwp_t", hue="Irr_name",
    whis=[0, 100], width=.6, palette={"Rainfed": "r", "Irrigated": "b"}
)

sns.stripplot(final_table, x="LUC_name", y="nbwp_t",size=4,hue='Irr_name',dodge=True,jitter=False,legend=False,palette={"Rainfed": "black", "Irrigated": "k"})
plt.xticks(rotation=30)

ax.set(ylabel="NBWP(kgDM/m3)")
#ax.set(xlabel="Land Use Category Name")
ax.set(title='Interannual Variability ' +'(' + 'all years' + ')')
plt.legend(loc='upper left')

# %% Facegrid plots: pixels per category, we dont use it 
ds = ma_nbwpt

# Determine the grid size for the subplot arrangement
num_vars = len(ds.data_vars)
num_cols = 3  # Set the number of columns in the grid
num_rows = -(-num_vars // num_cols)  # Ceiling division to determine the number of rows

# Create a figure with subplots in a grid
plt.figure(figsize=(15, num_rows * 5))
for i, var_name in enumerate(ds.data_vars, start=1):
    plt.subplot(num_rows, num_cols, i)
    data_var = ds[var_name]
    plt.pcolormesh(data_var.x, data_var.y, data_var, shading='auto')
    plt.colorbar(label=var_name)
    plt.title(f'Var {var_name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

plt.tight_layout()
plt.show()

# %% -------------------- VIOLIN  PLOT ---------------------------------------------------
import seaborn as sns
sns.set_theme(style="whitegrid")
ds = xr.open_mfdataset('./' + project_folder + '/ma_nbwpt*.nc') ##ma_nbwpt 

ds = ds[['40']] # if we want categories individually

ds = ds.where(ds<ds.quantile(0.9, dim=['x','y']) + 0.15,np.NaN)
ds = ds.where(ds>ds.quantile(0.1, dim=['x','y']) - 0.15,np.NaN)
# Prepare data for violin plots
data_for_plots = {var_name: ds[var_name].values.flatten() for var_name in ds.data_vars}

# Convert data to a DataFrame for seaborn compatibility
df0 = pd.DataFrame(data_for_plots)
df = df0.melt(var_name='LUC',value_name='nbwpt')


df['LUC_id'] = df['LUC'].astype(str).str[0]
df['Irr_id'] = df['LUC'].astype(str).str[1]
dict_LUC = {"1":"Citrus Trees","2":"Nuts","3":"Fruit trees","4":"Olive Trees","5":"Arable Lands","6":"Orchards","7":"Vineyards"}
dict_Irr = {"0":"Rainfed","1":"Irrigated"}
df["LUC_name"] = df['LUC_id'].map(dict_LUC) 
df["Irr_name"] = df['Irr_id'].map(dict_Irr) 

# Create Violinplot
sns.violinplot(data=df,linewidth=0.65,x='LUC_name',y='nbwpt',hue='Irr_name',inner="quart",split=True, fill=True,gap=.1, palette={"Rainfed": "r", "Irrigated": "b"}, bw_method = 'silverman', cut = 0, bw_adjust=1)
plt.ylim(0, 5)
plt.xticks(rotation=0)
plt.title('Spatial Variability per Category ' + '(' + aoi + ')')
plt.ylabel('NBWP(kgDM/m3)')
plt.xlabel('Land Use Category Name')
ax.set(title=aoi)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %%  --------------- LINE PLOT ------------------------------------------------------------
final_table = final_table.reset_index()
final_table['id'] = final_table['LUC'].astype(str).str[0]
final_table['id_irr'] = final_table['LUC'].astype(str).str[1]
final_table['year'] = final_table['year'].astype(str)

dict_LUC = {"1":"Citrus Trees","2":"Nuts","3":"Fruit trees","4":"Olive Trees","5":"Arable Lands","6":"Orchards","7":"Vineyards"}
dict_Irr = {"0":"Rainfed","1":"Irrigated"}
final_table["LUC_name"] = final_table['id'].map(dict_LUC) 
final_table["Irr_name"] = final_table['id_irr'].map(dict_Irr) 

sns.set_theme(style="whitegrid")

f, ax = plt.subplots(figsize=(7, 6))

sns.lineplot(
    final_table, x="year", y="et_mm", hue = 'LUC_name', style="Irr_name", palette = 'tab10'
     #palette={"Rainfed": "r", "Irrigated": "b"}
)
ax.set(ylabel="NBWP (kgDM/m3)")
ax.set(xlabel="year")
ax.set(title=aoi)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# %%