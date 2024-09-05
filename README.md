# <img src="https://github.com/mibrechb/SOSW_WP3/blob/main/imgs/sosw_logo.png" width="80"> SOS-Water - EOP5 Crop Water Productivity Mapper

The Earth Observation Prototype 5 (EOP5) for Crop water productivity mapping is a prototype algorithm designed to estimate......

This repository is part of the Deliverable 3.2 of SOS-Water - Water Resources System Safe Operating Space in a Changing Climate and Society ([DOI:10.3030/101059264](https://cordis.europa.eu/project/id/101059264)). Other code contributions to D3.2 can be found at the [SOS-Water - WP3 Earth Observation repository](https://github.com/mibrechb/SOSW_WP3).

Check out the project website at [sos-water.eu](https://sos-water.eu) for more information on the project.

## How to use

This folder contains all the necessary files and inputs to run pyWaPOR version 3.4.3 for the Jucar River Basin Pilot Case.
Steps to run the model: 

1. Create a new environment and install python (eg. *pywapor*) example in conda\
	*conda create --name pywapor python*
2. Install gdal=3.6 with conda\
	*conda install -c conda-forge gdal=3.6
3. Install the following packages\
	*conda install -c conda-forge pywapor=3.4.3 pyshp hvplot*
4. Create/Collect your EO accounts and password for the platforms below

5. Open config.cfg and fill accordingly (default values should work)
6. Open lulc_lut and change landuse parameters if needed
7. Files *custom_downloads.json* and *se_root_custom_config.json* include native sources from which download variables, change only if needed.
8. Open pywapor_v343 and uncomment *'Setup accounts'* section (or a subset of it, only the accounts that have been created in step 2). 
   When running these lines for the first time a popup will appear to request user and password for each account. 
   This only has to be done once, can be commented for further runs.
7. Run the file

Optional: 
   - For the extraction of spatially-aggregated values (.nc files and .xlsx tables) at the polygon level:
        . Run *pywapor_postprocessing.py* filling *shp_name* and *version* for your area of interest 
   - For the extraction of timeseries for specific sites:
        . Prepare a shapefile with the sites of interes (e.g. *area_valsites.shp*)
        . Run extract_tss4sites.py by filling demo_site parameter.


## Password instructions
"NASA": Used for MODIS, SRTM, CHIRPS and MERRA2 data.
- Create an account at https://urs.earthdata.nasa.gov.
- Make sure to accept the terms of use at "Applications > Authorized Apps > Approve More Applications":
  * NASA GESDISC DATA ARCHIVE
  * LP DAAC OPeNDAP

"TERRA": Used for TERRA (VITO:PROVA-V).
> Create an account at https://viewer.terrascope.be.""",

"ECMWF": Used for ERA5.
> Create an account at https://cds.climate.copernicus.eu.
  * On your profile page, scroll to the "API key" section.
  * Accept conditions when running `setup("ECMWF")` for the first time.""",
> You will be required to introduce UID and API Key later 
Accept the terms and conditions of Copernicus
https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products

"EARTHEXPLORER": Used for LANDSAT.
> Create an account at https://earthexplorer.usgs.gov,

"COPERNICUS_DATA_SPACE": """> Used for 'SENTINEL2' and SENTINEL3.
> Create an account at https://dataspace.copernicus.eu,

"VIIRSL1": Used for VIIRSL1
> Normally you do not need this account: it is only used when using the requests.get download method, which is not the default!
Create an account at https://ladsweb.modaps.eosdis.nasa.gov/.
In the top right, press Login > Generate Token

## Technical Notes

Detailed technical notes on the algorithms used are available at the [SOS-Water - WP3 Earth Observation repository](https://github.com/mibrechb/SOSW_WP3).

## Disclaimer
Views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union or CINEA. Neither the European Union nor the granting authority can be held responsible for them.

## Acknowledgement of funding
<table style="border: none;">
  <tr>
    <td><img src="https://github.com/mibrechb/SOSW_WP3/blob/main/imgs/eucom_logo.png" alt="EU Logo" width="100"/></td>
    <td>This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101059264.</td>
  </tr>
</table>
