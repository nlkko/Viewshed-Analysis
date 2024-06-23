from osgeo import gdal
import glob

demList = glob.glob("ridge_data/New_EU_DTM_be[0-9].tif")
demList = demList + glob.glob("ridge_data/New_EU_DTM_be[0-9][0-9].tif")
print(demList)

for dem in demList:
    d = gdal.Open(dem)

vrt = gdal.BuildVRT("merged.vrt", demList)
gdal.Translate("ridge_data/EU_DTM_be_Ridges.tif", vrt, xRes=30, yRes=-30)
vrt = None
