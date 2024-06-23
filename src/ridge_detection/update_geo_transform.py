from osgeo import gdal
import glob
import os

EU_DTM = os.path.join(os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir)),
                            "data/EU_DTM_be/dtm_elev.lowestmode_gedi.eml_mf_30m_0..0cm_2000..2018_eumap_epsg3035_v0.3_OT.tif")
original = gdal.Open(EU_DTM)
original_GT = original.GetGeoTransform()

x_parts, y_parts = 4, 5
[cols, rows] = (int(188000 / x_parts), int(152000 / y_parts))
part = 0

demList = glob.glob("ridge_data/EU_DTM_be[0-9].tif")
demList = demList + glob.glob("ridge_data/EU_DTM_be[0-9][0-9].tif")

for i in range(len(demList)):
    dem = demList[i]
    demList[i] = dem.split("\\")[1]

for i in range(y_parts):
    for j in range(x_parts):
        if i == 2 and j == 0:
            continue
        x = cols * j
        y = rows * i
        temp_1 = original_GT[0] + x * original_GT[1]
        temp_2 = original_GT[3] - y * original_GT[1]
        gt = (temp_1, original_GT[1], original_GT[2], temp_2, original_GT[4], original_GT[5])
        temp_data_set = gdal.Open("ridge_data/" + demList[part])
        driver = gdal.GetDriverByName("GTiff")
        out_path = os.path.join(os.getcwd(), "ridge_data/New_{}".format(demList[part]))
        data_set = driver.CreateCopy(out_path, temp_data_set, strict=0)
        data_set.SetGeoTransform(gt)
        data_set.SetProjection(original.GetProjection())
        data_set.FlushCache()
        data_set = None
        temp_data_set = None
        part = part + 1
