from osgeo import gdal, osr
import numpy as np

class Sampler(object):
    """Sample height information around a point"""

    def __init__(self, raster_file):
        gdal.UseExceptions()

        # open raster & check if exists, then get spatial reference
        self.ds = gdal.Open(raster_file)

        sr_raster = osr.SpatialReference(self.ds.GetProjection())

        # WGS84 spatial reference
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)  # WGS84

        # coordinate transformation
        self.ct = osr.CoordinateTransformation(sr, sr_raster)

        # geotransformation & inverse
        gt = self.ds.GetGeoTransform()
        det = (gt[1] * gt[5] - gt[2] * gt[4])
        gt_inv = (gt[0], gt[5] / det, -gt[2] / det, gt[3], -gt[4] / det, gt[1] / det)

        self.gt = gt
        self.gt_inv = gt_inv

        # get raster band - DEMs usually have 1 band for height values
        self.rb = self.ds.GetRasterBand(1)

        # get raster size
        self.width = self.ds.RasterXSize
        self.height = self.ds.RasterYSize

    def sample(self, lon, lat, chunk):

        # transform coordinates to dataset's spatial reference
        x_geo, y_geo, _ = self.ct.TransformPoint(lon, lat)

        # convert it to pixel/line on band
        u = x_geo - self.gt_inv[0]
        v = y_geo - self.gt_inv[3]

        xpix = int(self.gt_inv[1] * u + self.gt_inv[2] * v)
        ylin = int(self.gt_inv[4] * u + self.gt_inv[5] * v)

        data = self.ds.ReadAsArray(
            max(0, xpix - chunk // 2),
            max(0, ylin - chunk // 2),
            min(self.width - 1, chunk),
            min(self.height - 1, chunk),
            buf_type=gdal.GDT_Float32)

        # turn nan or negative values into 0
        data = np.nan_to_num(data)
        data[data < 0] = 0

        return data

    def get_gt(self):
        return self.gt

    def get_ct(self):
        return self.ct

    def set_ct(self, ct):
        self.ct = ct

if __name__ == "__main__":
    from rasterio.plot import show
    import matplotlib.pyplot as plt

    lat, lon = 61.632254641313835, 8.308715972906546  # Galdhøpiggen
    chunk = 100  # size of chunk

    s = Sampler(r'../data/EU_DTM_be.vrt')
    data = s.sample(lat, lon, chunk)

    fig, ax = plt.subplots()
    show(data, ax=ax, cmap="terrain")
    cax = ax.imshow(data, cmap="terrain")

    # colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Elevation')

    plt.figtext(0.5, 0.93, f"Latitude: {lat:.5f}, Longitude: {lon:.5f}", ha='center', fontsize=12)
    plt.figtext(0.02, 0.5, f"Sample Size: {chunk}", ha='left', fontsize=12, rotation='vertical', va='center')

    plt.show()
    #plt.savefig(fname="sample_3", dpi=1000, bbox_inches='tight')
    #show(s.sample(lat, lon, chunk), cmap="terrain")
