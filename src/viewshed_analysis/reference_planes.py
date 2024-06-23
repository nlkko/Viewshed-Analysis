import math
import numpy as np
from skspatial.objects import Line, Plane
import matplotlib.pyplot as plt
import multiprocessing
import time
import datetime


class ReferencePlanes(object):

    def inside_chunk(self, x, y):
        if x < 0: return False
        if y < 0: return False
        if x > self.chunk - 1: return False
        if y > self.chunk - 1: return False
        return True

    def sector_boundary(self, sector, row, col):

        # viewpoint
        s = np.array([row, col, self.r[row, col]])

        row_inc = 0
        col_inc = 0

        # N
        if sector == 0:
            row_inc = -1
            col_inc = 0

        # NE
        elif sector == 1:
            row_inc = -1
            col_inc = 1

        # E
        elif sector == 2:
            row_inc = 0
            col_inc = 1

        # SE
        elif sector == 3:
            row_inc = 1
            col_inc = 1

        # S
        elif sector == 4:
            row_inc = 1
            col_inc = 0

        # SW
        elif sector == 5:
            row_inc = 1
            col_inc = -1

        # W
        elif sector == 6:
            row_inc = 0
            col_inc = -1

        # NW
        elif sector == 7:
            row_inc = -1
            col_inc = -1

        prev_i = row + row_inc
        prev_j = col + col_inc

        i = row + (row_inc * 1)
        j = col + (col_inc * 1)

        ins = self.inside_chunk(i, j)

        d = np.array([0, 0, 0])
        prev_d = np.array([prev_i, prev_j, self.r[prev_i, prev_j]])

        while ins:
            d = np.array([i, j, self.ds[i, j]])
            z_axis = np.array([0, 0, 1])

            v = d - s  # get vector from viewpoint to d
            v_angle = np.arccos(np.dot(v, z_axis) / (np.linalg.norm(v) * np.linalg.norm(z_axis)))

            prev_v = prev_d - s
            prev_v_angle = np.arccos(np.dot(prev_v, z_axis) / (np.linalg.norm(prev_v) * np.linalg.norm(z_axis)))

            if prev_v_angle > v_angle:  # d is visible
                # print(f'visible:{prev_d}')
                self.v[i][j] = 1
                self.r[i][j] = self.ds[i][j]
                prev_d = d

            else:  # d is not visible
                # print(f'not visible:{prev_d}')
                prev_line = Line.from_points(s, s + prev_v)
                z_line = Line.from_points(d, d + z_axis)

                intersection = prev_line.intersect_line(z_line)
                self.r[i][j] = intersection[2]
                prev_d = np.array([d[0], d[1], intersection[2]])

            i += row_inc
            j += col_inc
            ins = self.inside_chunk(i, j)

    def sector_initialization(self, sector, row, col):
        # gets the indices for each sector with origin around the viewpoint

        # starting point
        i, j = 0, 0

        order = []

        ins = True

        # N - NW
        if sector == 0:
            while ins:
                modified = False  # bool flag to check whether anything has been modified
                i -= 1

                for j in range(-1, i, -1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False

        # N - NE
        elif sector == 1:
            while ins:
                modified = False
                i -= 1

                for j in range(1, -i, 1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False

        # E - NE
        elif sector == 2:
            while ins:
                modified = False
                j += 1

                for i in range(-1, -j, -1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False



        # S - SE
        elif sector == 3:
            while ins:
                modified = False
                j += 1

                for i in range(1, j, 1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False

        # S - SE
        elif sector == 4:
            while ins:
                modified = False
                i += 1

                for j in range(1, i, 1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False

        # S - SW
        elif sector == 5:
            while ins:
                modified = False
                i += 1

                for j in range(-1, -i, -1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False

        # W - SW
        elif sector == 6:
            while ins:
                modified = False
                j -= 1

                for i in range(1, -j, 1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False


        # W - NW
        elif sector == 7:
            while ins:
                modified = False
                j -= 1

                for i in range(-1, j, -1):
                    if self.inside_chunk(row + i, col + j):
                        order.append((row + i, col + j))
                        modified = True

                    if not modified:
                        ins = False

        return order

    def sector_inner(self, parameters):
        sec, row, col = parameters[0], parameters[1], parameters[2]
        return_arr = []
        for sector in sec:
            # points to make the reference plane from
            row_i, row_j = 0, 0
            col_i, col_j = 0, 0

            s = np.array([row, col, self.r[row, col]])

            if sector == 0:
                row_i = 1
                row_j = 1
                col_i = 1
                col_j = 0

            elif sector == 1:
                row_i = 1
                row_j = 0
                col_i = 1
                col_j = -1

            elif sector == 2:
                row_i = 1
                row_j = -1
                col_i = 0
                col_j = -1

            elif sector == 3:
                row_i = 0
                row_j = -1
                col_i = -1
                col_j = -1

            elif sector == 4:
                row_i = -1
                row_j = -1
                col_i = -1
                col_j = 0

            elif sector == 5:
                row_i = -1
                row_j = 0
                col_i = -1
                col_j = 1

            elif sector == 6:
                row_i = -1
                row_j = 1
                col_i = 0
                col_j = 1

            elif sector == 7:
                row_i = 0
                row_j = 1
                col_i = 1
                col_j = 1

            order = self.sector_initialization(sector, row, col)

            v = []

            for index in order:
                i, j = index[0], index[1]

                d = np.array([i, j, self.ds[i, j]])
                z_axis = np.array([0, 0, 1])
                z_line = Line.from_points(d, d + z_axis)

                row_point = np.array([i + row_i, j + row_j, self.r[i + row_i][j + row_j]])
                col_point = np.array([i + col_i, j + col_j, self.r[i + col_i][j + col_j]])

                # print(row_point, col_point)

                plane = Plane.from_points(s, row_point, col_point)
                intersection = plane.intersect_line(z_line)

                # visible
                if d[2] > intersection[2]:
                    self.r[i, j] = d[2]
                    v.append((i, j))
                else:
                    self.r[i, j] = intersection[2]

            return_arr.extend(v)
        return return_arr

    def __init__(self, ds, processes=8, filename="viewshed_wang2000"):
        self.ds = ds  # DEM dataset

        self.v = np.zeros(self.ds.shape)  # viewshed
        self.height = 1.8  # height added to the viewpoint

        self.chunk = self.ds.shape[0]
        centre = int(math.floor(self.chunk / 2))

        # initialize reference plane
        self.r = np.zeros(self.ds.shape)  # auxiliary grid / reference grid / reference plane

        # assume viewpoint is visible
        self.v[centre, centre] = 1
        self.r[centre, centre] = self.ds[centre, centre] + self.height  # middle

        for i in range(8):
            self.sector_boundary(i, centre, centre)

        params = []
        sectors_per_process = int(8 / processes)
        for i in range(0, 8, sectors_per_process):
            param = []
            for j in range(sectors_per_process):
                param.append(i + j)
            params.append((param, centre, centre))

        start = time.time()
        with multiprocessing.Pool() as pool:
            result = pool.map(self.sector_inner, params)

        end = time.time()
        self.time = datetime.timedelta(seconds=(end - start))
        print("All processes took {}".format(str(self.time)))

        for res in result:
            for point in res:
                self.v[point[0]][point[1]] = 1

        # vmin, vmax = np.nanpercentile(self.v, (0, 100))
        # plt.imshow(self.v, cmap='gray', vmin=vmin, vmax=vmax)
        # img_path = r"images/{}.png".format(filename)
        # plt.savefig(img_path, dpi=1000, bbox_inches='tight')

    def get_viewshed(self):
        return self.v

    def get_time(self):
        return self.time.total_seconds()


if __name__ == '__main__':
    from src.sample import Sampler

    lat, lon = 61.632254641313835, 8.308715972906546
    chunk = 200  # size of chunk

    s = Sampler(r'../../data/EU_DTM_be.vrt')
    sample_data = s.sample(lat, lon, chunk)
    # show(sample_data, cmap="terrain")

    wang2000 = ReferencePlanes(sample_data)
    # wang2000.get_viewshed()

    # Plot
    plt.imshow(wang2000.get_viewshed(), cmap='gray')
    plt.title("Viewshed Analysis")
    plt.show()
