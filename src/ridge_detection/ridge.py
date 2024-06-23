import json
import os
import rasterio as rio
import matplotlib.pyplot as plot
import numpy as np
import time
import datetime
import multiprocessing
from multiprocessing import shared_memory
from osgeo import gdal
import rasterio.windows
import re

g_0 = 50  # Threshold value
PROCESSES = multiprocessing.cpu_count()


def get_highest_neighbour(start_coord, img, rad_y, rad_x):
    # Sets which pixel it starts at
    y, x = start_coord[0], start_coord[1]
    highest = img[y][x]
    current_highest = start_coord

    # Checks for edge cases
    if y == 0:
        i_range = (0, 2)
    elif y == rad_y - 1:
        i_range = (-1, 1)
    else:
        i_range = (-1, 2)

    if x == 0:
        j_range = (0, 2)
    elif x == rad_x - 1:
        j_range = (-1, 1)
    else:
        j_range = (-1, 2)

    for i in range(i_range[0], i_range[1]):
        for j in range(j_range[0], j_range[1]):
            height = img[y + i][x + j]
            # Updates the highest neighbour if its higher
            if height > highest:
                highest = height
                current_highest = (y + i, x + j)
    return current_highest


def ridge_loop(parameters):
    if parameters[2]:
        print("Starting process: {}".format(multiprocessing.current_process()))
    ranges = parameters[0]
    shared = shared_memory.SharedMemory(name=parameters[1])
    rad_y, rad_x = parameters[3], parameters[4]
    img = np.ndarray((rad_y, rad_x), dtype=np.float32, buffer=shared.buf)
    # Sets the ranges for i and j
    start_i, stop_i, start_j, stop_j = ranges[0], ranges[1], ranges[2], ranges[3]
    i_range = stop_i - start_i
    i_range_step = int(i_range / 10)
    coord_array = np.zeros((i_range, stop_j - start_j))
    coord_counter = {}
    for i in range(start_i, stop_i):
        if i % i_range_step == 0 and i != start_i and parameters[2]:
            print("Process: {}. {}% done".format(multiprocessing.current_process(), (i - start_i) / i_range * 100))
        for j in range(start_j, stop_j):
            current_coord = (i, j)
            highest_neighbour = get_highest_neighbour((i, j), img, rad_y, rad_x)
            # If the current coord equals the highest neighbour, then the current coord is the highest
            while current_coord != highest_neighbour:
                # If the current process finds a point which is outside its limits, we need to save these points
                if start_i <= highest_neighbour[0] < stop_i:
                    if coord_array[highest_neighbour[0] % i_range, highest_neighbour[1]] > g_0:
                        break
                    coord_array[highest_neighbour[0] % i_range, highest_neighbour[1]] += 1
                else:
                    coord_counter[highest_neighbour] = coord_counter.get(highest_neighbour, 0) + 1
                    if coord_counter[highest_neighbour] > g_0 + 1:
                        break
                # Updates the current coord and then finds its highest neighbour
                current_coord = highest_neighbour
                highest_neighbour = get_highest_neighbour(highest_neighbour, img, rad_y, rad_x)
    # Returns coords outside its limits, the correctly counted parts, and the processes starting index for sorting later
    if parameters[2]:
        print("Process: {}. 100.0% done".format(multiprocessing.current_process()))
    return coord_counter, coord_array, start_i


def load_image(img_path, start_x, start_y, rad_x, rad_y, load_whole=False):
    with rio.open(img_path) as src:
        # Window(start x pixel, start y pixel, length x, length y)
        if load_whole:
            img = src.read(1).astype(np.float32)
        else:
            img = src.read(1, window=rasterio.windows.Window(start_x, start_y, rad_x, rad_y)).astype(np.float32)
        img[img == src.nodata] = 0  # Convert NoData to 0

    shared_mem = shared_memory.SharedMemory(create=True, size=img.nbytes)
    np_array = np.ndarray(img.shape, dtype=np.float32, buffer=shared_mem.buf)
    np_array[:] = img[:]
    print("Image loaded with shape {}, {}".format(img.shape[0], img.shape[1]))
    return shared_mem


def steepest_ascent_method(filename, generate_image=False, print_progress=True, is_eudem=False, write_data=False, write_tif=True, start_range=(0, 0), rad=(2000, 2000), part_id=None):
    if is_eudem:
        path = os.path.join(os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir)), "data/eudem/{}.TIF".format(filename))
    else:
        # Has shape (152000, 188000)
        path = os.path.join(os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir)), "data/EU_DTM_be/dtm_elev.lowestmode_gedi.eml_mf_30m_0..0cm_2000..2018_eumap_epsg3035_v0.3_OT.tif")

    start_x, start_y = start_range[0], start_range[1]
    rad_x, rad_y = rad[0], rad[1]
    shared_img = load_image(path, start_x, start_y, rad_x, rad_y, False)

    # Handles the ranges for each process and how many
    parameters = []
    process_range = int(rad_y / PROCESSES)
    for i in range(PROCESSES):
        index = i / PROCESSES
        range_start_y = int(rad_y * index)
        ranges = [range_start_y, range_start_y + process_range, 0, rad_x]
        parameters.append((ranges, shared_img.name, print_progress, rad_y, rad_x))

    # Starts the Processes
    print("Starting {}, with {} processes".format(filename, PROCESSES))
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(ridge_loop, parameters)

    shared_img.close()
    shared_img.unlink()
    print("Processes are done!")

    # Divides results into coords and count parts
    coord_array = []
    count_parts = [None] * PROCESSES
    for res in results:
        coord_array.append(res[0])
        count_parts[int((res[2] / (rad_y / PROCESSES)))] = res[1]

    # Merges the count parts and adds the remaining counts
    count = np.concatenate(count_parts)
    for arr in coord_array:
        for coord in arr:
            count[coord[0], coord[1]] += arr[coord]

    end_time = time.time()
    print("It took {}".format(str(datetime.timedelta(seconds=(end_time - start_time)))))

    if write_data:
        print("Starting data write")
        data_path = os.path.join(os.getcwd(), "ridge_data/{}.txt".format(filename))
        with open(data_path, "w") as filehandle:
            json.dump(count.tolist(), filehandle)
        print("Writing data done!")

    for y in range(rad_y):
        for x in range(rad_x):
            if count[y][x] >= g_0:
                count[y][x] = 0
            else:
                count[y][x] = 1

    if write_tif:
        start_time_write = time.time()
        print("Starting TIF write")
        ds = gdal.Open(path)
        gt_dem = gdal.Open(path).GetGeoTransform()
        temp_1 = gt_dem[0] + start_x * gt_dem[1]
        temp_2 = gt_dem[3] - start_y * gt_dem[1]
        gt = (temp_1, gt_dem[1], gt_dem[2], temp_2, gt_dem[4], gt_dem[5])
        data_tiff = np.asarray(count)
        [rows, cols] = data_tiff.shape
        driver = gdal.GetDriverByName("GTiff")
        out_path = os.path.join(os.getcwd(), "ridge_data/{}.tif".format(filename))
        out_data = driver.Create(out_path, cols, rows, 1, gdal.GDT_UInt16)
        out_data.SetGeoTransform(gt)
        out_data.SetProjection(ds.GetProjection())
        out_data.GetRasterBand(1).WriteArray(data_tiff)
        out_data.FlushCache()
        end_time_write = time.time()
        print("Writing TIF done! It took {}".format(str(datetime.timedelta(seconds=(end_time_write - start_time_write)))))

    if generate_image:
        vmin, vmax = np.nanpercentile(count, (0, 100))
        plot.imshow(count, cmap='gray', vmin=vmin, vmax=vmax)
        image_path = os.path.join(os.getcwd(), "images/{}.png".format(filename))
        plot.savefig(image_path, dpi=1000, bbox_inches='tight')
        print("Image generation done!")


if __name__ == "__main__":
    test = False
    eudem_data = False
    if test:
        steepest_ascent_method("Test", True, False, False, True, False,(188000 * 0.75, 152000 * 0.55), (500, 500))
    elif eudem_data:
        folder = os.path.join(os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir)), "data/eudem/")
        files = os.listdir(folder)
        data = []
        for f in files:
            if re.search(".TIF$", f):
                data.append(re.sub(".TIF", "", f))

        start_all = time.time()
        for d in data:
            steepest_ascent_method(d, True, True, True, True, True)

        end_all = time.time()
        print("Over all it took {}".format(str(datetime.timedelta(seconds=(end_all - start_all)))))
    else:
        # EU_DTM_be has shape (y = 152000, x = 188000) 9 hours 1-10
        # Divided x_parts = 2, y_parts = 4:
        #           1         |          2
        # --------------------|--------------------
        #           3         |          4
        # --------------------|--------------------
        #           5         |          6
        # --------------------|--------------------
        #           7         |          8
        x_parts, y_parts = 4, 5
        part = 0
        start_all = time.time()
        r = (int(188000 / x_parts), int(152000 / y_parts))
        for i in range(y_parts):
            for j in range(x_parts):
                if (i == 0 and j == 0) or (i == 2 and j == 0):
                    continue
                x = r[0] * j
                y = r[1] * i
                start = (x, y)
                start_t = time.time()
                print(start)
                print(r)
                print("Starting part {}".format(part))
                steepest_ascent_method("EU_DTM_be_testS{}".format(part), False, True, False, False, True, start, r, part)
                end_t = time.time()
                print("Finished part {}, it took: {}".format(part, str(datetime.timedelta(seconds=(end_t - start_t)))))
                print("\n---------------------------------------------------------\n")
                part += 1

        end_all = time.time()
        print("Over all it took {}".format(str(datetime.timedelta(seconds=(end_all - start_all)))))


