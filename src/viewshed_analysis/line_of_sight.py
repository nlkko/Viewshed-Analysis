import numpy as np
import math
import time
import datetime
import multiprocessing
from osgeo import gdal
import matplotlib.pyplot as plot


def line_of_sight_at_angle(row_start, col_start, degree, rad):
    points = []
    c = round(math.cos(degree), 4)
    a = round(math.sin(degree), 4)
    step = (a, c)
    step_length = math.sqrt(a * a + c * c)
    current = (row_start, col_start)
    steps = 0
    threshold = rad / step_length

    while steps < threshold:
        old = current
        current = tuple(np.add(current, step))
        row, col = round(current[0]), round(current[1])

        # Sometimes the new step will still lead to the same pixel as the previous step
        if not (row == round(old[0]) and col == round((old[1]))):
            steps += 1
            points.append((row, col))

    return points


def bresenham(row_start, col_start, row_end, col_end):
    dx = abs(col_end - col_start)
    sx = 1 if col_start < col_end else -1
    dy = -abs(row_end - row_start)
    sy = 1 if row_start < row_end else -1
    error = dx + dy
    points = []

    while True:
        points.append((row_start, col_start))
        e2 = 2 * error
        if e2 >= dy:
            if col_start == col_end:
                break
            error += dy
            col_start += sx
        if e2 <= dx:
            if row_start == row_end:
                break
            error += dx
            row_start += sy

    return points


def append_to_circle(y, x, points, col, row):
    points.append((col + y, row + x))
    points.append((col + y, row - x))
    points.append((col - y, row + x))
    points.append((col - y, row - x))
    points.append((col + x, row + y))
    points.append((col + x, row - y))
    points.append((col - x, row + y))
    points.append((col - x, row - y))


def circle(radius, row, col):
    rs2 = radius * radius * 4
    xs2 = 0
    ys2m1 = rs2 - 2 * radius + 1
    x = 0
    y = radius
    points = []
    append_to_circle(y, x, points, col, row)

    while x <= y:
        xs2 = xs2 + 8 * x + 4
        x += 1
        ycs2 = rs2 - xs2
        if ycs2 < ys2m1:
            ys2m1 = ys2m1 - 8 * y + 4
            y -= 1
        append_to_circle(y, x, points, col, row)

    return points


class RayHandler(object):
    def __init__(self, ds, gt, ridges=None, image_filename="viewshed", view_height=1.8,
                 processes=(multiprocessing.cpu_count() - 2), curvature=True, refraction=0.13, R3_range_middle=True):
        # chunk_size - starting_point * 2 > 0
        self.chunk = ds.shape[0]
        if self.chunk % 2 == 0:
            self.chunk += 1
        self.middle = int(math.floor(self.chunk / 2))
        self.starting_point = (self.middle, self.middle)
        self.filename = image_filename

        self.resolution = gt[1]
        self.data = ds
        self.starting_height = self.data[self.starting_point[1], self.starting_point[0]] + view_height

        if ridges is None:
            self.ridges = []
            self.use_ridges = False
        else:
            self.ridges = ridges
            self.use_ridges = True

        self.processes = processes if processes > 0 else 1
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.curvature = curvature
        self.refraction = refraction
        self.max_value = self.data.max()
        self.rays = []
        self.visible_points = np.zeros(self.data.shape, dtype=int)
        self.visible_points[self.starting_point[1]][self.starting_point[0]] = 1
        self.algorithm = ""
        self.time = 0
        self.possible_visible = []
        self.ridge_angle_delta = {self.starting_point: (-math.pi / 2, 10000)}
        self.R3_range_middle = R3_range_middle

    def start_all_rays_r3(self, start_stop):
        rays = []
        range_rays = []
        radius_square = self.middle ** 2
        for row in range(start_stop[0], start_stop[1]):
            for col in range(self.cols):
                if (row, col) == self.starting_point:
                    continue
                length = abs(row - self.middle) ** 2 + abs(col - self.middle) ** 2
                if length > radius_square:
                    continue

                if self.use_ridges and self.ridges[row][col]:
                    points = bresenham(self.starting_point[0], self.starting_point[1], row, col)
                    amount_new_points = len(points) * math.ceil(self.middle / len(points))
                    for i in range(amount_new_points):
                        new_point = row + points[i + 1][0] - self.middle, col + points[i + 1][1] - self.middle
                        points.append(new_point)

                    temp_ray = RayR3(points, self, (row, col))
                    if self.R3_range_middle:
                        temp_ray.find_ridge_range2()
                    else:
                        temp_ray.find_ridge_range()
                    range_rays.append(temp_ray)
                else:
                    points = bresenham(self.starting_point[0], self.starting_point[1], row, col)
                    temp_ray = RayR3(points, self, (row, col))
                    temp_ray.loop()
                    rays.append(temp_ray)
        return rays, range_rays

    def start_point_rays_r3(self, start_stop):
        rays = []
        for i in range(start_stop[0], start_stop[1]):
            point = self.possible_visible[i][0]
            starting_point = self.possible_visible[i][1]
            angle_delta = self.ridge_angle_delta.get(starting_point)
            points = bresenham(starting_point[0], starting_point[1], point[0], point[1])
            ray = RayR3(points, self, point)
            ray.set_angle_and_delta(angle_delta[0], angle_delta[1])
            ray.loop()
            rays.append(ray)
        return rays, []

    def start_rays_r2(self, parameters):
        (start, stop) = parameters
        rays = []
        for i in range(start, stop):
            point = self.possible_visible[i]
            points = bresenham(self.starting_point[0], self.starting_point[1], point[0], point[1])
            temp_ray = RayR2(points, self)
            if self.use_ridges:
                temp_ray.find_possible_visible()
                temp_ray.loop_again()
            else:
                temp_ray.loop()
            rays.append(temp_ray)
        return rays

    def start_processes(self, n_items, func, r2=False):
        # Finds the range of items each process has to calculate
        process_range = int(n_items / self.processes)
        ranges = []
        excess_rows = n_items - process_range * self.processes
        for process in range(self.processes):
            # If there are any excess items we need to add those
            current_range = process_range + 1 if process < excess_rows else process_range

            if process == 0:
                ranges.append((0, current_range))
            else:
                previous = ranges[process - 1][1]
                ranges.append((previous, previous + current_range))

        with multiprocessing.Pool() as pool:
            result = pool.map(func, ranges)

        if r2:
            for array in result:  # For each process
                for ray in array:  # For each ray
                    for point in ray.get_points():  # For the points in ray
                        if not self.visible_points[point[0]][point[1]]:
                            self.visible_points[point[0]][point[1]] = 1
        else:
            rays = []
            range_rays = []
            # Separates the result
            for res in result:
                rays.extend(res[0])
                range_rays.extend(res[1])

            if self.use_ridges and len(self.possible_visible) == 0:
                # Loops through and checks if ridges are visible
                for ray in rays:
                    self.rays.append(ray)
                    visible, point = ray.get_visible_point()
                    if visible:
                        self.visible_points[point[0]][point[1]] = 1
                    self.ridge_angle_delta[point] = ray.get_angle_and_delta()

                # Loops through and checks the ranges for possible visible points
                for range_ray in range_rays:
                    ridge_info, point = range_ray.get_ridge_range()
                    start_range = ridge_info[0]
                    end_range = ridge_info[1]
                    end_is_ridge = ridge_info[2]
                    if not end_is_ridge:  # If last point is not a ridge, its visibility is not found yet
                        self.possible_visible.append((point, self.starting_point))
                        continue
                    start_visible = bool(self.visible_points[start_range[0]][start_range[1]])
                    end_visible = bool(self.visible_points[end_range[0]][end_range[1]])
                    if not start_visible and not end_visible:
                        # Point is only skipped if start and end of range is not visible
                        continue
                    self.possible_visible.append((point, ridge_info[0]))  # Adds starting point for they ray

            else:
                for ray in rays:
                    self.rays.append(ray)
                    visible, point = ray.get_visible_point()
                    if visible:
                        self.visible_points[point[0]][point[1]] = 1

    def start_handler_r3(self):
        start_all = time.time()
        self.start_processes(self.rows, self.start_all_rays_r3)

        if self.use_ridges:
            self.start_processes(len(self.possible_visible), self.start_point_rays_r3)

        end_all = time.time()
        self.time = datetime.timedelta(seconds=(end_all - start_all))
        print("All processes for R3 took {}".format(str(self.time)))

        if self.use_ridges:
            self.algorithm = "R3_ridge"
        else:
            self.algorithm = "R3"

    def start_handler_r2(self):
        start_all = time.time()

        self.possible_visible = circle(self.middle - 1, self.starting_point[0], self.starting_point[1])
        self.start_processes(len(self.possible_visible), self.start_rays_r2, r2=True)

        end_all = time.time()
        self.time = datetime.timedelta(seconds=(end_all - start_all))
        print("All processes for R2 took {}".format(str(self.time)))

        if self.use_ridges:
            self.algorithm = "R2_ridge"
        else:
            self.algorithm = "R2"

    def save_result_as_img(self):
        if self.filename == "viewshed":
            self.filename += self.algorithm
        v_data = self.get_viewshed()
        vmin, vmax = np.nanpercentile(v_data, (0, 100))
        plot.imshow(v_data, cmap='gray', vmin=vmin, vmax=vmax)
        img_path = r"images/{}.png".format(self.filename)
        plot.savefig(img_path, dpi=1000, bbox_inches='tight')

    # Code inspired by https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    def get_viewshed(self):
        return self.visible_points

    def get_rays(self):
        return self.rays

    def get_data(self):
        return self.data

    def get_time(self):
        return self.time.total_seconds()


class RayR2(object):
    def __init__(self, points, handler):
        self.points = points  # In Rad
        self.handler = handler
        self.highest_angle = -math.pi / 2  # smallest possible angle
        self.lowest_delta = 10000  # lower delta means higher point
        self.refraction_value = 1 - self.handler.refraction
        self.max_value = handler.max_value
        self.visible = []

        self.possible_visible = []
        self.range = []
        self.start_of_range_visible = True
        self.last = points[-1]

    def loop(self):
        for point in self.points:
            if point == self.handler.starting_point:
                continue

            (row, col) = point[0], point[1]
            current_height = self.handler.data[row][col]
            length = math.sqrt(abs(row - self.handler.middle) ** 2 + abs(col - self.handler.middle) ** 2)
            delta_length = length * self.handler.resolution
            if self.handler.curvature:
                current_height = current_height - (delta_length ** 2 / 12742000) * self.refraction_value

            delta_height = self.handler.starting_height - current_height
            if delta_height >= self.lowest_delta and self.lowest_delta < 0:
                continue

            distance = math.sqrt(delta_height * delta_height + delta_length * delta_length)
            min_d_height = math.sin(self.highest_angle) * distance
            if min_d_height > self.max_value:
                return

            if delta_height > 0:
                angle = 0 - math.acos(delta_length / distance)
            else:
                angle = math.acos(delta_length / distance)
            visible = angle > self.highest_angle

            if visible:
                self.highest_angle = angle
                self.lowest_delta = delta_height
                self.visible.append(point)

    def find_possible_visible(self):
        for point in self.points:
            if point == self.handler.starting_point:
                continue
            (row, col) = point[0], point[1]
            # Checks if point is on a ridge or last (False means it's a ridge)
            if not bool(self.handler.ridges[row][col]) or point == self.last:
                if self.start_of_range_visible:  # If the start of the range was visible it is allways added
                    self.possible_visible.extend(self.range)
                    self.range = []

                current_height = self.handler.data[row][col]
                length = math.sqrt(abs(row - self.handler.middle) ** 2 + abs(col - self.handler.middle) ** 2)
                delta_length = length * self.handler.resolution
                if self.handler.curvature:
                    current_height = current_height - (delta_length ** 2 / 12742000) * self.refraction_value

                delta_height = self.handler.starting_height - current_height
                if delta_height >= self.lowest_delta and self.lowest_delta < 0:
                    self.range = []
                    self.start_of_range_visible = False

                distance = math.sqrt(delta_height * delta_height + delta_length * delta_length)
                min_d_height = math.sin(self.highest_angle) * distance
                if min_d_height > self.max_value:
                    return

                if delta_height > 0:
                    angle = 0 - math.acos(delta_length / distance)
                else:
                    angle = math.acos(delta_length / distance)
                visible = angle > self.highest_angle

                if visible:  # End of range visible
                    self.lowest_delta = delta_height
                    self.highest_angle = angle
                    self.start_of_range_visible = True
                    self.range.append((row, col))
                    self.possible_visible.extend(self.range)
                    if point == self.last:
                        self.possible_visible.append(point)
                    self.range = []
                else:  # End of range not visible
                    self.range = []
                    self.start_of_range_visible = False
            else:
                #  Adds point to range, will be removed if the range is not visible
                self.range.append((row, col))

    def loop_again(self):
        self.lowest_delta = 10000
        self.highest_angle = -math.pi / 2
        self.points = self.possible_visible
        self.loop()

    def get_points(self):
        return self.visible

    def get_possible_visible(self):
        return self.possible_visible


class RayR3(object):
    def __init__(self, points, handler, target):
        self.points = points
        self.handler = handler
        self.highest_angle = -math.pi / 2  # smallest possible angle
        self.lowest_delta = 10000  # lower delta means higher point
        self.refraction_value = 1 - self.handler.refraction
        self.max_value = handler.max_value
        self.visible = False
        self.target = target

        self.ridge_range = [points[0]]
        self.in_range = False
        self.last = self.points[-1]

    def loop(self):
        for point in self.points:
            if point == self.handler.starting_point:
                continue
            (row, col) = point[0], point[1]

            current_height = self.handler.data[row][col]
            length = math.sqrt(abs(row - self.handler.middle) ** 2 + abs(col - self.handler.middle) ** 2)
            delta_length = length * self.handler.resolution
            if self.handler.curvature:
                current_height = current_height - (delta_length ** 2 / 12742000) * self.refraction_value

            delta_height = self.handler.starting_height - current_height
            if delta_height >= self.lowest_delta and self.lowest_delta < 0:
                continue

            distance = math.sqrt(delta_height * delta_height + delta_length * delta_length)
            min_d_height = math.sin(self.highest_angle) * distance
            if min_d_height > self.max_value:
                return

            if delta_height > 0:
                angle = 0 - math.acos(delta_length / distance)
            else:
                angle = math.acos(delta_length / distance)
            visible = angle > self.highest_angle

            if visible:
                self.highest_angle = angle
                self.lowest_delta = delta_height
                if point == self.target:
                    self.visible = True

    def find_ridge_range(self):
        for point in self.points:
            (row, col) = point[0], point[1]

            if row >= self.handler.chunk or col >= self.handler.chunk:
                len_index = len(self.points) - 1
                for i in range(len(self.points)):
                    if point == self.points[len_index]:
                        self.ridge_range.append(point)
                        self.ridge_range.append(False)  # If end of range is a ridge
                        return
                    else:
                        len_index += -1

            if not bool(self.handler.ridges[row][col]):
                if self.in_range:
                    self.ridge_range.append(point)
                    self.ridge_range.append(True)  # If end of range is a ridge
                    return
                else:
                    self.ridge_range = [point]
            if point == self.target:
                self.in_range = True
        # Last point
        self.ridge_range.append(self.last)
        self.ridge_range.append(False)  # If end of range is a ridge

    def find_ridge_range2(self):
        index = self.points.index(self.target)
        range_start, range_end = (), ()
        check_back, check_front = True, True
        end_of_range_ridge = False

        for i in range(len(self.points)):
            if i == 0:
                continue

            if check_back:
                back_index = index + i
                back_point = self.points[back_index]
                row, col = back_point[0], back_point[1]

                if row >= self.handler.chunk or col >= self.handler.chunk:
                    range_end = self.points[back_index - 1]
                    if range_start != ():
                        break
                    check_back = False
                elif back_index == len(self.points) - 1:
                    range_end = self.target
                    if range_start != ():
                        break
                    check_back = False
                elif not bool(self.handler.ridges[row][col]):
                    range_end = back_point
                    end_of_range_ridge = True
                    if range_start != ():
                        break
                    check_back = False

            if check_front:
                front_index = index - i
                front_point = self.points[front_index]
                if front_index == 0:
                    range_start = self.points[0]
                    if range_end != ():
                        break
                    check_front = False
                elif not bool(self.handler.ridges[front_point[0]][front_point[1]]):
                    range_start = front_point
                    if range_end != ():
                        break
                    check_front = False

        self.ridge_range = (range_start, range_end, end_of_range_ridge)

    def get_visible_point(self):
        return self.visible, self.target

    def get_ridge_range(self):
        return self.ridge_range, self.target

    def get_angle_and_delta(self):
        return self.highest_angle, self.lowest_delta

    def set_angle_and_delta(self, highest_angle, lowest_delta):
        self.highest_angle = highest_angle
        self.lowest_delta = lowest_delta


if __name__ == "__main__":
    from src import sample

    gdal.UseExceptions()
    locations = [(49.14505835421747, 4.526987387501593)]
    chunks = [2000]

    # Loads data
    s = sample.Sampler(r'data/EU_DTM_be.vrt')

    R2 = True
    R3 = False
    ridges = False
    both_ranges = False

    for loc in locations:
        lat, long = loc

        for chunk in chunks:

            # Loads ridges
            if ridges:
                ridge_sample = sample.Sampler("../ridge_detection/ridge_data/EU_DTM_be_Ridges.tif")
                ridge_sample.set_ct(s.get_ct())
                ridges = ridge_sample.sample(lat, long, chunk)
            else:
                ridges = None

            ds = s.sample(lat, long, chunk)

            if R2:
                print("Running R2 {}:".format(chunk))
                h1 = RayHandler(ds, s.get_gt(), ridges, "Test_R2" + "_{}".format(chunk), 1.8, False)
                h1.start_handler_r2()
                h1.save_result_as_img()

                if ridges:
                    print("Running R2 with ridges {}:".format(chunk))
                    h2 = RayHandler(ds, s.get_gt(), ridges, "Test_R2_ridge" + "_{}".format(chunk), 1.8, True)
                    h2.start_handler_r2()
                    h2.save_result_as_img()

            if R3:
                print("Running R3 {}:".format(chunk))
                h1 = RayHandler(ds, s.get_gt(), ridges, "Test_R3" + "_{}".format(chunk), 1.8, False)
                h1.start_handler_r3()

                if ridges and chunk <= 400:
                    print("Running R3 with ridges {}:".format(chunk))
                    h2 = RayHandler(ds, s.get_gt(), ridges, "Test_R3_ridge" + "_{}".format(chunk), 1.8, True)
                    h2.start_handler_r3()

                    if both_ranges:
                        print("Running R3 with ridges and old range {}:".format(chunk))
                        h3 = RayHandler(ds, s.get_gt(), ridges, "Test_R3_ridge" + "_{}".format(chunk), 1.8, True,
                                        R3_range_middle=False)
                        h3.start_handler_r3()
