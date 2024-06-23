from src.viewshed_analysis.line_of_sight import RayHandler
from src.viewshed_analysis.reference_planes import ReferencePlanes
from src.sample import Sampler


def test(processes, locations, chunks, R2):
    times = []
    print("Starting with {} processes".format(processes))
    # Format time[0] = (loc, [range1, R2, R2_ridge], [range2, R2, R2_ridge], ...])
    # First five costal (mountains and not), next flatish, last five mountains
    for loc in locations:
        lat, long = loc
        appendix = []
        for chunk in chunks:
            app = [chunk]

            # Loads ridges
            ridge_sample = Sampler("../ridge_detection/ridge_data/EU_DTM_be_Ridges.tif")
            ridge_sample.set_ct(s.get_ct())
            ridges = ridge_sample.sample(lat, long, chunk)

            ds = s.sample(lat, long, chunk)

            if R2:
                print("Running R2 {}:".format(chunk))
                h1 = RayHandler(ds, s.get_gt(), image_filename="Test_R2" + "_{}".format(chunk), view_height=1.8, processes=processes)
                h1.start_handler_r2()
                app.append(h1.get_time())
                print("Running R2 with ridges {}:".format(chunk))
                h2 = RayHandler(ds, s.get_gt(), image_filename="Test_R2""Test_R2_ridge" + "_{}".format(chunk), view_height=1.8, processes=processes)
                h2.start_handler_r2()
                app.append(h2.get_time())
            if False:
                print("Running R3 {}:".format(chunk))
                h1 = RayHandler(ds, s.get_gt(), image_filename="Test_R3" + "_{}".format(chunk), view_height=1.8, processes=processes)
                h1.start_handler_r3()
                app.append(h1.get_time())
            if False:
                print("Running R3 with ridges {}:".format(chunk))
                h2 = RayHandler(ds, s.get_gt(), ridges=ridges, image_filename="Test_R3_ridge" + "_{}".format(chunk), view_height=1.8, processes=processes)
                h2.start_handler_r3()
                app.append(h2.get_time())
            if False:
                print("Running R3 with ridges and old range {}:".format(chunk))
                h3 = RayHandler(ds, s.get_gt(), ridges=ridges, image_filename="Test_R3_ridge" + "_{}".format(chunk), view_height=1.8,
                                R3_range_middle=False, processes=processes)
                h3.start_handler_r3()
                app.append(h3.get_time())
            if True:
                print("Running Wang range {}".format(chunk))
                s = Sampler(r'../data/EU_DTM_be.vrt')
                sample_data = s.sample(lat, long, chunk)
                wang2000 = ReferencePlanes(sample_data, processes)
                app.append(wang2000.get_time())

            appendix.append(app)

        times.append((loc, appendix))

    return times


def average(data, n_locs, n_chunks, n_algs):
    # data = (loc, [[range1, R2, R2_ridge], [range2, R2, R2_ridge], ...])
    sum_time = [0] * n_chunks * n_algs
    for location in data:
        part = location[1]
        for i in range(n_chunks):
            for a in range(n_algs):
                sum_time[i * n_algs + a] += part[i][a + 1]

    avg = [round(x / n_locs, 5) for x in sum_time]
    return avg


if __name__ == "__main__":
    locs = [ (42.69916413585336, 0.9808615763832157),
                 (41.96518962357312, 20.344203582825383), (63.41630074310854, 10.403150440553814), (45.550031738160584, 12.527660317753767),
                 (45.252210433310715, 14.218931523655508), (38.26164486305427, 23.87948183610944),
                 (54.91673772566345, 11.80893748850242),
                 (51.8561183540767, 17.262757628112084), (47.911110573342256, 1.0325389341390763),
                 (49.14505835421747, 4.526987387501593), (52.10241124307781, 6.136666707446241),
                 (56.043145729080955, 9.268284733273365),
                 (61.563699646637765, 8.32392019959587)
                 ]
    chunks = [100, 200, 300, 400, 600, 800, 1200]
    n_processes = [1, 2, 4]

    results = []
    R2 = False
    for n in n_processes:
        results.append(test(n, locs, chunks, R2))

    res_average = []
    if R2:
        n_algorithms = 2
    else:
        n_algorithms = 1
    for r in results:
        res_average.append(average(r, len(locs), len(chunks), n_algorithms))

    file = open("data/time_data_R3_Wang.txt", "w")
    print("Start writing")
    for i in range(len(n_processes)):
        string = "Number of processes {}:\n".format(n_processes[i])
        res = res_average[i]
        for j in range(0, len(chunks) * n_algorithms, n_algorithms):
            r = chunks[int(j / n_algorithms)]
            if R2:
                string += "range: {}, R2 {}, R2_ridge: {}\n".format(r, res[j], res[j + 1])
            elif n_algorithms == 1:
                string += "range: {}, Wang {}\n".format(r, res[j])
            else:
                string += "range: {}, R3: {}, R3_ridge_middle:  R3_ridge_start: \n".format(r, res[j], res[j + 1], res[j + 2])
        string += "-----------------------------------------------------------\n"
        file.write(string)
    print("Writing data done!")
    file.close()
