from src.sample import Sampler
from src.viewshed_analysis.line_of_sight import RayHandler
from src.viewshed_analysis.reference_planes import ReferencePlanes
import numpy as np


def compare_accuracy(viewshed, reference, chunk):
    mid = chunk / 2
    rad_square = mid ** 2
    # False Positive | False Negative
    fp, fn = 0, 0

    # True Positive | True Negative
    tp, tn = 0, 0

    total_p, total_n = 0, 0

    for i in range(reference.shape[0]):
        for j in range(reference.shape[1]):
            length = abs(i - mid) ** 2 + abs(j - mid) ** 2
            if length > rad_square:
                continue

            if viewshed[i][j] == reference[i][j] and reference[i][j] == 1:
                tp += 1

            elif viewshed[i][j] == reference[i][j] and reference[i][j] == 0:
                tn += 1

            elif viewshed[i][j] != reference[i][j] and reference[i][j] == 1:
                fp += 1

            elif viewshed[i][j] != reference[i][j] and reference[i][j] == 0:
                fn += 1

            if reference[i][j] == 1:
                total_p += 1
            else:
                total_n += 1

    return fp, fn, tp, tn, total_p, total_n


if __name__ == '__main__':
    locs =  [(49.14505835421747, 4.526987387501593)
            ]

    # Viewshed analysis
    #lat, lon = 42.20990134363223, 20.95536858665212
    chunk = 500  # size of chunk

    # Sample
    sampler = Sampler(r'../data/EU_DTM_be.vrt')

    algorithms = ["reference_planes", "r2", "r3", "r2_ridge", "r3_ridge"]
    algorithm = algorithms[1]



    for lat, lon in locs:
        sample_data = sampler.sample(lat, lon, chunk)

        # Accuracy is based on R3
        r3 = RayHandler(sample_data, sampler.get_gt())
        r3.start_handler_r3()
        r3.save_result_as_img()
        reference = r3.get_viewshed()

        match algorithm:
            case "reference_planes":
                viewshed_analysis = ReferencePlanes(sample_data)
            case "r2":
                viewshed_analysis = RayHandler(sample_data, sampler.get_gt())
                viewshed_analysis.start_handler_r2()
                viewshed_analysis.save_result_as_img()
            case "r3":
                viewshed_analysis = RayHandler(sample_data, sampler.get_gt())
                viewshed_analysis.start_handler_r3()

        viewshed = viewshed_analysis.get_viewshed()[:chunk, :chunk]

        fp, fn, tp, tn, total_p, total_n = compare_accuracy(viewshed, reference, chunk)

        print("================================================================")
        print(f"Testing for: {algorithm}")
        print(f"Location: {lat, lon}")

        print(f"FP: {fp} | FN: {fn}")
        print(f"TP: {tn} | TN: {tn}")

        v_1 = np.count_nonzero(viewshed == 1)
        v_0 = np.count_nonzero(viewshed == 0)

        print(f"Reference: Total P: {total_p} | Total N: {total_n}")
        print(f"Viewshed: Total P: {v_1} | Total N: {v_0}")
        print("================================================================")