from src.dataloader import generator
import collections
from statistics import mean, stdev, variance
from math import sqrt
from scipy import stats
from tqdm import tqdm
def global_mean_detector(generator_num,threshold,data_size,limit=10000):
    gen = generator(generator_num)

    dataset = collections.deque(data_size*[None],data_size)
    while dataset[1] is None:
        dataset.append(next(gen)) #stream from generator

    global_mean = None
    global_std = None
    i = data_size
    pbar = tqdm(total=limit)
    pbar.update(i)
    change_ids = []
    means = []
    # Get new data
    data = next(gen)
    dataset.append(data)

    while dataset is not None:
        if global_mean is None and global_std is None:
            global_mean = mean(dataset)
            global_std = stdev(dataset)
        else:
            global_mean = (global_mean*(i-1)+data)/(i)
        curr_mean = mean(dataset)
        curr_std = stdev(dataset)

        curr_threshold_mean = curr_mean * threshold
        curr_threshold_std = curr_std * threshold
        means.append(curr_mean)

        diff_mean = abs(global_mean-curr_mean)#+curr_std
        diff_std = abs(curr_std-global_std)

        if diff_mean > curr_threshold_mean and diff_std > curr_threshold_std:
            global_mean = curr_mean
            global_std = curr_std
            change_ids.append(i)


        data = next(gen)
        if data is not None:
            dataset.append(data)
            pbar.update(1)
            i += 1
        else:
            dataset = None

    return change_ids, means


def Kolmogorov_based_detector(generator_num,threshold,data_size,limit=10000):
    gen = generator(generator_num)

    dataset = collections.deque(data_size * [None], data_size)
    while dataset[1] is None:
        dataset.append(next(gen))  # stream from generator

    change_ids = []
    means = []
    pvalues = []
    i = data_size
    data = next(gen)
    dataset.append(data)
    i+=1
    pbar = tqdm(total=limit)
    pbar.update(i)
    while dataset is not None:
        mean_v = mean(dataset)
        means.append(mean_v)

        kolmogorov_result = stats.ks_2samp(dataset, [data])
        pvalues.append(kolmogorov_result.pvalue)

        if kolmogorov_result.pvalue < threshold:
            change_ids.append(i)
        data = next(gen)
        if data is not None:
            dataset.append(data)
            pbar.update(1)
            i += 1
        else:
            dataset = None
            print('loop end')
            break
    print(len(change_ids),len(means),len(pvalues))
    return change_ids, means, pvalues
