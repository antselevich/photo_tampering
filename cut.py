#!/usr/bin/python

from sys import argv
from sys import path
from os import listdir
from os.path import abspath, dirname, pardir, join
from numpy import array, zeros, float16, float32

from logger import Logger
from cv2 import imread, imwrite, Sobel, CV_64F, resize
from math import exp, log
from graph import Graph
from copy import deepcopy
from time import time

from profile import run
from scipy.sparse import vstack, coo_matrix

from pp import Server

import math
from math import sqrt

from gc import collect


WEIGHT_THRESHOLD = 0.2
INTENSITY_POWER_THRESHOLD = log(WEIGHT_THRESHOLD)
VER_SUMMAND_LOWER = -INTENSITY_POWER_THRESHOLD / 10.
VER_SUMMAND_UPPER = 0.01

logger = Logger(True, None)

def scale(i, n):
    return (1. * i) / n


def truncate(x, threshold):
    if x > threshold:
        return x
    return 0.


def variance_eq_prob_n(n):
    expected = (1./n) * sum(range(n))
    return (1./n) * sum([(expected - x)**2 for x in range(n)])


def calc_weight_for_test(ver1, pix1, grad1, dup1, ver2, pix2, dup2, grad2):
    return (sum(ver1) * sum(pix1) + 0.3) * (sum(ver2) * sum(pix2) + 0.3)


def calc_intensity_weight(ver1, pix1, grad1, dup1, ver2, pix2, grad2, dup2, ver_l=0.16, ver_u=0.01, intens_pow=-1.6):
    ver_summand = (ver1[0] - ver2[0]) ** 2 + (ver1[1] - ver2[1]) ** 2
    if ver_summand > ver_l:
        return 0.
    if ver_summand < ver_u:
        ver_summand = 0.
    else:
        ver_summand *= 10
    color_summand = (pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2 + (pix1[2] - pix2[2]) ** 2
    color_summand *= 100
    power = -(color_summand + ver_summand)
    if power <= intens_pow:
        return 0.
    return math.exp(power)


def calc_texture_weight(ver1, pix1, grad1, dup1, ver2, pix2, grad2, dup2):
    return calc_intensity_weight(ver1, grad1, pix1, dup1, ver2, grad2, pix2, dup2)


def calc_duplicate_weight(ver1, pix1, grad1, dup1, ver2, pix2, grad2, dup2):
    return calc_intensity_weight(ver1, dup1, pix1, grad1, ver2, dup2, pix2, grad2)


def get_coords_for_pixel(pixel_num, width):
    return (pixel_num / width, pixel_num % width)


def get_pixel_for_coords(row, col, width):
    return row * width + col


def scale_img(img, height, width, max_r=255, max_g=255, max_b=255):
    scaled_img = zeros((height, width), dtype=tuple)
    for row in range(height):
        for col in range(width):
            scaled_img[row, col] = (scale(img[row, col][0], max_r).item(), \
                                    scale(img[row, col][1], max_g).item(), \
                                    scale(img[row, col][2], max_b).item())
    return scaled_img



def calc_weights(calc_weight, pixels, scaled_vers, img_pixels, grad_img_pixels, dup_img_pixels, weight_threshold):
    weights = []
    for pixel1 in pixels:
        ver1 = scaled_vers[pixel1]
        pix1 = img_pixels[pixel1]
        grad1 = grad_img_pixels[pixel1]
        dup1 = dup_img_pixels[pixel1]
        for pixel2 in range(pixel1):
            ver2 = scaled_vers[pixel2]
            pix2 = img_pixels[pixel2]
            grad2 = grad_img_pixels[pixel2]
            dup2 = dup_img_pixels[pixel2]
            weight = calc_weight(ver1, pix1, grad1, dup1, ver2, pix2, grad2, dup2)
            if weight > weight_threshold:
                weights.append((pixel1, pixel2, weight))
    return weights

#@profile
def scaled_img_to_graph(scaled_image, scaled_grad_image, scaled_dup_image, height, width, calc_weight, processes, weight_threshold=WEIGHT_THRESHOLD):
    size = height * width
    graph = Graph(size)
    _add_edge = graph.add_edge

    range_size = range(size)

    scaled_vers = list()
    img_pixels = list()
    grad_img_pixels = list()
    dup_img_pixels = list()
    for p in range_size:
        x, y = get_coords_for_pixel(p, width)
        scaled_vers.append((scale(x, height), scale(y, width)))
        img_pixels.append(scaled_image[x, y])
        grad_img_pixels.append(scaled_grad_image[x, y])
        dup_img_pixels.append(scaled_dup_image[x, y])


    graph.set_diag([1. for i in range_size])


    times = 30
    new_size = size / times
    for main_part in range(times):
        new_begin = main_part * new_size
        job_server = Server(ncpus=processes)
        jobs = []

        start_range = 0
        step = new_size / processes
        for end_range in range(step, new_size + step, step):
            end_range = min(end_range, new_size)
            jobs.append(job_server.submit(calc_weights, (calc_weight, range(new_begin + start_range, new_begin + end_range), scaled_vers, img_pixels, grad_img_pixels, dup_img_pixels, weight_threshold), modules=('math',), depfuncs=(calc_intensity_weight, calc_weight_for_test, calc_texture_weight, calc_duplicate_weight)))
            start_range = end_range

        while len(jobs):
            for job in jobs:
                if job.finished:
                    edges = job()
                    if edges is None:
                        job_server.destroy()
                        logger.log('Couldn\'t complete one of the jobs', True)
                        return None
                    for pixel1, pixel2, weight in edges:
                        _add_edge(pixel1, pixel2, weight)
                    jobs.remove(job)
                    del edges
                    collect()
        job_server.destroy()
    graph.ready()
    return graph


def compute_grad_img(image, height, width):
    image_Sobel_x = Sobel(image, CV_64F, 1, 0, ksize=5)
    image_Sobel_y = Sobel(image, CV_64F, 0, 1, ksize=5)
    img_grad = zeros((height, width), dtype=tuple)
    for row in range(height):
        for col in range(width):
            img_grad[row, col] = (float16(sqrt(image_Sobel_x[row, col][0] ** 2 + image_Sobel_y[row, col][0] ** 2)), \
                                    float16(sqrt(image_Sobel_x[row, col][1] ** 2 + image_Sobel_y[row, col][1] ** 2)), \
                                    float16(sqrt(image_Sobel_x[row, col][2] ** 2 + image_Sobel_y[row, col][2] ** 2)))
    return img_grad


def compute_dup_for_lines(lines_range, pixels, row_cols):
    result = list()
    pixels_size = len(pixels)
    one_pixel_size = len(pixels[0])
    range_pixels_size = range(pixels_size)
    range_one_pixel_size = range(one_pixel_size)
    for i in lines_range:
        min_val = 1E9
        pixel1 = pixels[i]
        row = row_cols[i][0]
        col = row_cols[i][1]
        min_ver = (0, 0)
        min_pixel = (0, 0, 0)
        j = 0
        for pixel2 in pixels:
            if i == j:
                j += 1
                continue
            sum_I = 0
            for p1, p2 in zip(pixel1, pixel2):
                sum_I += (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
                if sum_I >= min_val:
                    break
            if sum_I < min_val:
                min_val = sum_I
                min_ver = row_cols[j]
                min_pixel = pixel2
                if min_val == 0:
                    break
            j += 1

        if min_val < 1E9:
            result.append((row, col, min_val))
    return result


def compute_dup_img(image, height, width, processes):
    result = zeros((height, width), dtype=tuple)
    pixels = list()
    row_cols = list()
    for row in range(height):
        for col in range(width):
            pixel = list()
            row_cols.append((row, col))
            for krow in (-2, -1, 0, 1, 2):
                for kcol in (-2, -1, 0, 1, 2):
                    x = row - krow
                    y = col - kcol
                    if x >= 0 and x < height and y >= 0 and y < width:
                        pixel.append((int(image[x, y][0]), int(image[x, y][1]), int(image[x, y][2])))
                    else:
                        pixel.append((0, 0, 0))
            pixels.append(pixel)
    job_server = Server(ncpus=processes)
    jobs = []

    start_range = 0
    step = len(pixels) / processes
    for end_range in range(step, len(pixels) + step, step):
        end_range = min(end_range, len(pixels))
        jobs.append(job_server.submit(compute_dup_for_lines, (range(start_range, end_range), pixels, row_cols)))
        start_range = end_range

    while len(jobs):
        for job in jobs:
            if job.finished:
                dup_distanses = job()
                if dup_distanses is None:
                    job_server.destroy()
                    logger.log('Couldn\'t complete one of the jobs', True)
                    return None
                for row, col, value in dup_distanses:
                    result[row, col] = (float32(value), float32(value), float32(value))
                jobs.remove(job)

    return result


def create_img_with_components(component_B, height, width):
    result = zeros((height, width), dtype=int)
    for pixel in component_B:
        x, y = get_coords_for_pixel(pixel, width)
        result[x, y] = 255
    return result


def get_int(img, height, width):
    for x in range(height):
        for y in range(width):
            img[x, y] = (int(img[x, y][0]), int(img[x, y][1]), int(img[x, y][2]))


def get_maximum(image, height, width):
    maximum = 0
    for row in range(height):
        for col in range(width):
            if maximum < image[row, col][0]:
                maximum = image[row, col][0]
    return maximum


def save_scaled_img(img, height, width, file_name):
    result = zeros((height, width), dtype=int)
    with open(file_name, 'w') as output:
        for row in range(height):
            for col in range(width):
                result[row, col] = int(255 * img[row, col][0])
    imwrite(file_name, result)


def write_dup_img(img, height, width, file_name):
    with open(file_name, 'w') as output:
        for row in range(height):
            for col in range(width):
                output.write(str((col, row)) + ' ' + str(img[row, col][0]) + '\n')


def test_img(graph, component_B, height, width):
    sum_in_B = 0
    sum_not_in_B = 0
    amount_in_B = 0
    amount_not_in_B = 0
    for row in range(height):
        for col in range(width):
            all_in_B = True
            all_not_in_B = True
            for row_c in (-1, 0, 1):
                for col_c in (-1, 0, 1):
                    if get_pixel_for_coords(row + row_c, col + col_c, width) in component_B:
                        all_not_in_B = False
                    else:
                        all_in_B = False
            if all_in_B:
                amount_in_B += 1
                sum_in_B += graph.get_edge(row, col)
            if all_not_in_B:
                amount_not_in_B += 1
                sum_not_in_B += graph.get_edge(row, col)

    prob_A = 0.
    prob_B = 0.
    if amount_in_B != 0:
        prob_B = 1. * sum_in_B / amount_in_B
    if amount_not_in_B != 0:
        prob_A = 1. * sum_not_in_B / amount_not_in_B
    return prob_A, prob_B


def write_results(fname, prob_A, prob_B):
    with open(fname, 'w') as output:
        output.write(str(prob_A) + '\t' + str(prob_B) + '\n' + str(max(prob_A, prob_B)))


def is_fake(image_file):
    processes = 8
    max_size = 100

    fname = image_file.split('.')[0]

    logger.log('Reading image from %s' % (image_file))
    img = imread(image_file)
    (height, width, _) = img.shape

    logger.log('Height: %d, width: %d' % (height, width))

    if max(height, width) > max_size:
        lam = 1. * max_size / max(height, width)
        logger.log('Resizing image as %.2f of original' % (lam))
        img = resize(img, (0,0), fx=lam, fy=lam)
        (height, width, _) = img.shape
        logger.log('New height: %d, new width: %d' % (height, width))

    logger.log('Scaling image')
    scaled_img = scale_img(img, height, width)

    logger.log('Getting gradient image')
    grad_img = compute_grad_img(img, height, width)

    logger.log('Scaling gradient image')
    max_grad = get_maximum(grad_img, height, width)
    scaled_grad_img = scale_img(grad_img, height, width, max_grad, max_grad, max_grad)

    logger.log('Getting duplicate image')
    dup_img = compute_dup_img(img, height, width, processes)

    write_dup_img(dup_img, height, width, 'dup_img')

    logger.log('Scaling duplicate image')
    max_dup = get_maximum(dup_img, height, width)
    scaled_dup_img = scale_img(dup_img, height, width, max_dup, max_dup, max_dup)

    save_scaled_img(scaled_dup_img, height, width, 'scaled_dup_img.jpg')

    logger.log('Building intensity graph')
    graph_intensity = scaled_img_to_graph(scaled_img, scaled_grad_img, scaled_grad_img, height, width, calc_intensity_weight, 8)

    if graph_intensity is None:
        logger.log('Couldn\'t build intensity graph', True)

    logger.log('Computing normalized cut for intensity graph')
    (intensity_A, intensity_B) = graph_intensity.lanczos_optimal_cut()

    intensity_image_fname = '%s_intensity.jpg' % (fname)
    logger.log('Saving intensity image in %s' % (intensity_image_fname))
    intensity_img = create_img_with_components(intensity_B, height, width)
    imwrite(intensity_image_fname, intensity_img)

    logger.log('Testing intensity image')
    intensity_prob_A, intensity_prob_B = test_img(graph_intensity, intensity_B, height, width)

    results_intensity_fname = '%s_results_intensity' % (fname)
    logger.log('Probabilities for intensity: %.2f, %.2f; writing to %s' % (intensity_prob_A, intensity_prob_B, results_intensity_fname))

    write_results(results_intensity_fname, intensity_prob_A, intensity_prob_B)

    logger.log('Building texture graph')
    graph_texture = scaled_img_to_graph(scaled_img, scaled_grad_img, scaled_grad_img, height, width, calc_texture_weight, 8)

    if graph_texture is None:
        logger.log('Couldn\'t build texture graph', True)

    logger.log('Computing normalized cut for texture graph')
    (texture_A, texture_B) = graph_texture.lanczos_optimal_cut()

    texture_image_fname = '%s_texture.jpg' % (fname)
    logger.log('Saving texture image in %s' % (texture_image_fname))
    texture_img = create_img_with_components(texture_B, height, width)
    imwrite(texture_image_fname, texture_img)

    logger.log('Testing texture image')
    texture_prob_A, texture_prob_B = test_img(graph_texture, texture_B, height, width)

    results_texture_fname = '%s_results_texture' % (fname)
    logger.log('Probabilities for texture: %.2f, %.2f; writing to %s' % (texture_prob_A, texture_prob_B, results_texture_fname))

    write_results(results_texture_fname, texture_prob_A, texture_prob_B)

    logger.log('Building duplicate graph')
    graph_duplicate = scaled_img_to_graph(scaled_img, scaled_grad_img, scaled_dup_img, height, width, calc_duplicate_weight, processes)

    if graph_duplicate is None:
        logger.log('Couldn\'t build duplicate graph', True)

    logger.log('Computing normalized cut for duplicate graph')
    (duplicate_A, duplicate_B) = graph_duplicate.lanczos_optimal_cut()

    duplicate_image_fname = '%s_duplicate.jpg' % (fname)
    logger.log('Saving duplicate image in %s' % (duplicate_image_fname))
    duplicate_img = create_img_with_components(duplicate_B, height, width)
    imwrite(duplicate_image_fname, duplicate_img)

    logger.log('Testing duplicate image')
    duplicate_prob_A, duplicate_prob_B = test_img(graph_duplicate, duplicate_B, height, width)

    results_duplicate_fname = '%s_results_duplicate' % (fname)
    logger.log('Probabilities for duplicate: %.2f, %.2f; writing to %s' % (duplicate_prob_A, duplicate_prob_B, results_duplicate_fname))

    write_results(results_duplicate_fname, duplicate_prob_A, duplicate_prob_B)

    logger.log('Done')


def is_fake_dir(list_of_files):
    for file_ in list_of_files:
        try:
            is_fake(file_)
        except:
            print file_

def main():
    dir_name = argv[1]
    is_fake_dir([join(dir_name, f) for f in listdir(dir_name)])
    return 0

if __name__ == '__main__':
    main()

