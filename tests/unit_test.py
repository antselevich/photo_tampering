#!/usr/bin/python

import unittest

from sys import path
from os.path import abspath, dirname, pardir, join
path.append(abspath(join(dirname(__file__), pardir)))

from graph import Graph
from cut import calc_intensity_weight, scaled_img_to_graph, calc_weight_for_test, compute_dup_img
from copy import deepcopy
from numpy import zeros

def get_real(g):
        min_cut_expected = float("inf")
        for s1, s2 in g.generate_all_partitions():
            current_cut = g.calculate_normalized_cut(s1, s2)
            if current_cut < min_cut_expected:
                min_cut_expected = current_cut
                t1 = deepcopy(s1)
                t2 = deepcopy(s2)
        print t1, t2

        min_cut_expected2 = g.calculate_normalized_cut(t1, t2)

        print min_cut_expected
        print min_cut_expected2


class TestGraph(unittest.TestCase):
    def test_lanczos_optimal_cut_1(self):
        g = Graph(4, [(0, 0, 0.), (1, 1, 0.), (2, 2, 0.), (3, 3, 0.), (1, 0, 0.6), (2, 0, 0.7), (3, 0, 0.55), (2, 1, 0.55), (3, 1, 0.6), (3, 2, 10.)])

        real1, real2 = g.lanczos_optimal_cut()
        min_cut_real = g.calculate_normalized_cut(real1, real2)

        expected1 = set([0, 1])
        expected2 = set([2, 3])
        min_cut_expected = 0.77376042124387123

        self.assertAlmostEqual(min_cut_expected, min_cut_real, places=3)

        all_real = set([str(real1), str(real2)])
        all_expected = set([str(expected1), str(expected2)])

        self.assertSetEqual(all_expected, all_real)


    def test_lanczos_optimal_cut_2(self):
        g = Graph(5, [
                        (0, 0, 0.),
                        (1, 1, 0.),
                        (2, 2, 0.),
                        (3, 3, 0.),
                        (4, 4, 0.),

                        (1, 0, 15.1),
                        (2, 0, 14,3),
                        (3, 0, 21.3),
                        (4, 0, 11.1),

                        (2, 1, 17.),
                        (3, 1, 10.2),
                        (4, 1, 4.9),

                        (3, 2, 19.99),
                        (4, 2, 17.1),

                        (4, 3, 11.1)])

        real1, real2 = g.lanczos_optimal_cut()
        min_cut_real = g.calculate_normalized_cut(real1, real2)

        expected1, expected2 = set([0, 1, 3]), set([2, 4])
        min_cut_expected = 1.151324986

        #self.assertAlmostEqual(min_cut_expected, min_cut_real, places=3)

        all_real = set([str(real1), str(real2)])
        all_expected = set([str(expected1), str(expected2)])

        self.assertSetEqual(all_expected, all_real)


    def test_lanczos_optimal_cut_3(self):

        g = Graph(7,
                [(0, 0, 0.),
                 (1, 1, 0.),
                 (2, 2, 0.),
                 (3, 3, 0.),
                 (4, 4, 0.),
                 (5, 5, 0.),
                 (6, 6, 0.),

                 (3, 0, 10.),
                 (4, 0, 7.),
                 (4, 3, 6.),

                 (2, 1, 5.),
                 (5, 1, 8.),
                 (6, 1, 6.),
                 (5, 2, 6.7),
                 (6, 2, 7.1),
                 (6, 5, 9.),

                 (1, 0, 1.2),
                 (2, 0, 0.9),
                 (5, 0, 1.1),
                 (6, 0, 2.1),
                 (3, 1, 1.3),
                 (3, 2, 2.),
                 (5, 3, 0.6),
                 (6, 3, 1.2),
                 (4, 1, 1.1),
                 (4, 2, 2.),
                 (5, 4, 1.1),
                 (6, 4, 0.9)])

        real1, real2 = g.lanczos_optimal_cut()
        min_cut_real = g.calculate_normalized_cut(real1, real2)

        expected1 = set([0, 4, 3])
        expected2 = set([1, 2, 5, 6])
        min_cut_expected = 0.408440189346

        self.assertAlmostEqual(min_cut_expected, min_cut_real, places=3)

        all_real = set([str(real1), str(real2)])
        all_expected = set([str(expected1), str(expected2)])

        self.assertSetEqual(all_expected, all_real)


class TestCut(unittest.TestCase):

    def test_calc_intensity_weight_1(self):
        ver1 = (0, 0)
        ver2 = (0.03, 0.04)
        pix1 = (0, 0, 0)
        pix2 = (0.1, 0.2, 0.3)
        expected = 0.
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 0.06931471805599453, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_2(self):
        ver1 = (0, 0)
        ver2 = (0.3, 0.4)
        pix1 = (0, 0, 0)
        pix2 = (0.01, 0.02, 0.03)
        expected = 0.
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 0.06931471805599453, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_3(self):
        ver1 = (0, 0)
        ver2 = (0.03, 0.04)
        pix1 = (0, 0, 0)
        pix2 = (0.01, 0.02, 0.03)
        expected = 0.8693582353988059
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 0.06931471805599453, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_4(self):
        ver1 = (0, 0)
        ver2 = (0.08, 0.08)
        pix1 = (0, 0, 0)
        pix2 = (0.01, 0.02, 0.03)
        expected = 0.764907781102864
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 0.06931471805599453, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_5(self):
        ver1 = (0, 0)
        ver2 = (0.03, 0.04)
        pix1 = (0, 0, 0)
        pix2 = (0.1, 0.2, 0.3)
        expected = 0.
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 1E9, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_6(self):
        ver1 = (0, 0)
        ver2 = (0.3, 0.4)
        pix1 = (0, 0, 0)
        pix2 = (0.01, 0.02, 0.03)
        expected = 0.
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 1E9, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_7(self):
        ver1 = (0, 0)
        ver2 = (0.03, 0.04)
        pix1 = (0, 0, 0)
        pix2 = (0.01, 0.02, 0.03)
        expected = 0.8693582353988059
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 1E9, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_calc_intensity_weight_8(self):
        ver1 = (0, 0)
        ver2 = (0.08, 0.08)
        pix1 = (0, 0, 0)
        pix2 = (0.01, 0.02, 0.03)
        expected = 0.764907781102864
        real = calc_intensity_weight(ver1, pix1, None, None, ver2, pix2, None, None, 1E9, 0.01, -0.6931471805599453)
        self.assertAlmostEqual(expected, real, places=3)


    def test_compute_dup_img_1(self):
        image = zeros((7,7), dtype=tuple)
        image[0, 0] = (32, 76, 44)
        image[0, 1] = (37, 45,  3)
        image[0, 2] = (48, 74, 54)
        image[0, 3] = (41, 71, 96)
        image[0, 4] = (74, 90,  0)
        image[0, 5] = (78, 54, 48)
        image[0, 6] = (10, 49, 28)
        image[1, 0] = (87, 29, 69)
        image[1, 1] = (28, 18, 29)
        image[1, 2] = (92, 78, 86)
        image[1, 3] = (48, 37, 29)
        image[1, 4] = (85, 29, 68)
        image[1, 5] = (99, 68, 28)
        image[1, 6] = ( 3, 84, 49)
        image[2, 0] = (22, 72, 59)
        image[2, 1] = (75, 58, 38)
        image[2, 2] = (38, 83, 10)
        image[2, 3] = (11, 47, 69)
        image[2, 4] = (90, 28, 99)
        image[2, 5] = (45,  5, 83)
        image[2, 6] = (31,  7, 29)
        image[3, 0] = ( 5, 28, 69)
        image[3, 1] = (31, 89, 19)
        image[3, 2] = (19, 11, 94)
        image[3, 3] = (82, 39, 80)
        image[3, 4] = (94, 96, 39)
        image[3, 5] = (66, 33,  6)
        image[3, 6] = (75,  9, 28)
        image[4, 0] = (39, 48, 59)
        image[4, 1] = (78, 10,  9)
        image[4, 2] = (37, 58, 85)
        image[4, 3] = (28, 20, 29)
        image[4, 4] = (29, 49, 11)
        image[4, 5] = (74, 63, 69)
        image[4, 6] = ( 6, 59, 93)
        image[5, 0] = (31, 92, 10)
        image[5, 1] = (96, 47, 59)
        image[5, 2] = (11,  9, 57)
        image[5, 3] = ( 8, 49, 68)
        image[5, 4] = (47, 68, 36)
        image[5, 5] = (94, 29, 54)
        image[5, 6] = (48, 59, 38)
        image[6, 0] = (29, 30, 69)
        image[6, 1] = (10, 59, 30)
        image[6, 2] = (73, 20, 59)
        image[6, 3] = (28, 59, 29)
        image[6, 4] = (31,  9,  5)
        image[6, 5] = (85, 39, 49)
        image[6, 6] = ( 9, 91, 29)
        new_image = compute_dup_img(image, 7, 7, 1)
        #print new_image
        '''
        [[ 70121.0,  67831.0,  73660.0,  67831.0,  77539.0,  63545.0,  63545.0]
         [ 68029.0,  84701.0, 108275.0, 108275.0, 101235.0,  82092.0,  74728.0]
         [ 65543.0,  77149.0, 101136.0, 105553.0, 116360.0,  97139.0,  78080.0]
         [ 72561.0,  95190.0, 111728.0, 112821.0, 113295.0,  93348.0,  74392.0]
         [ 66008.0,  81833.0, 112821.0, 113295.0, 101136.0,  93348.0,  74392.0]
         [ 65147.0,  77149.0,  91174.0,  93386.0,  95885.0,  76231.0,  60464.0]
         [ 65147.0,  65110.0,  53050.0,  74989.0,  65110.0,  53050.0,  60464.0]]
        '''
        self.assertAlmostEqual(73660., new_image[0, 2][0], places=2)
        self.assertAlmostEqual(84701., new_image[1, 1][1], places=2)
        self.assertAlmostEqual(116360., new_image[2, 4][2], places=2)
        self.assertAlmostEqual(74392., new_image[3, 6][0], places=2)
        self.assertAlmostEqual(113295., new_image[4, 3][1], places=2)
        self.assertAlmostEqual(76231., new_image[5, 5][2], places=2)
        self.assertAlmostEqual(65147., new_image[6, 0][0], places=2)

if __name__ == '__main__':
    unittest.main()
