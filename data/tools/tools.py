#

import os


test_path = '/home/yangshun/PycharmProjects/baidu_detection/data/test.txt'
test_gt_path = '/home/yangshun/PycharmProjects/baidu_detection/data/test_gt.txt'


def check_test_gt(test_path, test_gt_path):
    test_files = open(test_path, 'r')
    test_gt_files = open(test_gt_path, 'r')

    for test_line in test_files:
        test_line = test_line.strip()
        test_gt_line = test_gt_files.readline().strip().split(' ')[0]
        assert test_line == test_gt_line, '{}-{}'.format(test_line, test_gt_line)


if __name__ == "__main__":
    check_test_gt(test_path, test_gt_path)