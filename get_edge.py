#!/usr/bin/env python3

import pathlib

import cv2
import numpy as np

def main():
    neighbor_8 = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]],
                          np.uint8)

    here = pathlib.Path(__file__).resolve().parent

    for path in ((here / pathlib.Path('data/org')).glob('*.jpg')):
        img_org = cv2.imread(str(path))

        img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        img_dilated = cv2.dilate(img_gray, neighbor_8, iterations=1)
        img_diff = cv2.absdiff(img_gray, img_dilated)
        img_edge = cv2.bitwise_not(img_diff)

        base_name = path.name

        edge_path = here / pathlib.Path('data/edge') / base_name
        cv2.imwrite(str(edge_path), img_edge)

        img_edge_bgr = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
        img_pair = cv2.hconcat([img_org, img_edge_bgr])
        pair_path = here / pathlib.Path('data/org_edge_pair') / base_name
        cv2.imwrite(str(pair_path), img_pair)


if __name__ == '__main__':
    main()
