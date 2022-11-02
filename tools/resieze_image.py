import asyncio
from argparse import ArgumentParser
import os
import warnings

import cv2 as cv

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('out_file', help='output file')
    args = parser.parse_args()
    return args

def main(args):
    # test a single image
    assert os.path.isdir(args.out_file)
    img_file_list = os.listdir(args.img_file)
    for id, img_name in enumerate(img_file_list):
        if  img_name[-3:] not in ["jpg", "png", "bmp"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        print(img_name)
        output_name = str(id) + ".jpg"
        img_path = os.path.join(args.img_file, img_name)
        out_path = os.path.join(args.out_file, output_name)

        img = cv.imread(img_path)
        img = cv.resize(img, (128, 128))
        cv.imwrite(out_path, img)


if __name__ == '__main__':
    args = parse_args()
    main(args)
