import argparse
import os
from os import path
import rawpy
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='input image directory')
    parser.add_argument('out_path', type=str, help='output image directory')

    args = parser.parse_args()

    in_paths = sorted(os.listdir(path.join(args.in_path)))

    os.makedirs(args.out_path, exist_ok=True)

    for p in tqdm(in_paths):
        img = rawpy.imread(path.join(args.in_path, p)).postprocess()
        out_p = path.join(args.out_path, p.split('.')[0] + ".png")
        imageio.imwrite(out_p, img)    

if __name__ == "__main__":
    main()