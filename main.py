import argparse
import os
import sys
from pathlib import Path

from torchvision.utils import save_image
from tqdm.auto import tqdm

from hair_swap import HairFast, get_parser


def main(model_args, args):
    hair_fast = HairFast(model_args)

    face_path = args.face_path
    shape_path = args.shape_path
    color_path = args.color_path
    
    final_image = hair_fast.swap(face_path, shape_path, color_path, benchmark=args.benchmark)
    save_image(final_image, args.result_path)


if __name__ == "__main__":
    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')
    parser.add_argument('--benchmark', action='store_true', help='Calculates the speed of the method during the session')

    # Arguments for single experiment
    parser.add_argument('--face_path', type=Path, default="face.png", help='Path to the face image')
    parser.add_argument('--shape_path', type=Path, default="hair.jpg", help='Path to the shape image')
    parser.add_argument('--color_path', type=Path, default="hair.jpg", help='Path to the color image')
    parser.add_argument('--result_path', type=Path, default="result.jpg", help='Path to save the result')

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_)
        model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)

    main(model_args, args)
