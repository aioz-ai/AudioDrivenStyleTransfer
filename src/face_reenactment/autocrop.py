import sys
sys.path.append("..")
from crop_video import process_video, crop_video_with_cmd

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--inp", required=True, help='Input image or video')
parser.add_argument("--outp", required=True, help='Output image or video')



args = parser.parse_args()


cmds = process_video(args.inp)
crop_video_with_cmd(cmds,args.outp)
