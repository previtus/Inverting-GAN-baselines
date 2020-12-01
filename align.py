print("PS: USE THE PSP ENV!")

import sys
# ALAE repository path so that I can import it here
REPO_PATH = "/home/vitek/Vitek/python_codes/pixel2style2pixel"
assert REPO_PATH[-1] != "/"
sys.path.append(REPO_PATH)

# CUDA 10 path (will be different system to system ...)
import os
sys.path.append('/usr/local/cuda-10.0/bin')
os.environ['CUDADIR'] = '/usr/local/cuda-10.0'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64'

import cv2
import os
from os import listdir
from os.path import isfile, join

import sys
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import dlib
from scripts.align_all_parallel import align_face


def run_alignment(image_path):
  start = timer()

  predictor = dlib.shape_predictor(REPO_PATH+"/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  end = timer()
  time = (end - start)
  print("Aligning took " + str(time) + "s")

  return aligned_image

default_path = "test_inputs"
default_output = "test_inputs/aligned"

import argparse
parser = argparse.ArgumentParser(description='Project: alignment helper.')
parser.add_argument('-f', help='path to the folder', default=default_path)
parser.add_argument('-o', help='output folder', default=default_output)

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.f
    output_folder = args.o

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    paths = [f for f in listdir(path) if isfile(join(path, f))]

    paths.sort()
    for image_index in range(len(paths)):
        image_path = path + "/" + paths[image_index]
        print(image_path)


        aligned_image = run_alignment(image_path)
        aligned_image.resize((1024,1024))
        aligned_image = np.array(aligned_image)
        aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR)

        img_name = output_folder + "/" + str(image_index).zfill(5) + ".jpg"
        cv2.imwrite(img_name, aligned_image)

"""
def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]
"""




