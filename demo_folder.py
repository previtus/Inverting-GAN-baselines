default_path = "test_inputs/aligned"
default_path = "/home/vitek/Vitek/python_codes/Inverting-GAN-baselines/test_inputs/aligned_from_alae_1024x1024"
default_output = "results"

from handler_ALAE import ALAE_Handler
handler = ALAE_Handler()

#from handler_PSP import PSP_Handler
#handler = PSP_Handler()



import cv2
import os
from os import listdir
from os.path import isfile, join

import argparse
parser = argparse.ArgumentParser(description='Project: Demo for GAN inversion.')
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

    handler.report()
    handler.load_model()

    for image_index in range(len(paths)):
        image_path = path + "/" + paths[image_index]
        print(image_path)

        image = handler.load_image(image_path)

        preprocessed_image = handler.preprocess_image(image) # resize? center crop? dlib face?
        print("INPUT IMAGE:", type(image), image.shape) #INPUT IMAGE: <class 'numpy.ndarray'> (1024, 1024, 3)

        latent, time = handler.encode_image(preprocessed_image)
        print("Encoding took " + str(time) + "s")
        print("INTERMEDIATE LATENT:", type(latent), latent.shape)

        reconstruction, time = handler.generate_image(latent)
        print("Decoding/Generating took " + str(time) + "s")

        reconstruction = handler.postprocess_image(reconstruction) # to cpu? to numpy?
        print("OUTPUT IMAGE:", type(reconstruction), reconstruction.shape)

        side_by_side = np.hstack((image, reconstruction))

        side_by_side = np.array(side_by_side)
        side_by_side = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        img_name = output_folder + "/" + handler.name+"_" + str(image_index).zfill(5) + ".jpg"
        cv2.imwrite(img_name, side_by_side)


