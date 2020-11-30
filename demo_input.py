default_path = "00000.png"

from handler_ALAE import ALAE_Handler
handler = ALAE_Handler()

#from handler_PSP import PSP_Handler
#handler = PSP_Handler()

import argparse
parser = argparse.ArgumentParser(description='Project: Demo for GAN inversion.')
parser.add_argument('-i', help='path to the input image', default=default_path)

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parser.parse_args()

    handler.report()

    handler.load_model()
    image = handler.load_image(args.i)

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
    plt.imshow(side_by_side)
    name = "Out_"+handler.name+".jpg"
    plt.savefig(name)
    print("Saved to >>", name)
    plt.close()
