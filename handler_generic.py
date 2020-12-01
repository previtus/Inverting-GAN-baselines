# Generic class for a inverting GAN handler:
from PIL import Image
import cv2
import numpy as np

class GenericHandler:

    # Can be used by: Renderer, Camera

    def __init__(self):
        self.name = "generic_"
        self.saved_counter = 0
        self.create_model()

    def report(self):
        print("This is a generic version of the Inverting GAN handler.")

    def create_model(self):
        # Loads the necessary libraries etc...
        print("create_model To be implemented!")

    # Main functionality:

    def load_model(self, model_path):
        print("To be implemented!")
        #assert False

    def encode_image(self, image):
        # image to latent, measure speed
        print("To be implemented!")
        assert False

    def generate_image(self, latent):
        # latent to image, measure speed
        print("To be implemented!")
        assert False

    def preprocess_image(self, image):
        return image

    def postprocess_image(self, image):
        return image

    # Later:

    def blend_models(self, modelA, modelB, alpha):
        print("To be implemented!")
        assert False

    # Helpers:

    def load_image(self, image_path):
        # as numpy array
        return np.asarray(Image.open(image_path))

    def save_image(self, image, image_path="saved_frame_"):
        img_name = image_path+str(self.saved_counter).zfill(4)+".png"
        cv2.imwrite(img_name, image)
        print("{} written!".format(img_name))
        self.saved_counter += 1



    def input_camera(self):
        print("To be implemented!")
        assert False

    def output_window_render(self):
        print("To be implemented!")
        assert False

    def load_images_from_folder(self, folder_path):
        print("To be implemented!")
        assert False
    def save_images_to_folder(self, folder_path, images):
        print("To be implemented!")
        assert False


