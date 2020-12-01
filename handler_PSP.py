import sys
REPO_PATH = "/home/vitek/Vitek/python_codes/pixel2style2pixel"
assert REPO_PATH[-1] != "/"
sys.path.append(REPO_PATH)

run_dlib = True
from argparse import Namespace
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from timeit import default_timer as timer
import cv2
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from datasets import augmentations
sys.path.append(REPO_PATH+"/utils")
from common import tensor2im, log_input_image
from models.psp import pSp

from handler_generic import GenericHandler

class PSP_Handler(GenericHandler):
    def __init__(self):
        GenericHandler.__init__(self)
        self.name = "psp_"

    def report(self):
        print("This is a Inverting GAN handler for pixel2style2pixel method.")

    def create_model(self):
        experiment_type = 'ffhq_encode'
        self.EXPERIMENT_ARGS = {
            "model_path": REPO_PATH + "/pretrained_models/psp_ffhq_encode.pt",
            "image_path": REPO_PATH + "/notebooks/images/input_img.jpg",
            "transform": transforms.Compose([
                # EXPECTS input to be 256x256 - outputs are for 1024x1024 but get resized elsewhere
                #  -> see https://github.com/eladrich/pixel2style2pixel/issues/5
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }

        if os.path.getsize(self.EXPERIMENT_ARGS['model_path']) < 1000000:
            raise ValueError("Pretrained model was unable to be downlaoded correctly!")


    # Main functionality:

    def load_model(self, model_path=REPO_PATH+"/pretrained_models/psp_ffhq_encode.pt"):
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        # pprint.pprint(opts)

        # update the training options
        opts['checkpoint_path'] = model_path
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False

        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')

        self.net = net
        self.opts = opts

    def preprocess_image(self, image):
        input_image = Image.fromarray(image, 'RGB')
        img_transforms = self.EXPERIMENT_ARGS['transform']
        transformed_image = img_transforms(input_image)

        return transformed_image

    def encode_image(self, image_tensor):
        # Expected input: image in your own format
        # Output: latent ~ torch.Size([1, 18, 512])
        start = timer()

        print("image_tensor.shape", image_tensor.shape)

        tmp = image_tensor.unsqueeze(0)
        # get latent vector to inject into our input image
        latent_to_inject = None
        """
        vec_to_inject = np.random.randn(1, 512).astype('float32')
        _, latent_to_inject = self.net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
        """
        latent_mask = None
        image_tensor_on_cuda = tmp[0].unsqueeze(0).to("cuda").float()

        # TODO - move the .encode function here so that it works with the default net
        latents = self.net.encode(image_tensor_on_cuda, latent_mask=latent_mask, inject_latent=latent_to_inject)

        end = timer()
        time = (end - start)

        return latents, time

    def generate_image(self, latent):
        start = timer()

        # TODO - move the .decode function here so that it works with the default net
        images = self.net.decode(latent)
        decoded = images[0]
        print("image.shape", decoded.shape)

        end = timer()
        time = (end - start)

        return decoded, time

    def postprocess_image(self, image):
        reconstruction = tensor2im(image)
        reconstruction = np.asarray(reconstruction)
        #reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2RGB)

        return reconstruction

"""
test = PSP_Handler()
test.report()

test.load_model()
image_path = "00000.png"
image = test.load_image(image_path)

preprocessed_image = test.preprocess_image(image) # resize? center crop? dlib face?
print("INPUT IMAGE:", type(image), image.shape) #INPUT IMAGE: <class 'numpy.ndarray'> (1024, 1024, 3)

latent, time = test.encode_image(preprocessed_image)
print("Encoding took " + str(time) + "s")
print("INTERMEDIATE LATENT:", type(latent), latent.shape)

reconstruction, time = test.generate_image(latent)
print("Decoding/Generating took " + str(time) + "s")

reconstruction = test.postprocess_image(reconstruction) # to cpu? to numpy?
print("OUTPUT IMAGE:", type(reconstruction), reconstruction.shape)

side_by_side = np.hstack((image, reconstruction))
plt.imshow(side_by_side)
plt.savefig("psphandlerout.jpg")
plt.show()
"""