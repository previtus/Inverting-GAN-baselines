import sys
REPO_PATH = "/home/vitek/Vitek/python_codes/ALAE"
assert REPO_PATH[-1] != "/"
sys.path.append(REPO_PATH)

import os
import torch.utils.data
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
import logging
from PIL import Image
import bimpy
import cv2
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
lreq.use_implicit_lreq.set(True)
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from handler_generic import GenericHandler

class ALAE_Handler(GenericHandler):
    def __init__(self):
        GenericHandler.__init__(self)
        self.name = "alae_"

    def report(self):
        print("This is a Inverting GAN handler for ALAE method.")

    def create_model(self, config_path="/configs/ffhq.yaml"):
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(REPO_PATH+config_path)
        logger = logging.getLogger("logger")
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # print out all the info messages

        # print("cfg", cfg)

        model = Model(
            startf=self.cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=self.cfg.MODEL.LAYER_COUNT,
            maxf=self.cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=self.cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=self.cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=self.cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=self.cfg.MODEL.MAPPING_LAYERS,
            channels=self.cfg.MODEL.CHANNELS,
            generator=self.cfg.MODEL.GENERATOR,
            encoder=self.cfg.MODEL.ENCODER)

        model.cuda()
        model.eval()
        model.requires_grad_(False)

        decoder = model.decoder
        encoder = model.encoder
        mapping_tl = model.mapping_tl
        mapping_fl = model.mapping_fl
        dlatent_avg = model.dlatent_avg

        logger.info("Trainable parameters generator:")
        print(count_parameters(decoder))
        logger.info("Trainable parameters discriminator:")
        print(count_parameters(encoder))

        model_dict = {
            'discriminator_s': encoder,
            'generator_s': decoder,
            'mapping_tl_s': mapping_tl,
            'mapping_fl_s': mapping_fl,
            'dlatent_avg': dlatent_avg
        }

        checkpointer = Checkpointer(self.cfg, model_dict, {}, logger=logger, save=False)

        self.checkpointer = checkpointer
        self.model = model
        self.layer_count = self.cfg.MODEL.LAYER_COUNT
        self.encoder = encoder
        self.decoder = decoder


    # Main functionality:

    def load_model(self, model_path="/home/vitek/Vitek/python_codes/ALAE/training_artifacts/ffhq/model_157.pth"):

        extra_checkpoint_data = self.checkpointer.load(ignore_last_checkpoint=False, file_name=model_path)
        self.model.eval()


    def preprocess_image(self, image):
        # Expected input: image in numpy
        # Output: image in your own format

        # 1.) to torch tensor
        if image.shape[2] == 4:
            image = image[:, :, :3]  # only rgb
        im = image.transpose((2, 0, 1))
        # as tensor
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]  # only rgb

        # 2.) resize to the models resolution (1024x1024x3 prolly)
        needed_resolution = self.model.decoder.layer_to_resolution[-1]
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        # this all to preprocess
        #resized_image = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0,2).transpose(0, 1).numpy()
        img_tensor = x

        return img_tensor #, resized_image

    def encode_image(self, image_tensor):
        # Expected input: image in your own format
        # Output: latent ~ torch.Size([1, 18, 512])
        start = timer()

        on_cuda = image_tensor[None, ...].cuda()

        Z, _ = self.model.encode(on_cuda, self.layer_count - 1, 1)
        Z = Z.repeat(1, self.model.mapping_fl.num_layers, 1)

        latents = Z

        end = timer()
        time = (end - start)

        return latents, time

    def generate_image(self, latent):
        # latent to image, measure speed
        # Expected input: latent ~ torch.Size([1, 18, 512])
        # Output: image in some torch format ...

        start = timer()

        layer_idx = torch.arange(2 * self.layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        decoded = self.model.decoder(latent, self.layer_count - 1, 1, noise=True)

        end = timer()
        time = (end - start)

        return decoded, time

    def postprocess_image(self, image):
        # Expected input: image in some torch format ...
        # Output: numpy image

        resultsample = ((image * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
        resultsample = resultsample.cpu()[0, :, :, :]
        asimage = resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

        # to numpy
        asimage = asimage.numpy()
        asimage = cv2.cvtColor(asimage, cv2.COLOR_BGR2RGB)

        return asimage

"""
test = ALAE_Handler()
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
#plt.imshow(side_by_side)
#plt.show()
#plt.savefig("alaehandlerout.jpg")

"""
