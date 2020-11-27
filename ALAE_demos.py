import sys
# ALAE repository path so that I can import it here
REPO_PATH = "/home/vitek/Vitek/python_codes/ALAE"
assert REPO_PATH[-1] != "/"
sys.path.append(REPO_PATH)

# CUDA 10 path (will be different system to system ...)
import os
sys.path.append('/usr/local/cuda-10.0/bin')
os.environ['CUDADIR'] = '/usr/local/cuda-10.0'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64'

import subprocess
#subprocess.check_call(['nvidia-smi'], env=dict(os.environ))


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

cfg = get_cfg_defaults()
cfg.merge_from_file(REPO_PATH+"/configs/ffhq.yaml")


logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # print out all the info messages


#print("cfg", cfg)

model = Model(
    startf=cfg.MODEL.START_CHANNEL_COUNT,
    layer_count=cfg.MODEL.LAYER_COUNT,
    maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
    latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
    truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
    truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
    mapping_layers=cfg.MODEL.MAPPING_LAYERS,
    channels=cfg.MODEL.CHANNELS,
    generator=cfg.MODEL.GENERATOR,
    encoder=cfg.MODEL.ENCODER)

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

### LOAD
checkpointer = Checkpointer(cfg, model_dict, {}, logger=logger, save=False)

def load(checkpointer, model_path):
    return checkpointer.load(ignore_last_checkpoint=False, file_name=model_path)

model_path = "/home/vitek/Vitek/python_codes/ALAE/training_artifacts/ffhq/model_157.pth"
extra_checkpoint_data = load(checkpointer, model_path)

model.eval()

layer_count = cfg.MODEL.LAYER_COUNT


def encode(x):
    start = timer()

    Z, _ = model.encode(x, layer_count - 1, 1)
    Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
    # print(Z.shape)
    end = timer()
    time = (end - start)
    print("Encoding took "+str(time)+"s")

    return Z



#######################################################################################################################

def load_image(path):
    img = np.asarray(Image.open(path))
    return img

def image_to_tensor(img):
    start = timer()
    # 1.) to torch tensor
    if img.shape[2] == 4:
        img = img[:, :, :3] # only rgb
    im = img.transpose((2, 0, 1))
    # as tensor
    x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
    if x.shape[0] == 4:
        x = x[:3] # only rgb

    # 2.) resize to the models resolution (1024x1024x3 prolly)
    needed_resolution = model.decoder.layer_to_resolution[-1]
    while x.shape[2] > needed_resolution:
        x = F.avg_pool2d(x, 2, 2)
    if x.shape[2] != needed_resolution:
        x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

    resized_image = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(
        0, 1).numpy()

    end = timer()
    time = (end - start)
    #print("Image to tensor took "+str(time)+"s")

    return x, resized_image

def encode_from_image(img_tensor):
    start = timer()

    on_cuda = img_tensor[None, ...].cuda()

    #end = timer()
    #time = (end - start)
    #print("To cuda took "+str(time)+"s")

    #start = timer()

    Z, _ = model.encode(on_cuda, layer_count - 1, 1)
    Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

    latents = Z

    end = timer()
    time = (end - start)
    print("Encoding took "+str(time)+"s")

    return latents

def decode(x):
    start = timer()

    layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
    ones = torch.ones(layer_idx.shape, dtype=torch.float32)
    coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
    # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
    decoded = model.decoder(x, layer_count - 1, 1, noise=True)

    end = timer()
    time = (end - start)
    print("Decoding/Generating took "+str(time)+"s")

    return decoded

def postprocess_to_cpu_to_numpy(x_rec):
    resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
    resultsample = resultsample.cpu()[0, :, :, :]
    images = resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    # to numpy
    images = images.numpy()
    return images



"""
path = REPO_PATH+'/dataset_samples/faces/realign1024x1024'
paths = list(os.listdir(path))
paths.sort()
for image_index in range(10):
    image_path = path + '/' + paths[image_index]
    #image_path = "/home/vitek/Vitek/python_codes/Inverting-GAN-baselines/00000.png"
    start = timer()

    img = load_image(image_path)
    img_tensor, resized_image = image_to_tensor(img)
    latents = encode_from_image(img_tensor)

    decoded = decode(latents)
    reconstruction = postprocess_to_cpu_to_numpy(decoded)

    side_by_side = np.hstack((img, reconstruction))
    plt.imshow(side_by_side)

    end = timer()
    time = (end - start)
    print("Loop took "+str(time)+"s ===> this means ", (1.0 / time), "fps!" )

    plt.show()
"""





import cv2
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

img_counter = 0

while True:
    start = timer()

    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    print(frame.shape)
    cropSize = min([frame.shape[0],frame.shape[1]])

    def crop_center(img, cropx, cropy):
        y, x, _ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]
    frame = crop_center(frame, cropSize, cropSize)


    #print(type(frame), frame.shape)
    #img = load_image(image_path)
    img = frame
    img_tensor, resized_image = image_to_tensor(img)
    latents = encode_from_image(img_tensor)

    decoded = decode(latents)
    reconstruction = postprocess_to_cpu_to_numpy(decoded)
    reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2RGB)

    side_by_side = np.hstack((resized_image, reconstruction))
    #plt.imshow(side_by_side)

    end = timer()
    time = (end - start)
    print("Loop took "+str(time)+"s ===> this means ", (1.0 / time), "fps!" )

    w, h, ch = side_by_side.shape
    dim = (int(h/2),int(w/2))
    print(side_by_side.shape, dim)
    smaller_side_by_side = cv2.resize(side_by_side, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("test", smaller_side_by_side)


    k = cv2.waitKey(1)
    if (k == ord('q')) or (k%256 == 27):
        print("Quitting...")
        break


    if (k == ord('x')) or (k%256 == 32):
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, side_by_side)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
