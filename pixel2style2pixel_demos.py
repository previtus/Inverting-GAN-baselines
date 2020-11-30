print("SWITCH TO PSP ENV!")
assert True
run_dlib = True

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


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import subprocess
#subprocess.check_call(['nvidia-smi'], env=dict(os.environ))

# Commented out IPython magic to ensure Python compatibility.
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

import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from datasets import augmentations
sys.path.append(REPO_PATH+"/utils")
from common import tensor2im, log_input_image
from models.psp import pSp

experiment_type = 'ffhq_encode'
EXPERIMENT_ARGS = {
        "model_path": REPO_PATH+"/pretrained_models/psp_ffhq_encode.pt",
        "image_path": REPO_PATH+"/notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            # EXPECTS input to be 256x256 - outputs are for 1024x1024 but get resized elsewhere
            #  -> see https://github.com/eladrich/pixel2style2pixel/issues/5
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
  raise ValueError("Pretrained model was unable to be downlaoded correctly!")


model_path = REPO_PATH+"/pretrained_models/psp_ffhq_encode.pt"

def create_model_from_path(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    #pprint.pprint(opts)

    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return net, opts


def run_alignment(image_path):
  import dlib
  from scripts.align_all_parallel import align_face
  predictor = dlib.shape_predictor(REPO_PATH+"/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image

def open_image(image_path):
    original_image = Image.open(image_path)
    if opts.label_nc == 0:
        original_image = original_image.convert("RGB")
    else:
        original_image = original_image.convert("L")

    # load real img ?
    print(image_path)
    print(original_image.size)



net, opts = create_model_from_path(model_path)

image_path = EXPERIMENT_ARGS["image_path"]

if run_dlib:
  input_image = run_alignment(image_path)
else:
  input_image = open_image(image_path)

print("input_image", type(input_image)) # PIL image

input_image.resize((256, 256))

"""## Step 6: Perform Inference"""

#care input should be 256x256 ....
##input_image = original_image
## ps alignment script also resized it to 256x256

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)
print("transformed_image.shape", transformed_image.shape)

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
        print("result_batch.shape",result_batch.shape)

    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            print("res.shape",res.shape)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

latent_mask = None

def process_image(input_image):
    #input_image.resize((256, 256))
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)
    print("transformed_image.shape", transformed_image.shape)
    return transformed_image


def encode(image_tensor, net):
    print("image_tensor.shape", image_tensor.shape)

    tmp = image_tensor.unsqueeze(0)
    # get latent vector to inject into our input image
    """
    vec_to_inject = np.random.randn(1, 512).astype('float32')
    _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                  input_code=True,
                                  return_latents=True)
    """
    latent_to_inject = None

    image_tensor_on_cuda = tmp[0].unsqueeze(0).to("cuda").float()

    latents = net.encode(image_tensor_on_cuda, latent_mask=latent_mask, inject_latent=latent_to_inject)
    return latents


def decode(latents, net):
    images = net.decode(latents)
    image = images[0]
    print("image.shape", image.shape)

    return image

def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


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

    print("frame.shape", frame.shape)
    cropSize = min([frame.shape[0],frame.shape[1]])

    frame = crop_center(frame, cropSize, cropSize)

    print("INPUT IMAGE:", type(frame), frame.shape)

    print("cropped frame.shape", frame.shape)
    frame = Image.fromarray(frame, 'RGB')

    transformed_image = process_image(frame)
    latents = encode(transformed_image, net)

    print("INTERMEDIATE LATENT:", type(latents), latents.shape)

    """
    INPUT IMAGE: <class 'numpy.ndarray'> (480, 480, 3)
    INTERMEDIATE LATENT: <class 'torch.Tensor'> torch.Size([1, 18, 512])
    OUTPUT IMAGE: <class 'numpy.ndarray'> (1024, 1024, 3)
    """


    result_image = decode(latents, net)

    vis_frame = frame.resize((1024,1024))
    #input_vis_image = log_input_image(transformed_image, opts)
    #input_vis_image = input_vis_image.resize((1024,1024))

    reconstruction = tensor2im(result_image)
    reconstruction = np.asarray(reconstruction)
    reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2RGB)

    print("OUTPUT IMAGE:", type(reconstruction), reconstruction.shape)

    #side_by_side = np.hstack((input_vis_image, output_image))
    side_by_side = np.hstack((vis_frame, reconstruction))

    end = timer()
    time = (end - start)
    print("Loop took "+str(time)+"s ===> this means ", (1.0 / time), "fps!" )

    w, h, ch = side_by_side.shape
    dim = (int(h/2),int(w/2))
    print(side_by_side.shape, dim)

    side_by_side = cv2.resize(side_by_side, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("test", side_by_side)


    k = cv2.waitKey(1)
    if (k == ord('q')) or (k%256 == 27):
        print("Quitting...")
        break


    if (k == ord('x')) or (k%256 == 32):
        # SPACE pressed
        img_name = "p2s2p_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, side_by_side)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


