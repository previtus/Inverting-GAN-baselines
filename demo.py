#from handler_ALAE import ALAE_Handler
#handler = ALAE_Handler()

from handler_PSP import PSP_Handler
handler = PSP_Handler()
handler.report()
handler.load_model()

#########################################################################

from timeit import default_timer as timer
import numpy as np
import cv2
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


while True:
    start = timer()

    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    print(frame.shape)
    cropSize = min([frame.shape[0],frame.shape[1]])
    image = crop_center(frame, cropSize, cropSize)
    resized_image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_AREA)

    preprocessed_image = handler.preprocess_image(image)  # resize? center crop? dlib face?

    latent, time = handler.encode_image(preprocessed_image)
    print("Encoding took " + str(time) + "s")
    print("INTERMEDIATE LATENT:", type(latent), latent.shape)

    reconstruction, time = handler.generate_image(latent)
    print("Decoding/Generating took " + str(time) + "s")

    reconstruction = handler.postprocess_image(reconstruction)  # to cpu? to numpy?
    print("OUTPUT IMAGE:", type(reconstruction), reconstruction.shape)

    side_by_side = np.hstack((resized_image, reconstruction))
    #plt.imshow(side_by_side)

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
        handler.save_image(side_by_side, handler.name+"frame_")

cam.release()
cv2.destroyAllWindows()
