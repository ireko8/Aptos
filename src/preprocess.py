from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


np.random.seed(71)
image_size = 512


# https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img > tol
    return img[np.ix_(mask.any(1),mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3], axis=-1)
        return img


def load_ben_color(path, sigmaX=30):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (image_size, image_size))
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image , (0, 0) , sigmaX), -4, 128)        
    return image


def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (image_size, image_size))

    return img


def preprocess(args):
    fname, root_dir = args
    name = fname.stem
    out_path = Path(f"preprocessed/{root_dir}/{name}.npy")
    if out_path.exists():
        return
    else:
        print(fname)
        image = load_ben_color(str(fname))
        np.save(out_path, image)


def main():

    dirs = [("input/train", "train_512"),
            ("input/test", "test_512"),
            ("previous/train", "previous_train_512"),
            ("previous/test", "previous_test_512")]
    for input_dir, root_dir in dirs:
        files = Path(input_dir).glob("*")
        Path(f"preprocessed/{root_dir}").mkdir(exist_ok=True, parents=True)
        args = [(f, root_dir) for f in files]
        with Pool(16) as p:
            p.map(preprocess, args)


if __name__ == "__main__":
    main()

