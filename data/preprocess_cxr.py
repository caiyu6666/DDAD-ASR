import pydicom
import numpy as np
import os
from PIL import Image
from joblib import Parallel, delayed


in_dir = "stage_2_train_images"  # Your path to original data (.dcm or .dicom)
out_dir = "images"  # Your output path for preprocessed data (.png)
img_size = 512  # image size of png


def preprocess_img(img_name):
	dicom = pydicom.read_file(os.path.join(in_dir, img_name))
	img = dicom.pixel_array
	if dicom.PhotometricInterpretation == "MONOCHROME1":
		img = np.max(img) - img

	if img.dtype != np.uint8:
		img = ((img - np.min(img)) * 1.0 / (np.max(img) - np.min(img)) * 255).astype(np.uint8)

	img = Image.fromarray(img).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
	img.save(os.path.join(out_dir, img_name.split(".")[0] + ".png"))


if __name__ == "__main__":
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	print(len(os.listdir(in_dir)), "images to preprocess.")
	Parallel(n_jobs=-1, verbose=1)(delayed(preprocess_img)(file) for file in os.listdir(in_dir))
