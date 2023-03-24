import glob
nst_path = "./data/nst_data"
img_name = "DME-3410772-1"
nst_imgs = [nst_img_path for nst_img_path in glob.glob(nst_path + f"/{img_name}_?.jpg")]
print(nst_imgs)

