import os
import cv2
from logone.utilities.utils import make_border
from tqdm import tqdm
import random as r
import numpy as np
from PIL import Image, ImageOps, ImageTransform

second_transform = False

def cylindricalWarp(img, K, rev=False):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    # img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    img_background = np.ones_like(img) * 255
    return cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, dst=img_background, borderMode=cv2.BORDER_TRANSPARENT)

def apply_cylindrical_transformation(img, diag=800, vert=2, horiz=2):
    """
    apply cylindrical projection, return image after cylindrical projection

    Inputs:
        img (cv2 img): image to transform
        diag (float): controls cylinder diameter
        vert (float): controls cylinder pitch
        horiz (float): controls left/right cylinder shift
    """
    h, w = img.shape[:2]
    K = np.array([[diag,0,w/horiz],[0,diag,h/vert],[0,0,1]]) # mock intrinsics
    img_cyl = cylindricalWarp(img, K)
    return img_cyl

def test_random_transformation(img_file, random_seed=0):
    img_filepath = os.path.join(os.getcwd(),'test_images', 'original_256',img_file)
    save_dir = os.path.join(os.getcwd(), 'test_images', 'transformed_test')
    basename = os.path.basename(img_file)[:-4]
    print(basename)
    cv_img = cv2.imread(img_filepath)
    if not second_transform:
        padding = 100
        cv_img = cv2.copyMakeBorder(cv_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value=0)
    img_cylindrical = random_cylindrical_transform(cv_img, random_seed)
    cv2.imwrite(os.path.join(save_dir, f'{basename}_cylindrical.jpg'), img_cylindrical)

def test_transformation(img_file):
    img_filepath = os.path.join(os.getcwd(),'test_images', 'original',img_file)
    with Image.open(img_filepath) as img:
        basename = os.path.basename(img_file)
        save_dir = os.path.join(os.getcwd(), 'test_images', 'transformed_test')
        
        # Affine transformation
        # img_affine = apply_affine_transform(img)
        # img_affine.save(os.path.join(save_dir, f'{basename}_affine.jpg'))
        
        # Cylindrical transformation
        cv_img = cv2.imread(img_filepath)
        padding = 100
        cv_img = cv2.copyMakeBorder(cv_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value=0)
        img_cylindrical = apply_cylindrical_transformation(cv_img)
        cv2.imwrite(os.path.join(save_dir, f'{basename}_cylindrical.jpg'), img_cylindrical)

def generate_random_perspective_matrix():
    a, e = np.random.uniform(0.8, 1.2, 2)  # Slight scaling
    b, d = np.random.uniform(-0.5, 0.5, 2)  # Mild rotation/skew
    g, h = np.random.uniform(-0.001, 0.001, 2)
    return np.array([[a, b, 0], [d, e, 0], [g, h, 1]], dtype=np.float32)

def apply_perspective_transform_terms(img, a,b,d,e,g,h):
    t_mat = np.array([[a, b, 0], [d, e, 0], [g, h, 1]], dtype=np.float32)
    return apply_perspective_transform(img, t_mat)

def apply_logo_transform(img, diag, vert, horiz,a,b,d,e,g,h):
    img = apply_cylindrical_transformation(img, diag,vert,horiz)
    return apply_perspective_transform_terms(img,a,b,d,e,g,h)

def apply_perspective_transform(img, transformation_matrix):
    h, w = img.shape[:2]
    img_background = np.ones_like(img) * 255
    transformed_image = cv2.warpPerspective(img, transformation_matrix, (w, h), dst=img_background, borderMode=cv2.BORDER_TRANSPARENT)
    return transformed_image

def random_cylindrical_transform(img, random_seed=None, padding=100):
    """
    return random cylindrical transform with prameters used for transform

    Input:
        img (cv2 img): image to transform
        random_seed (float, optional): random seed for repeatability
    """
    if random_seed is not None: r.seed(random_seed)

    diag = r.uniform(130, 256)
    vert = 2*10**(r.uniform(-1,1)/3)
    horiz = 2*10**(r.uniform(-1,1)/8)
    img = make_border(img)
    img = apply_cylindrical_transformation(img, diag, vert, horiz)
    return img, diag, vert, horiz

def load_and_save(original_img_filepath, transformed_img_filepath):
    """
    loads logo, transforms randomly, then saves transformed img accordingly, returns label to save
    """
    img = cv2.imread(original_img_filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Couldnt load image: ', os.path.basename(original_img_filepath))
        return None, None, None
    cyl_img, d, v, h = random_cylindrical_transform(img, random_seed=None)
    transformation_matrix = generate_random_perspective_matrix()
    combo_img = apply_perspective_transform(cyl_img, transformation_matrix)
    # print(transformed_img_filepath)
    cv2.imwrite(transformed_img_filepath, combo_img)
    h_, w_ = img.shape[:2]
    return d,v,h, h_,w_, transformation_matrix

def create_preliminary_dataset(dataset_dir=os.path.join(os.getcwd())):

    # get paths
    original_path=os.path.join(dataset_dir, 'original_256')
    transform_path=os.path.join(dataset_dir, 'transformed_256')
    labels_file=os.path.join(dataset_dir, 'labels.csv')

    with open(labels_file,'w', newline='') as file:
        # headers
        file.write('original_image_file,transformed_image_file,cyl_diag,cyl_vert,cyl_horiz,a,b,d,e,g,h\n')
        for imgfile in tqdm(os.listdir(original_path)):
            for i in range(10):
                # name the img transformed with cylinder warping as name_t.ext
                second_transform = False
                name, ext = os.path.splitext(imgfile)
                imgfile_t = name + '_t' + str(i) + ext 
                o_file = os.path.join(original_path, imgfile)
                t_file = os.path.join(transform_path, imgfile_t)
                out = load_and_save(o_file, t_file)
                if out is None: continue
                else: d, v, h, h_, w_, transformation_matrix = out
                line = [imgfile, imgfile_t, str(d), str(v), str(h), str(transformation_matrix[0,0]), str(transformation_matrix[0,1]), str(transformation_matrix[1,0]), str(transformation_matrix[1,1]), str(transformation_matrix[2,0]), str(transformation_matrix[2,1])]
                file.write(','.join(line))
                file.write('\n')

                # name the img transformed with cylinder warping as name_t.ext
                second_transform = True
                name, ext = os.path.splitext(imgfile)
                imgfile_tt = name + '_tt' + str(i) + ext 
                o_file = os.path.join(transform_path, imgfile_t)
                t_file = os.path.join(transform_path, imgfile_tt)
                out = load_and_save(o_file, t_file)
                if out is None: continue
                else: d, v, h, h_, w_, transformation_matrix = out
                line = [imgfile_t, imgfile_tt, str(d), str(v), str(h), str(h_), str(w_), str(transformation_matrix[0,0]), str(transformation_matrix[0,1]), str(transformation_matrix[1,0]), str(transformation_matrix[1,1]), str(transformation_matrix[2,0]), str(transformation_matrix[2,1])]
                file.write(','.join(line))
                file.write('\n')

# # Processing all images in the original_images directory
# for filename in os.listdir('original_images'):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         transform_and_save(os.path.join('original_images', filename))

if __name__ == "__main__":
    # test_transformation('3m.jpg')
    # seed = r.randint(0,1000)
    # print(seed)
    # test_random_transformation('lego.jpg', seed)
    create_preliminary_dataset()