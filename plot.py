import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np


def predict(img_path):
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    corners = [img[0,0].tolist(),img[0,w-1].tolist(),img[h-1,w-1].tolist(),img[h-1,0].tolist()]
    corners = [tuple(c) for c in corners]
    bg = max(set(corners), key=corners.count)
    
    fg_mask = np.logical_not(np.all(img==bg,-1))
    plt.imshow(fg_mask)
    plt.show()

    kernel = np.ones((5, 5), np.uint8)
    
    # Using cv2.erode() method 
    fg_mask = cv2.erode(fg_mask.astype(np.uint8), kernel)
    plt.imshow(fg_mask)
    plt.show()

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = cnt[:3]

    cnt.sort(key=lambda x : np.amin(x[:,:,0]).item())

    with open('protoypes.pkl','rb') as f:
        prototypes = pickle.load(f)

    code = []
    for c in cnt[:3]:
        img = np.zeros((h,w),dtype=np.uint8)
        img = cv2.drawContours(img, [c], 0, 255, -1)
        l_x = max(np.amin(c[:,:,0]).item()-20, 0)
        r_x = min(np.amax(c[:,:,0]).item()+20, w)
        loc_img = fg_mask[:,l_x:r_x]
        # LOC IMG IS THE LOCALISED LETTER
        res = min(prototypes, key=lambda x : cv2.matchShapes(c, x[1], 1, 0.0))
        code.append(res[0])
        plt.imshow(loc_img)
        plt.title(res[0])
        plt.show()
    return code

predict('train/0.png')
