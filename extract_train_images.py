import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

with open('train/labels.txt','r') as f:
    labels = f.readlines()

for i in range(1):
    img_path = 'train/'+str(i)+'.png'
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    corners = [img[0,0].tolist(),img[0,w-1].tolist(),img[h-1,w-1].tolist(),img[h-1,0].tolist()]
    corners = [tuple(c) for c in corners]
    bg = max(set(corners), key=corners.count)
    
    fg_mask = np.logical_not(np.all(img==bg,-1))
    
    kernel = np.ones((5, 5), np.uint8)
    
    # Using cv2.erode() method 
    fg_mask = cv2.erode(fg_mask.astype(np.uint8), kernel)
    # plt.imshow(fg_mask)
    # plt.show()

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = cnt[:3]

    cnt.sort(key=lambda x : np.amin(x[:,:,0]).item())

    code = []
    curr_img_labels = labels[i][:-1].split(',')
    for idx, c in enumerate(cnt[:3]):
        l_x = max(np.amin(c[:,:,0]).item()-20, 0)
        r_x = min(np.amax(c[:,:,0]).item()+20, w)
        loc_img = np.clip(fg_mask[:,l_x:r_x].astype(np.float32)*255, 0,255)
        # LOC IMG IS THE LOCALISED LETTER
        plt.imshow(loc_img)
        plt.title(curr_img_labels[idx])
        plt.show()
        loc_img = cv2.resize(loc_img.astype(np.uint8),(32,32))

        # print(np.amax(loc_img))
        # img_dir = 'train_data_2/'+curr_img_labels[idx]
        # if not os.path.exists(img_dir):
        #     os.makedirs(img_dir)
        # cv2.imwrite(os.path.join(img_dir, str(i)+'_'+str(idx)+'.png'),loc_img)
        
