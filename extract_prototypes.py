import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

prototypes = []
for i in os.listdir('reference'):
    img = cv2.imread('reference/'+i).astype(np.float32)
    img = img.mean(-1).astype(np.uint8)
    ret, thresh = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh)
    plt.show()
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    prototypes.append([i[:-4], cnt[0]])
print(len(prototypes))
print(prototypes[0][0])
with open('protoypes.pkl', 'wb') as f:
    pickle.dump(prototypes, f)