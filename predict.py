import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from my_cnn import MyCNNGAP
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

def predict(img_path, model, mapping):
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
	batch = []
	for c in cnt[:3]:
		l_x = max(np.amin(c[:,:,0]).item()-20, 0)
		r_x = min(np.amax(c[:,:,0]).item()+20, w)
		loc_img = np.clip(fg_mask[:,l_x:r_x].astype(np.float32)*255, 0,255)
		# LOC IMG IS THE LOCALISED LETTER
		loc_img = cv2.resize(loc_img.astype(np.uint8),(32,32))
		# plt.imshow(loc_img)
		# plt.show()
		batch.append(loc_img)
		# print(np.amax(loc_img))
	
	batch = np.stack(batch, axis=0).reshape(-1, 1, 32, 32).astype(np.float32)
	batch = batch/255.0
	batch = torch.tensor(batch)
    
	# print(batch.size())
	with torch.no_grad():
		preds = model(batch).numpy()
		# print(preds.shape)
		preds = np.argmax(preds, 1)
		# print(preds.shape)
	pred_code = ','.join([mapping[preds[idx]] for idx in range(3)])
	return pred_code

		  
		
def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration
	mapping = {0: 'ALPHA', 1: 'BETA', 2: 'CHI', 3: 'DELTA', 4: 'EPSILON',
		  5: 'ETA', 6: 'GAMMA', 7: 'IOTA', 8: 'KAPPA', 9: 'LAMDA', 10: 'MU',
		11: 'NU', 12: 'OMEGA', 13: 'OMICRON', 14: 'PHI', 15: 'PI', 
		16: 'PSI', 17: 'RHO', 18: 'SIGMA', 19: 'TAU', 20: 'THETA', 21: 'UPSILON', 22: 'XI', 23: 'ZETA'}
	labels = []
	model = MyCNNGAP()
	model.load_state_dict(torch.load('model_correct.pth', map_location='cpu'))
	model.eval()
	for i in filenames:
		labels.append(predict(i, model, mapping))

	return labels