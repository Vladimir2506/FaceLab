#!usr/bin/python
import time
import scipy.io as sio
import cv2
import numpy as np
import os.path as op
from skimage import transform as trans

targetSize = [112, 96]
src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], 
      dtype = np.float32 )
LIST_PATH = './vggface2_train.txt'
NO_DETECT_PATH = './no_detect_idx.mat'


def ReadImgPathFromMeta(pathMeta) :
	
	assert isinstance(pathMeta, str)

	imgList = []

	with open(pathMeta, 'r') as metaData :
		for lines in metaData.readlines() :
			imgList.append(lines.strip())

	return imgList

def ProcessImg(imgOri, boundingBox, landMark) :
	
	imgCrp = None
	M = None

	try :
		if targetSize[1] == 112 :
			src[:, 0] += 8.0
		dst = landMark.astype(np.float32)
		tform = trans.SimilarityTransform()
		tform.estimate(dst, src)
		M = tform.params[0:2, :]
		imgCrp = cv2.warpAffine(imgOri, M, (targetSize[1], targetSize[0]), borderValue = 0.0)
	except :
		imgCrp = None

	return imgCrp

def main() :
	
	noFaces = []
	failure = []
	noDetect = sio.loadmat(NO_DETECT_PATH)
	for nil in noDetect['no_detect_idx'][0] :
		noFaces.append(nil)

	path = LIST_PATH	

	dataList = ReadImgPathFromMeta(path)

	ID = 0
	cnt = 0

	for data in dataList :
		ID = ID + 1
		if ID in noFaces :
			failure.append(data)
			continue

		#if ID == 2 :
		#	break

		begin = time.time()
		
		img = cv2.imread(data)
		fileName, ext = op.splitext(data)
		bName = fileName + '_bb.mat'
		lName = fileName + '_fp.mat'
		bbox = sio.loadmat(bName)
		lndmrk = sio.loadmat(lName)
		BBArr = np.array(bbox['bboxes'], dtype = np.float32)
		FPArr = np.array(lndmrk['facial5point'], dtype = np.float32)
		ret = ProcessImg(img, BBArr, FPArr)
		rName = fileName
		if targetSize[1] == 112 :
			rName += '_112*112_arc.jpg'
		else :
			rName += '_112*96_arc.jpg'
		print rName
		cv2.imwrite(rName, ret)

		end = time.time()

		# cv2.imshow('test', ret)
		# cv2.waitKey(0)

		
		if ret is not None :
			print ID ,'th Image cropped.'
		else :
			print ID, 'th Image failed.'
			failure.append(data)
		
		print 'Elapsed time: ', end - begin
	
	if len(failure) > 0 :
		print 'Failed to crop:'
	for failImage in failure :
		print failImage


if __name__ == '__main__':
	main()
