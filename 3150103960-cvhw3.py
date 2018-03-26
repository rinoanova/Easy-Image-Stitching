#!/usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import cv2
import os

class Stitcher:
	def calHomo(self, imgs, ratio=0.75, reprojThresh=4.0):
		# Step 1: 从输入的两张图片中检测关键点并计算 SIFT 特征子
		# 要用第二张图片变形去符合第一张，所以倒着传进来
		(imgB, imgA) = imgs
		(kpsA, featuresA) = self.doSIFT(imgA)
		(kpsB, featuresB) = self.doSIFT(imgB)

		# Step 2: 匹配两图像间的特征
		match = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)

		# 如果匹配为 None ，则匹配不足以创造全景图
		if match is None:
			return None
		
		(matches, H, index) = match

		return (matches, H, index, kpsA, kpsB)

	def doSIFT(self, image):
		# SIFT
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sift = cv2.FeatureDetector_create("SIFT")
		kps = sift.detect(gray)
		sift2 = cv2.DescriptorExtractor_create("SIFT")
		(kps, features) = sift2.compute(gray, kps)
		
		# 把关键点转为 Numpy 数组
		kps = np.float32([kp.pt for kp in kps])
		
		# 返回关键点和特征的元组
		return (kps, features)
	
	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		# cv2.DescriptorMatcher_create(arg) 这个方法是 OpenCV 自带的方法：循环两张图片中所有的特征子，计算它们之间的距离，并且找到每对特征子之间的最小距离。
		# BruteForce: 这个参数意思是暴力尝试所有可能的匹配，所以总能够找到最佳的匹配。如果想要提高算法速度并且只需要相对好的匹配（而不是最好的），往往使用 FlannBased 这个参数。
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		# 在两个特征向量集之间使用 kNN 算法进行匹配，并且设 k = 2 （就是返回每个特征向量的前两个匹配）
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# 上面计算得到的 raw 里面有一些 false-positive 的，干掉它们（加入 Lowe's ratio 这个参数）
		# 一般 Lowe's ratio 在 [0.7, 0.8]
		for m in rawMatches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# 计算一个 homography 至少需要 4 个 matches ，如果不符合这个条件就返回 None
		if len(matches) > 4:
			# 构造两个点集
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# 计算两个点集之间的 homography
			(H, index) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
			
			# 返回 (matches, homography 矩阵, 各匹配点的匹配状况)
			return (matches, H, index)
		return None

	# 生成 SIFT 特征子的匹配可视化
	def drawMatches(self, imgA, imgB, kpsA, kpsB, matches, index):
		# 初始化输出图像
		(hA, wA) = imgA.shape[:2]
		(hB, wB) = imgB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imgA
		vis[0:hB, wB:] = imgB

		for((trainIdx, queryIdx), s) in zip(matches, index):
			# 只匹配那些成功匹配的关键点
			if s == 1:
				# 画匹配点之间的连线
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0])+wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis

if __name__ == "__main__": 
	# 键入一些参数
	dir = input('键入图像所在文件夹的绝对地址(例如 "/Users/Renne/testImgs/"')
	n = input("键入图像张数:")
	ifShowWarp = input('是否显示变形图 ("T" or "F"):')
	ifShowVis = input('是否显示特征子匹配图 ("T" or "F"):')
	
	# 初始参数设置
	n=int(n)
	boolShowWarp = False
	boolShowVis = False
	if ifShowWarp=='T':
		boolShowWarp = True
	if ifShowVis=='T':
		boolShowVis = True
	stitcher = Stitcher()
	
	H = np.zeros((n,3,3))
	ref = n/2+1
	for i in range(1, n):
		imgA = cv2.imread(dir+str(i)+'.jpg')
		imgB = cv2.imread(dir+str(i+1)+'.jpg')
		if i<ref:
			H[i,:,:]=stitcher.calHomo([imgB, imgA])[1]
		if i>=ref:
			H[i,:,:]=stitcher.calHomo([imgA, imgB])[1]
		if boolShowVis==True:
			args = stitcher.calHomo([imgB, imgA])
			vis = stitcher.drawMatches(imgA, imgB, args[3], args[4], args[0], args[2])
			cv2.imshow("vis"+"pic"+str(i)+"-"+str(i+1), vis)
		# ref 不参与计算
	if n !=2 and n!=3:
		if n%2!=0:
			refH = (n-1)/2
		else:
			refH = n/2
		for i in range(refH-1, 0, -1):
			H[i,:,:]=H[i,:,:].dot(H[i+1,:,:])
		if n!=4 :
			for i in range(refH+2, n):
				H[i,:,:]=H[i,:,:].dot(H[i-1,:,:])

	result = np.zeros((imgA.shape[0],imgA.shape[1]*n,3), dtype="uint8")
	for i in range(1,ref):
		img = cv2.imread(dir+str(i)+'.jpg')
		H[i,:,:]=np.mat(H[i,:,:])
		T=np.mat(np.float32([[1,0,img.shape[1]],[0,1,0], [0,0,1]]))
		temp = cv2.warpPerspective(img, T*H[i,:,:],(img.shape[1]*n, img.shape[0]))
		
		for k in range(0, result.shape[1]):
			if temp[:,k].all() == True:
				llen = k
				break
			else:
				llen = 0
		for k in range(result.shape[1]-1,llen,-1):
			if temp[:,k].all() == True:
				rlen = k
				break
			else:
				rlen = temp.shape[1]
		if boolShowWarp==True:
			cv2.imshow("warp"+str(i), temp)
		result[:,llen:rlen:,]=temp[:,llen:rlen]
	leftlen=rlen
	for i in range(n,ref, -1):
		img = cv2.imread(dir+str(i)+'.jpg')
		H[i-1,:,:]=np.mat(H[i-1,:,:])
		temp = cv2.warpPerspective(img, H[i-1,:,:],(img.shape[1]*n, img.shape[0]))
		
		# 切掉纯黑的部分
		for k in range(0, result.shape[1]):
			if temp[:,k].all() == True:
				llen = k
				break
			else:
				llen = 0
		for k in range(result.shape[1]-1,llen,-1):
			if temp[:,k].all() == True:
				rlen = k
				break
			else:
				rlen = temp.shape[1]
		if llen==0 and rlen==temp.shape[1]:
			for k in range(0, result.shape[1]):
				if temp[:,k].any() == True:
					llen = k
					break
			for k in range(result.shape[1]-1,llen,-1):
				if temp[:,k].any() == True:
					rlen = k
					break
		#temp = temp[:, 0:length]
		if boolShowWarp==True:
			cv2.imshow("warp"+str(i), temp)
		result[:,leftlen+llen:leftlen+rlen,:]=temp[:,llen:rlen]
	img = cv2.imread(dir+str(ref)+'.jpg')
	result[:,leftlen:leftlen+img.shape[1],:]=img

	for k in range(0,result.shape[1]):
		if result[:,k].all()==True or result[:,k].any() == True:
			llen=k
			break
		else:
			llen=0
	result = result[:,llen:]
	
	for k in range(result.shape[1]-1,llen,-1):
		if result[:,k].all() == True or result[:,k].any() == True:
			rlen = k
			break
		else:
			rlen = result.shape[1]
	result = result[:, 0:rlen]
	
	cv2.imshow("result", result) 
	cv2.waitKey(0)