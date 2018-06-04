#coding=utf-8
from ImageServer import ImageServer
import numpy as np

imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
boundingBoxFiles = ["../data/py3boxesLFPWTrain.pkl", "../data/py3boxesHelenTrain.pkl", "../data/py3boxesAFW.pkl"]

datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect') # 훈련을하기 위해 bbx를 사용하지 않는 것과 동일하게, feature point interception box를 직접 사용하십시오.
trainSet.PrepareData(imageDirs, None, meanShape, 0, 2, True) # 이미지 이름 목록, 이미지 랜드 마크 목록, 해당 이미지의 bbx 목록 및 도구 모음을 준비하십시오. 저를 괴롭히는 이유는 startIdx = 100, nImgs = 100000 및 300W 데이터 세트에는 이미지가 많지 않기 때문입니다. 准备好图片名list，对应图片landmark的list，和对应图片的bbx的list，和meanshape。令我疑惑的是，startIdx=100，nImgs=100000，,300W数据集可没有那么多图片
trainSet.LoadImages() # 그림을 읽고 각 그림의 meanShape를 조정하십시오.
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25]) # 변위 0.2, 회전 20도, 축척 + -0.25
# import pdb; pdb.set_trace()
trainSet.NormalizeImages() # Deavered, 표준 편차로 나눈 값 <<< 去均值，除以标准差
# trainSet.Save(datasetDir) # 사전으로 저장, 키는 'imgs', 'initlandmarks', 'gtlandmarks' <<< 保存成字典形式，key为'imgs'，'initlandmarks'，'gtlandmarks'

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
import pdb; pdb.set_trace()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)