#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*  * *** *  * *  *
*  *  *   **  *  *
****  *   **  *  *
*  *  *   **  *  *
*  * **  *  * ****

@File     :KeyPointStu/showPoint.py
@Date     :2020/12/3 下午7:07
@Require  :numpy、pycocotools、opencv
@Author   :hjxu2016， https://blog.csdn.net/hjxu2016/
@Funtion  :读取Coco数据集，并显示关键点到图像上
"""
import numpy as np
from pycocotools.coco import COCO
import cv2

aColor = [(0, 255, 0, 0), (255, 0, 0, 0), (0, 0, 255, 0), (0, 255, 255, 0)]
annFile = './coco/2017/annotations/person_keypoints_val2017.json'
img_prefix = './coco/2017/val2017'

# initialize COCO api for instance annotations
coco = COCO(annFile)

# getCatIds(catNms=[], supNms=[], catIds=[])
# 通过输入类别的名字、大类的名字或是种类的id，来筛选得到图片所属类别的id
catIds = coco.getCatIds(catNms=['person'])
print("catIds: ", catIds)
# getImgIds(imgIds=[], catIds=[])
# 通过图片的id或是所属种类的id得到图片的id
# 得到图片的id信息后，就可以用loadImgs得到图片的信息了
# 在这里我们随机选取之前list中的一张图片
imgIds = 425226
img = coco.loadImgs(imgIds)[0]
matImg = cv2.imread('%s/%s' % (img_prefix, img['file_name']))
# # 通过输入图片的id、类别的id、实例的面积、是否是人群来得到图片的注释id
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

# # 通过注释的id，得到注释的信息
anns = coco.loadAnns(annIds)
print(anns)
for ann in anns:
    sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton']) - 1
    kp = np.array(ann['keypoints'])

    x = kp[0::3]
    y = kp[1::3]
    v = kp[2::3]

    for sk in sks:
        # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        c = aColor[np.random.randint(0, 4)]
        if np.all(v[sk] > 0):
            # 画点之间的连接线
            cv2.line(matImg, (x[sk][0], y[sk][0]), (x[sk][1], y[sk][1]), c, 1)
    for i in range(x.shape[0]):
        c = aColor[np.random.randint(0, 4)]
        cv2.circle(matImg, (x[i], y[i]), 2, c, lineType=1)

cv2.imshow("show", matImg)
cv2.waitKey(0)

