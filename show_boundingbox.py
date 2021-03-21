import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import random
import time


def showAnnotation(base_dir, setName, fileName):
    # create figure
    fig, ax = plt.subplots()

    img = cv2.imread(os.path.join(
        base_dir, 'images', setName, fileName + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    (hImg, wImg) = img.shape[:2]

    ax.imshow(img)
    ax.axis('off')

    annotations = open(os.path.join(base_dir, 'labels', setName, fileName + '.txt')
                       ).read().splitlines()
    print(annotations)
    for box in annotations:
        content = box.split()
        boxType = int(content[0])
        x = int(float(content[1]) * wImg)  # center of box
        y = int(float(content[2]) * hImg)  # center of box
        w = int(float(content[3]) * wImg)
        h = int(float(content[4]) * hImg)

        print('x: ', x, ' y: ', y, ' w: ', w, ' h: ', h)

        x = int(x-w/2)
        y = int(y-h/2)

        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate((str(boxType) + ", x: " + str(x) + ", y: " + str(y)),
                    (x, y-16),
                    color='w',
                    weight='bold',
                    bbox=dict(alpha=.5, fc="k"),
                    fontsize=6,
                    ha='left',
                    va='bottom')

    name = base_dir + '/annotated/' + setName + '/' + fileName + '.png'

    # plt.show()
    plt.savefig(name, bbox_inches='tight')


base_dir = 'yolo_dataset'
imgFile = 'IMG_3654_JPG.rf.f4311000e64ae0d17e346a006a9a3e23'

#showAnnotation(base_dir, 'train', imgFile)


def allImages(base_dir, setName):
    path = os.path.join(base_dir, 'images', setName)
    listDir = os.listdir(path)
    # print(listDir)

    for imgFile in listDir:
        name = imgFile[:-4]
        showAnnotation(base_dir, setName, name)


allImages(base_dir, 'temp')
