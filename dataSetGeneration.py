import numpy as np
from glob import glob
import random, argparse, pickle, cv2

def createData(imagePath,typeData):
    if(typeData == 'train'):
        # Train image size: 240x320 with rho = 32
        height = 240
        width = 320
        rho = 32
        sizePatch = 128
    elif(typeData == 'test'):
        # Test image size: 480x640 with rho = 64
        height = 480
        width = 640
        rho = 64
        sizePatch = 256

    #--Resize loaded image--#
    image = cv2.imread(imagePath,0)                         # Read image
    image = cv2.resize(image,(width,height))                # Resize image

    #--Random point in image--#
    x = random.randint(rho, width - rho - sizePatch)        # not exceed horizontal borders
    y = random.randint(rho, height - rho - sizePatch)       # not exceed vertical borders

    #--Draw the square--#
    point1 = (x, y)                                         # top-left
    point2 = (x + sizePatch, y)                             # top-right
    point3 = (x, y + sizePatch)                             # bottom-left
    point4 = (x + sizePatch, y + sizePatch)                 # bottom-right
    imageCorners = np.array([point1,point2,point4,point3])

    #--Get patches--#
    Ip = image[ y:y + sizePatch, x:x + sizePatch ]                  # Patch of original image
    randomPerturb = np.random.randint(low=-rho,high=rho,size=(4,2)) # Random values for perturbation
    imagePerturbedCorners = imageCorners + randomPerturb    # Perturb square randomly
    H = cv2.getPerspectiveTransform(np.float32(imageCorners), \
        np.float32(imagePerturbedCorners))                  # Homography, H
    H_inv = np.linalg.inv(H)                                # H^(-1)
    imageWarped_H = cv2.warpPerspective(image, H, (width,height))

    imageWarped = cv2.warpPerspective(image, H_inv, (width,height)) # Warp image using H^(-1)
    Ip_prime = imageWarped[ y:y + sizePatch, x:x + sizePatch ]      # Patch of perturbed image

    #--Features & Labels for network--#
    imageFeature = np.dstack((Ip,Ip_prime))                 # 2-channel image
    H_4point = imagePerturbedCorners - imageCorners         # 4-point matrix

    #--For DEMO of data created--#
    homographyData = [image, imageWarped_H, Ip, Ip_prime,\
                        imageCorners, imagePerturbedCorners]

    return imageFeature, H_4point.flatten(), homographyData   # Data Set, sample patch and patch points


def main():
    # Receive input from user to separately generate data for training, validation and testing
    parser = argparse.ArgumentParser()
    parser.add_argument("-P","--path_to_folder", help="indicate path to folder of image data (jpeg images)",type=str)
    parser.add_argument("-N","--samples",help="number of data sample to be created",type=int)
    parser.add_argument("-F","--filename",help="name of pickle file to be created (.p file)",type=str)
    parser.add_argument("-T","--type",help="train or test data",type=str,default="train")
    parser.add_argument("-a","--append",help="append new data to an existing pickle file",action='store_true')
    args = parser.parse_args()

    #### DATA SET GENERATION ###
    imageSet = glob("./" + args.path_to_folder + "/*.jpg")

    print("-- GENERATE DATA --\n")

    H_features = []
    H_labels = []

    for dataCount in range(args.samples):
        if(args.type == 'train'):   # for training (patch lower size)
            image = imageSet[random.randint(0,len(imageSet)-1)]                 # Randomly pick images in folder
            feature,label,_ = createData(image,args.type)   # Patches + 4-point
        elif(args.type == 'test'):  # for testing (patch larger size )
        	image = imageSet[dataCount % 40775]                                     # Unique image
        	feature_temp,label,_ = createData(image,args.type)  # Patches(256,256) + 4-point
        	feature = cv2.resize(feature_temp,(128,128))                            # Resize patch to (128,128)
        H_features.append(feature)          # (128,128,2)
        H_labels.append(label)              # (8,1)

        if( ((dataCount+1) % 200 == 0) ):
            print("{} samples generated".format(dataCount+1))

    print("{} samples generated".format(dataCount+1))

    print("\nSaving data set in pickle format\n")

    H_features = np.stack(H_features)       # Concatenate features
    H_labels = np.stack(H_labels)           # Concatenate labels
    H_data = [H_features,H_labels]          # Save features and labels in list

    pkl_file = open(args.filename,"wb")     # Save to pickle
    pickle.dump(H_data,pkl_file)
    pkl_file.close()                        # Close pickle file

    print("-- Data set generation FINISHED --")

if __name__ == "__main__":
    main()
