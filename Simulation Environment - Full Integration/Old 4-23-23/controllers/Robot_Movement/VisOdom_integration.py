# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:08:09 2023

@author: SrTempest
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:08:09 2023

@edited By: Demargio Glanville
"""

#import os
import numpy as np
import cv2 as cv
import struct
from controller import Camera


#from matplotlib import pyplot as plt

class VisualOdom():

    #----- Constructor -----#
    def __init__(self, cameras, timestep):
        # Loading Data. -Nicolai Nielsen
        self.K, self.P = self._solve_calib(cameras) 
        self.images = []
        #self.loadImage(cameras)
        self.hz = 1/(timestep/1000)
        
        
        self.orb = cv.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=100) #Orignally 50
        self.flann = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    
    # Loading Data. -Nicolai Nielsen
    @staticmethod
    def _solve_calib(cam):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        
        P = np.zeros((3,4))
        
        
        ux = cam.getWidth()/2
        uy = cam.getHeight()/2
        f = ux/np.tan(cam.getFov()/2)
        y = 0
        
        K = np.array([[f, y, ux], [0, f, uy], [0, 0, 1]])
        
        

        P[0:3, 0:3] = K

        return K, P

    #@staticmethod
    def loadImage(self,cam):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
       
        
        pict = np.frombuffer(cam.getImage(), dtype=np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        img = np.copy(pict)
        grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.images.append(grayImg)
        
        if len(self.images) > 2:
            
            self.images.pop(0)
        
        return 

    @staticmethod
    def _quick_Transf(R, t):
        
        T = np.eye(4, dtype=np.float64)
        
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        
        return T
    @staticmethod
    def _rot_2_eul(R):
        
        eul = []
        
        if R[2,0] != 1:
           
            theta1 = -np.arcsin(R[2,0])
            psi1 = np.arctan2((R[2,1]/np.cos(theta1)),(R[2,2]/np.cos(theta1)))       
            phi1 = np.arctan2((R[1,0]/np.cos(theta1)),(R[0,0]/np.cos(theta1)))
            eul = [phi1, theta1, psi1]
           
        else:
           
            phi = 0
           
            if R[2,0] == -1:
               
                theta = np.pi/2
                psi = phi + np.arctan2(R[0,1], R[0,2])
                eul = [phi, theta, psi]
               
            else:
               
                theta = -np.pi/2
                psi = -phi + np.arctan2(-R[0,1], -R[0,2])
                eul = [phi, theta, psi]
        
        return eul
    
    def calcVel(self,newPose,oldPose):
        
        sub = np.subtract(newPose,oldPose) 
        hyp = np.sqrt(sub[0,3]**2 + sub[1,3]**2)
        v = hyp/(1/self.hz)
        
        deltaEul = np.subtract(self._rot_2_eul(newPose[0:3, 0:3]),self._rot_2_eul(oldPose[0:3, 0:3]))
        
        omega = deltaEul/(1/self.hz)
     
        return v, omega
    
    @staticmethod
    def state2transf(actualState):
    
        t = np.array([actualState[0], actualState[1], 0])
         
        R = np.array([[np.cos(actualState[2]), -np.sin(actualState[2]), 0],[np.sin(actualState[2]), np.cos(actualState[2]), 0], [0, 0, 1]])

        T = np.eye(4)
         
        T[0:3, 0:3] = R
        T[0:3, 3] = t
     
        return T
    
    
    def transf2state(self, newPose, currentPose):
    
        stateBased = np.zeros((5))
    
        stateBased[0] = newPose[0, 3]
        stateBased[1] = newPose[1, 3]

        eul = self._rot_2_eul(newPose[0:3, 0:3])
        stateBased[2] = eul[1]
        
        v, omega = self.calcVel(newPose,currentPose)
        stateBased[3] = v
        stateBased[4] = omega[1]
     
        return stateBased 
        
    # Key Point Detection
    def getKeyPoints(self):
       """
    
        Parameters
        ----------
        i : int
            images index.

        Returns
        -------
        kp1 : ndarray
              Key Points of image one.
              
        kp2 : ndarray
              Key Points of image two.
              
        des1 : -
              Descriptors for the Key Points of image one.
              
        des2 : -
              Descriptors for the Key Points of image two.

       """ 
       
       # Key point detection is the process of finding unique points in an 
       # image. This is often done through a gradient comparison. Over a small
       # window, the change in average RGB value per pixel is evaluated. If
       # the threshold is reached, the area (colection of pixels) will be 
       # marked as a "Key Point".
       #
       # The key points are then given descriptors, an identifier for the Key
       # Point. Simple descriptors are a vector which align themselves with the
       # decline of the gradient. These are later used to match Key Points
       # across images.
       kp1, des1 = self.orb.detectAndCompute(self.images[1],None)
       kp2, des2 = self.orb.detectAndCompute(self.images[0],None)
       
       return kp1, des1, kp2, des2
       
    # Matches Key Points and their descriptors across images.
    def matchMaker(self,cam):
    
        self.loadImage(cam)
       
        kp1, des1, kp2, des2 = self.getKeyPoints()
      
        match = self.flann.knnMatch(des1,des2,k=2)
      
        good = []
        
        for m,n in match:
            if m.distance < 0.8 * n.distance:
                good.append(m)
              
        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2
         
    def solvePose(self,q1,q2,actualState):

        # First, we compute the essential matrix. Using the good matches, we
        # first solve for the Fundamental Matrix using an algorithm like
        # RANSAC. Then the function solves for the Essential Matrix using the
        # Fundamental Matrix and the camera intrinsic data (normally given).
        E, _ = cv.findEssentialMat(q1,q2,self.K,threshold=1)
        
        # Then using the Essential Matrix, we can solve for Rotation Matrcies
        # and the position. These however, may not be correct because there
        # for possible solutions (according to the math) and all must be
        # compared to find the best solution. This solution will have the most
        # key points in its view.
        R1, R2, t = cv.decomposeEssentialMat(E)
        t = np.squeeze(t) 
        possibCombin = [[R1,t], [R1,-t], [R2,t], [R2,-t]]
        
        # Change State into a Transformation Matrix
        currentPose = self.state2transf(actualState)
        print(actualState)
        print(currentPose)
        
        # By Nicolai Neilsen
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._quick_Transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
        
        zSum = []
        relScales = []
        
        for j in range(len(possibCombin)):
            z_sum, scale = sum_z_cal_relative_scale(possibCombin[j][0], possibCombin[j][1])
            zSum.append(z_sum)
            relScales.append(scale)
            
        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(zSum)
        right_pair = possibCombin[right_pair_idx]
        relScale = relScales[right_pair_idx]
        R1, t = right_pair
        t = t * relScale
        
        T = self._quick_Transf(R1, t)
        
        new_pose = np.matmul(currentPose, np.linalg.inv(T))
        
        stateBased = self.transf2state(new_pose, currentPose)
        print(stateBased)
        return stateBased
        
        
        """------------------------------------------------------------
        From the Robot Movement File:
        
        # Set up Visual Odom
        vo = VisualOdom(left_camera)
        currentPose = np.eye(4)


    # Main loop:

        #Visual Odom execution
        
        if len(vo.images) < 2:
            vo.loadImage(left_camera)
        else:
            q1,q2 = vo.matchMaker(left_camera)
            currentPose, vel, eul = vo.solvePose(q1,q2,currentPose)
       
            print(currentPose, vel, eul)
            
        ------------------------------------------------------------"""