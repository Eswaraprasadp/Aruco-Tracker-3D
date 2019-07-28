
"""
Author: Eswara prasad
Domain: Signal Processing and ML
Sub-domain: Image processing
Functions: mean,mean_arr, add_velocity_values, max_area, drawCar, draw, compositeArray, idle, reshape, keyboard, main
Global variables: windowWidth, windowHeight, dictionary, parameters, mtx, dist, alpha, beta, cx, cy, cap, seen_ids 
"""

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import cv2.aruco as aruco
import numpy as np
import traceback
import sys

windowWidth = 800
windowHeight = 600

dictionary = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
parameters =  aruco.DetectorParameters_create()

# Load camera parameter
with np.load('../Data/camera_calibration.npz') as X:

    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    # mtx --> Camera matrix representing [[fx, 0, cx],
    #									  [0, fy, cy],
    #                                     [0, 0, p]], 
    #		  where fx = focal length in x direction
    #		        fy = focal length in y direction
    #		        cx = X center of camera
    #		        cy = Y center of camera

    # dist --> Radial distortion matrix
  
alpha = mtx[0,0]
beta = mtx[1,1]
cx = mtx[0,2]
cy = mtx[1,2]

"""
Function Name: mean
Output: Return the mean of a list
Logic: Sum of values divided the number of elements
Example Call: mean = mean(arr)
"""
    
def mean(arr):
    return sum(arr)/float(len(arr))

"""
Function Name: mean_arr(arr)
Output: Returns the mean of list of vertices of quadrilateral (ie. list of tuples) as a tuple
Logic: Calls mean(arr) for list of tuples
Example Call: average = mean_arr(arr)
"""

def mean_arr(arr):

    return (mean(arr[:,0]),mean(arr[:,1]))
    # Traversing through x and y of each vertex and calling mean of the obtained list

"""
Function Name: add_velocity_values
Output: Adds present velocity to saved list and computes average velocity of each vertex
Logic: Calls mean_arr for each vertex (list of tuples)
Example Call: add_velocity_values(saved_velocities, values, average_velocity)
"""

def add_velocity_values(saved, values, average):
    values = np.float32([values])
    if(len(saved) <= 50):
        saved = np.concatenate((saved, values)) # Saves velocities for 50 frames
    if(len(saved) > 50):
        saved = saved[1:] # If more than 50 values are stored pop first element to make the length 50 again

    for i in range(4):
        average[i] = mean_arr(saved[:,i])
        # First obtain list of each vertex in saved list eg. saved[:,2] returns the list of third vertex coordinates 

"""
Function name: max_area
Output: Returns the area of maximum bounding rectangle of a quadrilateral
Input: Vertices of quadrilateral
Logic: Area is product of difference of width and height
Example call: area = max_area(quad)
"""

def max_area(quad):
    quad = np.float32(quad)
    return ((max(quad[:,0]) - min(quad[:,0])) * (max(quad[:,1]) - min(quad[:, 1])))        

"""
Function name: drawCar
Output: Draws the car in 3d using OpenGL functions
Logic: Draws using glBegin() and glEnd()
Example call: drawCar()
"""

def drawCar():
	z = 1.5
    
	# Back window frame    
	glColor3f(206/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, 0.25, -z)
	glVertex3f(-3.0, 0.25, z)
	glVertex3f(-3.0, -1.0, z)
	glVertex3f(-3.0, -1.0, -z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.5, -z)
	glVertex3f(-3.0, 1.5, z)
	glVertex3f(-3.0, 1.0, z)
	glVertex3f(-3.0, 1.0, -z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-3.0, 0.25, -z)
	glVertex3f(-3.0, 0.25, -z+0.5)
	glVertex3f(-3.0, 1.0, -z+0.5)
	glVertex3f(-3.0, 1.0, -z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-3.0, 0.25, z-0.5)
	glVertex3f(-3.0, 0.25, z)
	glVertex3f(-3.0, 1.0, z)
	glVertex3f(-3.0, 1.0, z-0.5)
	glEnd()

	# Top
	glColor3f(240/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.5, -z)
	glVertex3f(-3.0, 1.5, z)
	glVertex3f(0.6, 1.5, z)
	glVertex3f(0.6, 1.5, -z)
	glEnd()

	# Bottom
	glColor3f(190/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, -1.0, -z)
	glVertex3f(-3.0, -1.0, z)
	glVertex3f(3.0, -1.0, z)
	glVertex3f(3.0, -1.0, -z)
	glEnd()

	# Front
	glColor3f(206/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(3.0, -1.0, -z)
	glVertex3f(3.0, 0.15, -z)
	glVertex3f(3.0, 0.15, z)
	glVertex3f(3.0, -1.0, z)

	# Lamp
	glColor3f(0.9,0.9,0.9)
	glVertex3f(3.006, -0.65, -z+0.101)
	glVertex3f(3.006, -0.35, -z+0.101)
	glVertex3f(3.006, -0.35, -z+0.601)
	glVertex3f(3.006, -0.65, -z+0.601)

	glVertex3f(3.006, -0.65, z-0.101)
	glVertex3f(3.006, -0.35, z-0.101)
	glVertex3f(3.006, -0.35, z-0.601)
	glVertex3f(3.006, -0.65, z-0.601)

	glColor3f(0,0,0)
	glVertex3f(3.006, -0.6, -z+1)
	glVertex3f(3.006, -0.37, -z+1)
	glVertex3f(3.006, -0.37, z-1)
	glVertex3f(3.006, -0.6, z-1)
	
	# Lamp2
	glColor3f(0.6,0.6,0.6)
	glVertex3f(3.005, -0.7, -z)
	glVertex3f(3.005, -0.3, -z)
	glVertex3f(3.005, -0.3, z)
	glVertex3f(3.005, -0.7, z)

	glVertex3f(2.9, -0.3, -z-0.0014)
	glVertex3f(3.0, -0.3, -z-0.0014)
	glVertex3f(3.0, -0.7, -z-0.0014)
	glVertex3f(2.9, -0.7, -z-0.0014)

	glVertex3f(2.9, -0.3, z+0.0014)
	glVertex3f(3.0, -0.3, z+0.0014)
	glVertex3f(3.0, -0.7, z+0.0014)
	glVertex3f(2.9, -0.7, z+0.0014)
	
	glColor3f(226/255, 152/255, 22/255)
	glVertex3f(2.95, -0.35, z+0.0015)
	glVertex3f(2.985, -0.35, z+0.0015)
	glVertex3f(2.985, -0.65, z+0.0015)
	glVertex3f(2.95, -0.65, z+0.0015)

	glVertex3f(2.95, -0.35, -z-0.0015)
	glVertex3f(2.985, -0.35, -z-0.0015)
	glVertex3f(2.985, -0.65, -z-0.0015)
	glVertex3f(2.95, -0.65, -z-0.0015)
	glEnd()

	# Front cover
	glColor3f(230/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(3.0, 0.15, -z)
	glVertex3f(1.2, 0.25, -z)
	glVertex3f(1.2, 0.25, z)
	glVertex3f(3.0, 0.15, z)
	glEnd()

	# Front window frame
	glColor3f(235/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(0.6, 1.5, -z)
	glVertex3f(0.6, 1.5, z)
	glVertex3f(0.65, 1.42, z)
	glVertex3f(0.65, 1.42, -z)

	glVertex3f(1.15, 0.34, -z)
	glVertex3f(1.15, 0.34, -z+0.1)
	glVertex3f(0.65, 1.42, -z+0.1)
	glVertex3f(0.65, 1.42, -z)

	glVertex3f(1.15, 0.34, z)
	glVertex3f(1.15, 0.34, z-0.1)
	glVertex3f(0.65, 1.42, z-0.1)
	glVertex3f(0.65, 1.42, z)

	glVertex3f(1.15, 0.34, -z)
	glVertex3f(1.15, 0.34, z)
	glVertex3f(1.2, 0.25, z)
	glVertex3f(1.2, 0.25, -z)
	glEnd()

	# Left above (window frame part)
	glColor3f(206/255, 20/255, 55/255)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.5, -z)
	glVertex3f(0.6, 1.5, -z)
	glVertex3f(0.696, 1.3, -z)
	glVertex3f(-3.0, 1.3, -z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.3, -z)
	glVertex3f(-3.0, 0.25, -z)
	glVertex3f(-2.5, 0.25, -z)
	glVertex3f(-2.5, 1.3, -z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-1.2, 1.3, -z)
	glVertex3f(-1.2, 0.25, -z)
	glVertex3f(-1.0, 0.25, -z)
	glVertex3f(-1.0, 1.3, -z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(1.2, 0.25, -z)
	glVertex3f(0.696, 1.3, -z)
	glVertex3f(0.496, 1.3, -z)
	glVertex3f(1.0, 0.25, -z)
	glEnd()

	# Left front door
	glBegin(GL_POLYGON)
	glVertex3f(1.2, 0.25, -z)
	glVertex3f(3.0, 0.15, -z)
	glVertex3f(3.0, -1.0, -z)
	glVertex3f(1.2, -1.0, -z)
	glEnd()

	# Left back door
	glBegin(GL_POLYGON)
	glVertex3f(1.2, 0.25, -z)
	glVertex3f(1.2, -1.0, -z)
	glVertex3f(-3.0, -1.0, -z)
	glVertex3f(-3.0, 0.25, -z)
	glEnd()

	# Right back door
	glBegin(GL_POLYGON)
	glVertex3f(1.2, 0.25, z)
	glVertex3f(1.2, -1.0, z)
	glVertex3f(-3.0, -1.0, z)
	glVertex3f(-3.0, 0.25, z)
	glEnd()

	# Right front door
	glBegin(GL_POLYGON)
	glVertex3f(1.2, 0.25, z)
	glVertex3f(3.0, 0.15, z)
	glVertex3f(3.0, -1.0, z)
	glVertex3f(1.2, -1.0, z)
	glEnd()

	# Right above (window frame part)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.5, z)
	glVertex3f(0.6, 1.5, z)
	glVertex3f(0.696, 1.3, z)
	glVertex3f(-3.0, 1.3, z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.3, z)
	glVertex3f(-3.0, 0.25, z)
	glVertex3f(-2.5, 0.25, z)
	glVertex3f(-2.5, 1.3, z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(-1.2, 1.3, z)
	glVertex3f(-1.2, 0.25, z)
	glVertex3f(-1.0, 0.25, z)
	glVertex3f(-1.0, 1.3, z)
	glEnd()

	glBegin(GL_QUADS)
	glVertex3f(1.2, 0.25, z)
	glVertex3f(0.696, 1.3, z)
	glVertex3f(0.496, 1.3, z)
	glVertex3f(1.0, 0.25, z)
	glEnd()

	# Side bottom
	glColor3f(165/255, 8/255, 37/255)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, -0.3, -z-0.0013)
	glVertex3f(3.0, -0.3, -z-0.0013)
	glVertex3f(3.0, -0.7, -z-0.0013)
	glVertex3f(-3.0, -0.7, -z-0.0013)

	glVertex3f(-3.0, -0.3, z+0.0013)
	glVertex3f(3.0, -0.3, z+0.0013)
	glVertex3f(3.0, -0.7, z+0.0013)
	glVertex3f(-3.0, -0.7, z+0.0013)
	glEnd()

	# Left mirror
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)
	glColor4f(190/255, 190/255, 190/255, 0.6)
	glBegin(GL_QUADS)
	glVertex3f(0.85, 0.5, -z-0.4)
	glVertex3f(0.85, 0.5, -z)
	glVertex3f(0.85, 0.25, -z)
	glVertex3f(0.85, 0.25, -z-0.4)
	glEnd()
	glDisable(GL_BLEND)

	glColor3f(180/255, 30/255, 30/255)
	glBegin(GL_QUADS)
	glVertex3f(0.85, 0.5, -z-0.4)
	glVertex3f(1, 0.5, -z-0.4)
	glVertex3f(1, 0.25, -z-0.4)
	glVertex3f(0.85, 0.25, -z-0.4)

	glVertex3f(0.85, 0.5, -z-0.4)
	glVertex3f(0.85, 0.5, -z)
	glVertex3f(1, 0.5, -z)
	glVertex3f(1, 0.5, -z-0.4)

	glVertex3f(1, 0.5, -z-0.4)
	glVertex3f(1, 0.5, -z)
	glVertex3f(1, 0.25, -z)
	glVertex3f(1, 0.25, -z-0.4)

	glVertex3f(0.85, 0.5, -z)
	glVertex3f(1, 0.5, -z)
	glVertex3f(1, 0.25, -z)
	glVertex3f(0.85, 0.25, -z)

	glVertex3f(0.85, 0.25, -z-0.4)
	glVertex3f(0.85, 0.25, -z)
	glVertex3f(1, 0.25, -z)
	glVertex3f(1, 0.25, -z-0.4)

	glEnd()

	# Right mirror
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)
	glColor4f(190/255, 190/255, 190/255, 0.6)
	glBegin(GL_QUADS)
	glVertex3f(0.85, 0.5, z+0.4)
	glVertex3f(0.85, 0.5, z)
	glVertex3f(0.85, 0.25, z)
	glVertex3f(0.85, 0.25, z+0.4)
	glEnd()
	glDisable(GL_BLEND)

	glColor3f(180/255, 30/255, 30/255)
	glBegin(GL_QUADS)
	glVertex3f(0.85, 0.5, z+0.4)
	glVertex3f(1, 0.5, z+0.4)
	glVertex3f(1, 0.25, z+0.4)
	glVertex3f(0.85, 0.25, z+0.4)

	glVertex3f(1, 0.5, z+0.4)
	glVertex3f(1, 0.5, z)
	glVertex3f(1, 0.25, z)
	glVertex3f(1, 0.25, z+0.4)

	glVertex3f(0.85, 0.5, z+0.4)
	glVertex3f(0.85, 0.5, z)
	glVertex3f(1, 0.5, z)
	glVertex3f(1, 0.5, z+0.4)

	glVertex3f(0.85, 0.5, z)
	glVertex3f(1, 0.5, z)
	glVertex3f(1, 0.25, z)
	glVertex3f(0.85, 0.25, z)

	glVertex3f(0.85, 0.25, z+0.4)
	glVertex3f(0.85, 0.25, z)
	glVertex3f(1, 0.25, z)
	glVertex3f(1, 0.25, z+0.4)
	glEnd()

	# Left window glass
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)
	glColor4f(90/255, 90/255, 90/255, 0.3)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.5, -z+0.01)
	glVertex3f(0.5, 1.5, -z+0.01)
	glVertex3f(1.2, 0.25, -z+0.01)
	glVertex3f(-3.0, 0.25, -z+0.01)
	glEnd()
	glDisable(GL_BLEND)
	
	# Right window glass
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)
	glColor4f(90/255, 90/255, 90/255, 0.3)
	glBegin(GL_QUADS)
	glVertex3f(-3.0, 1.5, z-0.01)
	glVertex3f(0.5, 1.5, z-0.01)
	glVertex3f(1.2, 0.25, z-0.01)
	glVertex3f(-3.0, 0.25, z-0.01)
	glEnd()
	glDisable(GL_BLEND)

	# Front window glass
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)
	glColor4f(90/255, 90/255, 90/255, 0.3)
	glBegin(GL_QUADS)
	glVertex3f(0.5, 1.5, -z)
	glVertex3f(0.5, 1.5, z)
	glVertex3f(1.2, 0.25, z)
	glVertex3f(1.2, 0.25, -z)
	glEnd()
	glDisable(GL_BLEND)

	#back window glass
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)
	glColor4f(90/255, 90/255, 90/255, 0.3)
	glBegin(GL_QUADS)
	glVertex3f(-2.99, 0.25, -z+0.5)
	glVertex3f(-2.99, 0.25, z-0.5)
	glVertex3f(-2.99, 1.0, z-0.5)
	glVertex3f(-2.99, 1.0, -z+0.5)
	glEnd()
	glDisable(GL_BLEND)

	# Car's Wheel
	glColor3f(0.0, 0.0, 0.0)
	quadric = gluNewQuadric()
	gluQuadricNormals(quadric, GLU_SMOOTH)
	gluQuadricTexture(quadric, GL_TRUE)
	glTranslatef(1.7,-1.0,-1.7)
	gluCylinder(quadric,0.6,0.6,0.2,15,15)
	gluDisk(quadric, 0, 0.6, 15, 15)
	glTranslatef(0.0,0.0,0.2)
	gluDisk(quadric, 0, 0.6, 15, 15)
	
	glTranslatef(0.0, 0.0, -0.2)
	glTranslatef(-3.3, 0.0, 0.0)
	gluCylinder(quadric,0.6,0.6,0.2,15,15)
	gluDisk(quadric, 0, 0.6, 15, 15)
	glTranslatef(0.0,0.0,0.2)
	gluDisk(quadric, 0, 0.6, 15, 15)
	
	glTranslatef(0.0, 0.0, -0.2)
	glTranslatef(0.0, 0.0, 3.2)
	gluCylinder(quadric,0.6,0.6,0.2,15,15)
	gluDisk(quadric, 0, 0.6, 15, 15)
	glTranslatef(0.0,0.0,0.2)
	gluDisk(quadric, 0, 0.6, 15, 15)
	
	glTranslatef(0.0, 0.0, -0.2)
	glTranslatef(3.3, 0.0, 0.0)
	gluCylinder(quadric,0.6,0.6,0.2,15,15)
	gluDisk(quadric, 0, 0.6, 15, 15)
	glTranslatef(0.0,0.0,0.2)
	gluDisk(quadric, 0, 0.6, 15, 15)
	
	glColor3f(1.0, 1.0, 1.0)
	gluDisk(quadric, 0.2, 0.4, 15, 15)
	glTranslatef(-3.3, 0.0, 0.0)
	gluDisk(quadric, 0.2, 0.4, 15, 15)
	glTranslatef(0.0, 0.0, -0.2)
	glTranslatef(0.0, 0.0, -3.2)
	gluDisk(quadric, 0.2, 0.4, 15, 15)
	glTranslatef(+3.3, 0.0, 0.0)
	gluDisk(quadric, 0.2, 0.4, 15, 15)    

"""
Function name: draw
Output: Draws the captured frame and car onto the screen. In PyOpenGL it is called as glutDisplayFunc(draw)
Logic: Tracks Aruco using OpenCV aruco library. It is projected onto window and a car is drawn using OpenGL by calling drawCar()
Example call: draw()
"""
    
def draw():    
    # Aruco
    ret, img = cap.read()
    if not ret:
        return
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary, parameters = parameters)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 8.0, mtx, dist)
    
    markers = np.float32(corners).reshape(-1,4,2)
    # Reshape to arrays containing four vertices with two elements (x and y coordinates)
    
    global seen_ids             
    
    if not ids is None:
        
        ids = ids.ravel()               
        
        for i in range(len(ids)):
            
            marker = markers[i]
            m_id = ids[i]                
            
            if m_id not in seen_ids:
                seen_ids[m_id] = {'vertices':marker, 'av_velocity':np.float32([(0, 0),(0, 0),(0, 0),(0, 0)]), 'saved_velocities':np.zeros((0,4,2),dtype=np.float32), 'seen':True, 't':0}
                # If the Marker ID is seen for the first time or again after a specific time, it is added to seen_ids.
                # Here t denotes the time it is unseen. It is 0 as long as it is seen

            else:
                marker_details = seen_ids[m_id]
                if(len(marker_details['saved_velocities']) > 0):
                    prev_velocity = marker_details['saved_velocities'][-1]

                else:
                    prev_velocity = marker_details['av_velocity']
                    # Initially previous velocity is average velocity ie. full of zeros
                
                new_velocity = []
                # Present velocity is new velocity
                
                for i in range(4):
                    vx = marker[i][0] - prev_velocity[i][0] # Velocity is present value of x or y minus the previous velocity
                    vy = marker[i][1] - prev_velocity[i][1]
                    
                    if(vx > 40 or vy > 40):
                        # If the velocity is too high (shaky) replace it with average velocity
                        new_velocity = marker_details['av_velocity']
                        break
                    
                    new_velocity.append((vx, vy))
                    # Add computed velocity to new velocity
                
                new_velocity = np.float32(new_velocity)
                add_velocity_values(marker_details['saved_velocities'],new_velocity, marker_details['av_velocity'])
                # Calls add_velocity_values for saving current velocity and computing average velocity

                marker_details['vertices'] = marker
                marker_details['seen'] = True
                marker_details['t'] = 0            
          
        
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #BGR-->RGB
    h, w = img.shape[:2]
    
    # This draws the 2d projected on the screen
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    
    # Enable / Disable
    glDisable(GL_DEPTH_TEST)    # Disable GL_DEPTH_TEST
    glDisable(GL_LIGHTING)      # Disable Light
    glDisable(GL_LIGHT0)        # Disable Light
    glEnable(GL_TEXTURE_2D)     # Enable texture map
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear Buffer
    glColor3f(1.0, 1.0, 1.0)    # Set texture Color(RGB: 0.0 ~ 1.0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glPushMatrix()

    glBegin(GL_QUADS)
    
    glTexCoord2d(0.0, 1.0)
    glVertex3d(-1.0, -1.0,  0)
    glTexCoord2d(1.0, 1.0)
    glVertex3d( 1.0, -1.0,  0)
    glTexCoord2d(1.0, 0.0)
    glVertex3d( 1.0,  1.0,  0)
    glTexCoord2d(0.0, 0.0)
    glVertex3d(-1.0,  1.0,  0)
    
    glEnd()

    glPopMatrix()
    
    # Enable / Disable
    glEnable(GL_DEPTH_TEST)     # Enable GL_DEPTH_TEST
    glEnable(GL_LIGHTING)       # Enable Light
    glEnable(GL_LIGHT0)         # Enable Light
    glDisable(GL_TEXTURE_2D)    # Disable texture map
    glEnable(GL_COLOR_MATERIAL)

    f = 1000.0  #far
    n = 1.0     #near

    ## Make projection matrix

    m1 = np.array([
    [(alpha)/cx, 0, 0, 0],
    [0, beta/cy, 0, 0],
    [0, 0, -(f+n)/(f-n), (-2.0*f*n)/(f-n)],
    [0,0,-1,0],
    ])
    glLoadMatrixd(m1.T)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Push Projection of frame
    glPushMatrix()
    
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.0,0.0,1.0,1.0])
    # GL_FRONT_AND_BACK --> Specifies both front and back faces are updated
    
    if ids is not None:    
        for i in range(len(ids)):
            
            tvec = tvecs[i]
            rvec = rvecs[i]
            
            # fix axis
            tvec[0,0] = tvec[0,0]
            tvec[0,1] = -tvec[0,1]
            tvec[0,2] = -tvec[0,2]
        
            rvec[0,1] = -rvec[0,1]
            rvec[0,2] = rvec[0,2]
            
            try:
                m = compositeArray(cv2.Rodrigues(rvec)[0], tvec[0]) # Rodrigues converts rotation matrix to rotation vector.
                
                glPushMatrix()
                glLoadMatrixd(m.T)
                glRotatef(180,0,0,1)
                
                glTranslatef(0,0,0)

                # Draws the car
                drawCar()
                
                glPopMatrix()
                
            except Exception as e:
                print(e)
    
    # Filter seen_ids: If time unseen is greater than time for 100 frames, remove it.
    seen_ids = {key:value for (key,value) in seen_ids.items() if value['t'] < 100}

    # Find the Aruco markers undetected by Aruco dictionary and estimate their position ourselves.
    undetected = (seen_ids[m_id] for m_id in seen_ids if not seen_ids[m_id]['seen'])
    
    for marker_details in undetected:
    	# Estimate its pose from previous or calculated vertices
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers([np.array([marker_details['vertices']])], 8.0, mtx, dist)
        rvec = rvec[0]
        tvec = tvec[0]
        
        # Fix axis
        tvec[0,0] = tvec[0,0]
        tvec[0,1] = -tvec[0,1]
        tvec[0,2] = -tvec[0,2]
       
        rvec[0,1] = -rvec[0,1]
        rvec[0,2] = rvec[0,2]
        
        try:
            m = compositeArray(cv2.Rodrigues(rvec)[0], tvec[0]) # Rodrigues converts rotation matrix to rotation vector.
            
            glPushMatrix()
            glLoadMatrixd(m.T)
            glRotatef(180,0,0,1)
            
            glTranslatef(0,0,0)

            # Draws the Car
            drawCar()
            
            glPopMatrix()
            
        except Exception as e:
            print(e)
            
    for m_id in seen_ids:
        marker_details = seen_ids[m_id]
        
        if not marker_details['seen']:
        	# Increase time a marker is unseen for each frame it is unseen
            marker_details['t'] += 1    

            # Estimate the position of vertices using its previous vertices and average velocity                
            marker_details['vertices'] = np.float32([marker_details['vertices'] + marker_details['av_velocity']]).reshape(-1,2)
            
            # Velocities are saved only if they are detected
            marker_details['saved_velocities'] = np.zeros((0,4,2), dtype=np.float32)
        
        # Mark all markers undetected by default. This will be updated if it is detected.
        marker_details['seen'] = False
    
    glPopMatrix()

    # To update the screen
    glFlush()
    glutSwapBuffers()
    
"""
Function name: compositeArray
Output: Returns 4x4 array to draw the 3d image on the background
Input: Rotation vector rvec and Translation vector tvec
Logic: Appends transpose of tvec to column and [0,0,0,1] as last row
Example call: matrix = compositeArray(rvec, tvec)
"""

def compositeArray(rvec, tvec):
    v = np.c_[rvec, tvec.T]
    v_ = np.r_[v, np.array([[0,0,0,1]])]
    return v_

"""
Function name: idle
Output: Redisplays the last image when the screen is idle or ended
Logic: Calls glutPostRedisplay()
Example call: idle() (In OpenGL main loop it is called as glutIdleFunc(idle))
"""

def idle():
    glutPostRedisplay()

"""
Function name: reshape
Output: Changes the window size of the output window
Input: width and height
Example call: reshape(width, height). In OpenGL main loop it is called as glutReshapeFunc(reshape)
"""
   
def reshape(w, h):
    glViewport(0, 0, w, h)
    glLoadIdentity()
    glOrtho(-w / windowWidth, w / windowWidth, -h / windowHeight, h / windowHeight, -1.0, 1.0)

"""
Function name: keyboard
Output: Respnds to keyboard inputs. If q is pressed, window is exited
Input: key pressed, x and y
Logic: Uses sys.exit() to exit
Example call: keyboard(key,x,y). In OpenGL main loop it is called as glutKeyboardFunc(keyboard))
"""
    
def keyboard(key, x, y):

    # convert byte to str
    key = key.decode('utf-8')
    if key == 'q':
        cap.release()
        sys.exit("q pressed. Exiting")

"""
Function name: main
Logic: Initializes GL window and GL
Example call: main()
"""
    
def main():

	# Initialize OpenGL window
    glutInitWindowPosition(0,0)
    glutInitWindowSize(windowWidth, windowHeight)
    glutInit(sys.argv)    
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)    
    glutCreateWindow("Aruco Tracker")

    # Displays the image and car in draw. It is called till the program ends    
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)

    # To quit the window when 'q' is pressed
    glutKeyboardFunc(keyboard)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # Enable lighting for 2d background projection
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Idle function is called when the screen is idle ie. when video is over and window is open    
    glutIdleFunc(idle)
    
    print("Recording started")

    # This is the main loop for OpenGL
    glutMainLoop()

    sys.exit("Recording ended")
    
try:
	# Start capturing video
    cap = cv2.VideoCapture(0) 

    # Initialize seen_ids with empty dictionary 
    seen_ids = {}

    # Call main
    main()
    
except Exception as e:
    print(e)
    print(traceback.format_exc())
    
finally:
    cap.release()