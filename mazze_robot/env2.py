import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from vrep import vrep 


class RLCar(object):

    def __init__(self):
        # just in case, close all opened connections
        vrep.simxFinish(-1)

        self.client_id = vrep.simxStart(
            '127.0.0.1', 19997, True, True, 5000, 5)

        if self.client_id != -1:  # check if client connection successful
            print('Connected to remote API server')
        else:
            print('Connection not successful')
            sys.exit('Could not connect')

        # Restart the simulation
        self.reset()

        # Get handles
        self.get_handles()

    def get_handles(self):
        # retrieve motor  handles
        _, self.left_motor_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_blocking)
        _, self.right_motor_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_blocking)
        
      

        # empty list for handles
        self.proxy_sensors = []

        # for loop to retrieve proxy sensor arrays and initiate sensors
        for i in range(3):
            _, sensor_handle = vrep.simxGetObjectHandle(
                self.client_id, 'Pioneer_p3dx_ultrasonicSensor' + str(i),
                vrep.simx_opmode_blocking)
            # Append to the list of sensors
            self.proxy_sensors.append(sensor_handle)

        # empty list for handles
        self.light_sensors = []

        # for loop to retrieve light sensor arrays and initiate sensors
        for i in range(1):
            _, sensor_handle = vrep.simxGetObjectHandle(
                self.client_id, 'light_sensor#' + str(i),
                vrep.simx_opmode_blocking)
            # Append to the list of sensors
            self.light_sensors.append(sensor_handle)

    def destroy(self):
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        
        
    def readSensor(self):
        # Get observations
        observations = {}
        observations['proxy_sensor'] = []
        observations['light_sensor'] = []

        # Fetch the vals of proxy sensors
        for sensor in self.proxy_sensors:
            _, _, detectedPoint, _, _ = vrep.simxReadProximitySensor(
                self.client_id, sensor, vrep.simx_opmode_blocking)
            # Append to list of values
            # print (detectedPoint)
            observations['proxy_sensor'].append(
                np.linalg.norm(detectedPoint))

        # Fetch the vals of light sensors
        for sensor in self.light_sensors:
            # Fetch the initial value in the suggested mode
            _, _, image = vrep.simxGetVisionSensorImage(
                self.client_id, sensor, 1, vrep.simx_opmode_blocking)
            # extract image from list
            image = image[0] if len(image) else -1
            # print(image)
            # Append to the list of values
            observations['light_sensor'].append(image)
            
            return observations

    def reset(self):
        # Restart the simulation
        stop = vrep.simxStopSimulation(
            self.client_id, vrep.simx_opmode_blocking)
        time.sleep(.5)
        
        start = vrep.simxStartSimulation(
            self.client_id, vrep.simx_opmode_blocking)
        print("Resetting Simulation. Stop Code: {} Start Code: {}".format(stop, start))
        
        
         # retrieve motor  handles
        _, self.left_motor_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_streaming)
        _, self.right_motor_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_streaming)
        
        
        # vrep.simxSetJointTargetVelocity(
        #     self.client_id, self.left_motor_handle, 0, vrep.simx_opmode_blocking)
        # vrep.simxSetJointTargetVelocity(
        #     self.client_id, self.right_motor_handle, 0, vrep.simx_opmode_blocking)
        

    def step(self, action):
        # Activate the motors
        vrep.simxSetJointTargetVelocity(
            self.client_id, self.left_motor_handle, action[0], vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(
            self.client_id, self.right_motor_handle, action[1], vrep.simx_opmode_streaming)

        # Get observations
        observations = {}
        observations['proxy_sensor'] = []
        observations['light_sensor'] = []

        # Fetch the vals of proxy sensors
        for sensor in self.proxy_sensors:
            _, _, detectedPoint, _, _ = vrep.simxReadProximitySensor(
                self.client_id, sensor, vrep.simx_opmode_streaming)
            # Append to list of values
            # print (detectedPoint)
            observations['proxy_sensor'].append(
                np.linalg.norm(detectedPoint))

        # Fetch the vals of light sensors
        for sensor in self.light_sensors:
            # Fetch the initial value in the suggested mode
            _, _, image = vrep.simxGetVisionSensorImage(
                self.client_id, sensor, 1, vrep.simx_opmode_streaming)
            # extract image from list
            image = image[0] if len(image) else -1
            # print(image)
            # Append to the list of values
            observations['light_sensor'].append(image)

        # vrep gives a positive value for the black strip and negative for the
        # floor so convert it into 0 and 1

        observations['light_sensor'] = np.asarray(observations['light_sensor'])
        observations['light_sensor'] = np.sign(observations['light_sensor'])

        # When nothing is detected a very small value is retured -> changing it to 0.1
        observations['proxy_sensor'] = np.asarray(observations['proxy_sensor'])
        observations['proxy_sensor'][observations['proxy_sensor'] < 0.1] = 0.0

        # Assign reward
        reward = {}

        # For light sensors
        # If any of the center 2 sensors is 1 give high reward
        if (observations['light_sensor'] > 0).any():
            reward['light_sensor'] = 500
        else:
            reward['light_sensor'] = 0
        
        # For proximity sensors
        reward['proxy_sensor'] =  (observations['proxy_sensor'] < 0.7).sum() * -10
        
        # if (observations['proxy_sensor'][1]) < 0.3:
        #     reward['proxy_sensor'] = -3
        
        # reward['proxy_sensor'] += (observations['proxy_sensor'] < 0.3).sum() * -20
        # reward['proxy_sensor'] += (observations['proxy_sensor'] > 1).sum() * 3
        # reward['proxy_sensor'] += (observations['proxy_sensor'] > 2).sum() * 6
        # reward['proxy_sensor'] += (observations['proxy_sensor'] > 3).sum() * 10

        # Should be rewarded for movement
        r = (action[0]*action[1]) * 0
        
        r = int ((observations['proxy_sensor'] > 0.3).sum()/3)*10

        reward['light_sensor'] += r
        reward['proxy_sensor'] += r

        return observations, reward