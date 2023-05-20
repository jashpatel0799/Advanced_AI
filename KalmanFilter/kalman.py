from env import *
import numpy as np
class KalmanFilter:
    def __init__(self, noise_velocity, noise_position) -> None:
        # Complete this function to construct

        # Assume that nothing is known 
        # about the state of the target at this instance
        

        self.position_noise = noise_position
        self.velocity_noise = noise_velocity

        self.x_est = np.array([108, 108, 18, 18, 1, 1])

        self.x_est = self.x_est.reshape(6,1)

        self.dt = 1 # delta t
        self.Q = self.velocity_noise * np.eye(6)
        self.R = self.position_noise * np.eye(3)
        # self.Q = np.eye(6)
        # self.R = np.eye(3)
        self.F = np.array([[1, 0, 0, 0, 0, 0],
                           [self.dt, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, self.dt, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, self.dt, 1]])
        
        self.P = (self.velocity_noise)**2 * np.eye(6) 
        # self.P = np.eye(6) 


        # self.G = np.matrix([[self.dt],
        #                     [-((self.dt)**2)/2],
        #                     [self.dt],
        #                     [-((self.dt)**2)/2],
        #                     [self.dt],
        #                     [-((self.dt)**2)/2]])
        self.G = np.matrix([[self.dt, 0, 0],
                            [-((self.dt)**2)/2, 0, 0],
                            [0, self.dt, 0],
                            [0,-((self.dt)**2)/2, 0],
                            [0, 0, self.dt],
                            [0, 0, -((self.dt)**2)/2]])
        
        # self.G = np.matrix([[0],
        #                     [1],
        #                     [0],
        #                     [1],
        #                     [0],
        #                     [1]])
        # print(self.G.shape)
        

        # self.m = int(5.0/self.dt)

        self.alpha = (self.x_est[0]**2 + self.x_est[2]**2 + self.x_est[4]**2)**(3/2)
        # print(self.alpha)
        self.beta = (self.x_est[0]**2 + self.x_est[4]**2)**(3/2)


        self.H = -1/1000 * np.array([[0, self.x_est[0]/self.alpha, 0, self.x_est[2]/self.alpha, 0, self.x_est[4]/self.alpha],
                                   [0, self.x_est[2]/self.beta, 0, self.x_est[4]/self.beta, 0, -1/self.x_est[0]],
                                   [0, self.x_est[4]/self.x_est[0]**2, 0, -1/self.x_est[0], 0, -3]], dtype=float)
        
        # self.H = np.array([[0, 1, 0, 0, 0, 0],
        #                    [0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 0, 1]])
        # print(self.H)
        # pass

    def input(self, observed_state:State, accel:numpy.ndarray, justUpdated:bool):

        # This function is executed multiple times during the reading.
        # When an observation is read, the `justUpdated` is true, otherwise it is false
        
        # accel is the acceleration(control) vector. 
        # It is dynamically obtained regardless of the state of the RADAR 
        # (i.e regardless of `justUpdated`) 

        # When `justUpdated` is false, the state is the same as the previously provided state
        # (i.e, given `observed_state` is not updated, it's the same as the previous outdated one)


        # Complete this function where current estimate of target is updated
        accel = accel.reshape(3,1)

        if justUpdated:
              

              ph = np.matmul(self.P, self.H.T)

              hpht = np.matmul(np.matmul(self.H, self.P),self.H.T)

              hr = hpht + self.R

              inv_hr = np.linalg.pinv(hr)
              Kalman_gain = np.matmul(ph, inv_hr)

              self.z = np.array([observed_state.position[0],observed_state.position[1],observed_state.position[2]])
              self.z = self.z.reshape(3,1)


              self.y_est = self.z - np.matmul(self.H, self.x_est) + accel

              self.x_est = self.x_est + np.matmul(Kalman_gain, self.y_est)
            #   print(accel.shape)
              #incorporate new measurements
              
                                #  observed_state.velocity[0],observed_state.velocity[1],observed_state.velocity[2]])
            #   self.z = self.z.reshape(6,1)
              
              #self.y = self.y - np.dot(self.H, self.x_estimate)
            #   self.y_measurement = self.z - self.y_est

              #print(self.y_measurement)
        else: 
              #propagate x and P
            #   self.P = self.P + np.linalg.inv(np.dot(np.dot(self.H.T, np.linalg.inv(self.R)), self.H))
              p_new = np.matmul(np.matmul(self.F, self.P), self.F.T)
              self.P = p_new + self.Q   #P = F.P.Ft + Q
              
              self.x_est = np.matmul(self.F, self.x_est) +  np.matmul(self.G, accel)#Xk = F.x + B.u 
              #print(self.x_estimate.shape)


        # pass

    def get_current_estimate(self)->State:
        
        # Complete this function where the current state of the target is returned
        #estimate state x based on previous k measurements

        #self.X = self.F.dot(self.X) + self.B.dot(self.u)
        #self.P = self.F.dot(self.P)
        #self.P = self.P.dot(self.F.T) + self.Q
        #return State(self.X[0], self.X[1])
        # self.x_est = np.dot(self.F, self.x_est) + np.dot(self.G, self.u) 
        # self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        #print(self.x_estimate.shape)

        #x_position = np.array([self.x_estimate[0], self.x_estimate[1], self.x_estimate[2]])
        #x_velocity = np.array([self.x_estimate[3], self.x_estimate[4], self.x_estimate[5]])

        x_pos= Vector3D(self.x_est[1], self.x_est[3], self.x_est[5])
        x_velo = Vector3D(self.x_est[0], self.x_est[2], self.x_est[4])

        state_estimate =  State(x_pos.reshape(3), x_velo.reshape(3))
        #print(state_estimate.position.shape)
        #print(state_estimate.velocity.shape)

        return state_estimate
        # pass


# from env import *
# import numpy as np
# class KalmanFilter:
#     def __init__(self, noise_velocity, noise_position) -> None:

#        self.noise_position = noise_position
#        self.noise_velocity = noise_velocity

#        self.curstate = State(RandomVector3D(), RandomVector3D())
       
#        self.x_estimate = np.array(
#               [self.curstate.position[0],
#               self.curstate.position[1],
#               self.curstate.position[2], 
#               self.curstate.velocity[0],
#               self.curstate.velocity[1],
#               self.curstate.velocity[2]])

#        self.x_estimate = self.x_estimate.reshape(6,1)

#        self.dt = 1
#        #self.Q = self.noise_velocity * np.eye(6)
#        #self.R = self.noise_position * np.eye(6)
#        self.Q = np.eye(6)
#     #    self.R = np.eye(6)
#        self.R = np.eye(3)
#        self.F = np.eye(6)
#        self.P = np.eye(6) #the covariance of the estimation error
       
       

#        sj = 0.1
#        #self.Q = np.matrix([[(self.dt**6)/36, 0, 0, (self.dt**5)/12, 0, 0, (self.dt**4)/6, 0, 0],
#               #[0, (self.dt**6)/36, 0, 0, (self.dt**5)/12, 0, 0, (self.dt**4)/6, 0],
#               #[0, 0, (self.dt**6)/36, 0, 0, (self.dt**5)/12, 0, 0, (self.dt**4)/6],
#               #[(self.dt**5)/12, 0, 0, (self.dt**4)/4, 0, 0, (self.dt**3)/2, 0, 0],
#               #[0, (self.dt**5)/12, 0, 0, (self.dt**4)/4, 0, 0, (self.dt**3)/2, 0],
#               #[0, 0, (self.dt**5)/12, 0, 0, (self.dt**4)/4, 0, 0, (self.dt**3)/2],
#               #[(self.dt**4)/6, 0, 0, (self.dt**3)/2, 0, 0, (self.dt**2), 0, 0],
#               #[0, (self.dt**4)/6, 0, 0, (self.dt**3)/2, 0, 0, (self.dt**2), 0],
#               #[0, 0, (self.dt**4)/6, 0, 0, (self.dt**3)/2, 0, 0, (self.dt**2)]]) *sj**2
       
#        self.B = np.matrix([[0.0],
#                [0.0],
#                [0.0],
#                [0.0],
#                [0.0],
#                [0.0]])
#        self.u = 0.1
#        self.m = int(5.0/self.dt)

#     #    self.H = np.eye(6)
#     #    self.H = -1/1000 * np.array([[1, 0, 0, 0, 0, 0],
#     #                               [0, 1, 0, 0, 0, 0],
#     #                               [0, 0, 1, 0, 0, 0]])

#        self.H = np.array([[0, 1, 0, 0, 0, 0],
#                           [0, 0, 0, 1, 0, 0],
#                           [0, 0, 0, 0, 0, 1]])

#     def input(self, observed_state:State, accel:numpy.ndarray, justUpdated:bool):
       
#        #self.u = accel
#        #self.u = self.u.reshape(1, 3)
#     #    print(accel)

#        #print(observed_state.position.shape)
#        #print(observed_state.velocity.shape)

#        if justUpdated:
#               accel = accel.reshape(3,1)
#               self.y_estimate = np.dot(self.H, self.x_estimate) + accel
#               #incorporate new measurements
#               self.z = np.array([observed_state.position[0],observed_state.position[1],observed_state.position[2]]) 
#                             #    observed_state.velocity[0],observed_state.velocity[1],observed_state.velocity[2]])

#             #   self.z = self.z.reshape(6,1)
#               self.z = self.z.reshape(3,1)
#               #self.y = self.y - np.dot(self.H, self.x_estimate)
#               self.y_measurement = self.z - self.y_estimate

#               #print(self.y_measurement)
#        else: 
#               #propagate x and P
#             #   self.P = self.P + np.linalg.inv(np.dot(np.dot(self.H.T, np.linalg.inv(self.R)), self.H ))
#               self.P = self.P + (np.dot(np.dot(self.H.T, self.R), self.H ))
#               self.x_estimate = self.x_estimate + np.dot(np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.R)), self.y_measurement)
#               #print(self.x_estimate.shape)

       

#     def get_current_estimate(self)->State: # called just before new measurements come in

#        #estimate state x based on previous k measurements

#        #self.X = self.F.dot(self.X) + self.B.dot(self.u)
#        #self.P = self.F.dot(self.P)
#        #self.P = self.P.dot(self.F.T) + self.Q
#        #return State(self.X[0], self.X[1])
#        self.x_estimate = np.dot(self.F, self.x_estimate) + np.dot(self.B, self.u) #Xk = F.x + B.u
#        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q #P = F.P.Ft + Q

#        #print(self.x_estimate.shape)

#        #x_position = np.array([self.x_estimate[0], self.x_estimate[1], self.x_estimate[2]])
#        #x_velocity = np.array([self.x_estimate[3], self.x_estimate[4], self.x_estimate[5]])

#        x_position = Vector3D(self.x_estimate[0], self.x_estimate[1], self.x_estimate[2])
#        x_velocity = Vector3D(self.x_estimate[3], self.x_estimate[4], self.x_estimate[5])

#        state_estimate =  State(x_position.reshape(3), x_velocity.reshape(3))
#        #print(state_estimate.position.shape)
#        #print(state_estimate.velocity.shape)

#        return state_estimate

# obj = KalmanFilter(0.1, 0.1)