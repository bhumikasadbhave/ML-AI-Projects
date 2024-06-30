Kalman and Particle Filter Implementation
----------------------------------------------------------
(Jupyter Notebook is to be run sequentially)


Jupyter Notebook Structure:
----------------------------
-> Imports for the notebook
-> Kalman Filter Implementation
   - True Trajectory
   - Simulate Observations
   - Notations
   - Kalman Filter Class
   - Setting Initial Parameters
   - Running the filter
   - Result Plot
 -> Particle Filter Implementation
   - True Trajectory
   - Simulate Observations
   - Notations
   - Kalman Filter Class
   - Setting Initial Parameters
   - Running the filter
   - Result Plot  
   

Short Overview of the Implementations:

1. Kalman Filter Implementation: State is [x,y,vx,vy]
--------------------------------

Goal: Predict trajectory of a ball when it is thrown at an angle. Estimate the position and velocity vector of the ball only from the observed erroneous positions over time.


-> Simulate Trajectory Function: In this function we simulate the trajectory of the ball thrown by using the velocity, angle and initial position of the ball. The formulas for physical movement given in the class lecture are used, taking the angle into account as well.
  
-> Simulate Observations: Since we don't have any sensor recorded observations, we simulate the observations. We take the true positions/trajectory of the ball and add Gaussian noise to the x and y values. 
  
-> Notations used in the code: This contains the notations used in the implementation of Kalman Filter Algorithm.
  
-> Kalman Filter Algorithm: Here, the the Kalman Filter is implemented as a Python class. We have the below functions:
         a) init function: Initialize all the parameters. The parameters are initialized as given in the lecture slides.
         b) Predict step: Predict the next state and update the covariance Matrix.
         c) Update step: Calculate the Kalman gain based on observations, and updates the state as well as covariance matrix.
         d) Apply filter function: Runs the filter in a loop for all the observations
  
-> Running the Algorithm: Here we define all the parameters, we can change the parameters we want to set from here, or directly in the init function. We call all the functions.
  
-> Plot of the Results: Plot of the ball tractectory true positions and Kalman Filter estimates.
  

2. Particle Filter Implementation: Here we have to balls, so state is [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
----------------------------------

Goal: Predict trajectory of 2 balls when they are thrown simultaneously at an angle. Estimate the position and velocity vector of the balls only from the observed erroneous positions over time.


-> Simulate Trajectory Function: In this function we simulate the trajectory of the balls thrown by using the velocity, angle and initial positions of both the balls. 
  
-> Simulate Observations: Since we don't have any sensor recorded observations, we simulate the observations. We take the true positions/trajectory of the ball and add Gaussian noise to the x1,x2,y1 and y2 values. 
  
-> Notations used in the code: This contains the notations used in the implementation of Kalman Filter Algorithm.

-> Particle Filter Algorithm: Here, the the Particle Filter is implemented as a Python class. We have the below functions:
        a) init function: Initialize all the parameters. The parameters are initialized as given in the lecture slides.
        b) gaussian_formula function: Applies Gaussian formula on the parameter passed.
        c) estimate function: Comuputes the weighted average over all particles.
        d) Predict Step: Propagate through transition model: Apply physical movement and update particles  
        e) Update Step: Evaluate particles with observation model: Calculate weights and resample 
        f) Resample Step: Do sampling from multinomial distribution based on weights so that particles with higher weights have a higher probability to get sampled.     
        
-> Running the Algorithm: Here we define all the parameters, we can change the parameters we want to set from here, or directly in the init function. We call all the functions.
  
-> Plot of the Results: Plot of the ball tractectory true positions and Kalman Filter estimates.        
        
  
  