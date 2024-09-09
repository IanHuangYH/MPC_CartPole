import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

folder_path = "data/NMPC/"
NegFilename = "ini_states_1ini_guess_neg.txt"
PosFilename = "ini_states_1ini_guess_pos.txt"
neg_finalfilename = os.path.join(folder_path, NegFilename)
pos_finalfilename = os.path.join(folder_path, PosFilename)

data_ini_guess_pos = np.loadtxt(pos_finalfilename)
data_ini_guess_neg = np.loadtxt(neg_finalfilename)

X_DIM = 5
time = data_ini_guess_pos[:,0]
x_pos = data_ini_guess_pos[:,1:X_DIM+1]
x_neg = data_ini_guess_neg[:,1:X_DIM+1]
u_pos = data_ini_guess_pos[:,6]
u_neg = data_ini_guess_neg[:,6]
MPCcost_pos = data_ini_guess_pos[:,7]
MPCcost_neg = data_ini_guess_neg[:,7]

theta_pos = x_pos[:,2]
theta_neg = x_neg[:,2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(u_pos, theta_pos, MPCcost_pos, c='r', marker='o')
ax.scatter(u_neg, theta_neg, MPCcost_neg, c='b', marker='o')

# Set labels
ax.set_xlabel('u')
ax.set_ylabel('theta')
ax.set_zlabel('Cost')

# Create a 2D plot
fig = plt.figure()
plt.plot(theta_pos[1:], MPCcost_pos[1:], 'ro-', label='pos ini')
plt.plot(theta_neg[1:], MPCcost_neg[1:], 'bo-', label='neg ini')

# Set labels
plt.xlabel('theta')
plt.ylabel('Cost')

# Show the plot
plt.show()


