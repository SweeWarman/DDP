from matplotlib import pyplot as plt
import numpy as np
import DDPCore as DDP

def Model(X,U,dt=0.1):
    x = X[0,0] + U[0,0]*np.cos(X[2,0])*dt
    y = X[1,0] + U[0,0]*np.sin(X[2,0])*dt
    psi = X[2,0] + U[1,0]*dt
    return np.array([[x],[y],[psi]])

def GenerateInitialTrajectory(X0):
    traj = [X0] 
    control = []
    X = np.array(X0).reshape(3,1)
    U = np.zeros((2,1))
    U[0,0] = 1
    while (X[0]**2 + X[1]**2) > 1e-3:
        if X[0] > 3:
            U[1,0] = -0.1*(X[2] - np.pi)
        else:
            U[1,0] = -0.1*(X[2] - 3*np.pi/2) 
        Xnew = Model(X,U)
        traj.append(np.squeeze(Xnew,-1))
        control.append(np.squeeze(U,-1))
        X = Xnew
        if X[1] < 0:
            break
    return np.array(traj),np.array(control)
    

variations = {}
variations['L'] = lambda X,U: (X[0]**2 + X[1]**2 + U[0]**2 + 1.0e2* U[1]**2)*0.5 
variations['Lx'] = lambda X,U: np.array([[X[0]],[X[1]],[0.0]])
variations['Lxx'] = lambda X,U: np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
variations['Lu'] = lambda X,U: np.array([[U[0]],[1.0e2*U[1]]])
variations['Luu'] = lambda X,U: np.array([[1.0, 0.0],[0.0, 1.0e2]])
variations['Lux'] = lambda X,U: np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]])
variations['Fx'] = lambda X,U:np.array([[1.0,0.0,-np.sin(X[2])*0.1*U[0]],
                                        [0.0,1.0, np.cos(X[2])*0.1*U[0]],
                                        [0.0,0.0,                  1.0]])
variations['Fu'] = lambda X,U:np.array([[np.cos(X[2])*0.1,0.0],[np.sin(X[2])*0.1,0.0],[0.0,0.1]])
variations['Fxx'] = lambda X,U: [np.array([[0.0,0.0,0.0],
                                           [0.0,0.0,0.0],
                                           [0.0,0.0,-0.1*U[0]*np.cos(X[2])]]),
                                np.array([[0.0,0.0,0.0],
                                           [0.0,0.0,0.0],
                                           [0.0,0.0,-0.1*U[0]*np.sin(X[2])]]),
                                np.zeros((3,3))] 
variations['Fuu'] = lambda X,U: [np.array([[0.0,0.0],[0.0, 0.0]]),
                                 np.array([[0.0,0.0],[0.0, 0.0]]),
                                 np.array([[0.0,0.0],[0.0, 0.0]])]

trajectory0,control0 = GenerateInitialTrajectory([10.0,10.0,np.pi])
traj,ctrl,Vs,DeltaV = DDP.Solve(Model,variations,trajectory0,control0,500)
N= len(traj)
plt.figure(1)
for i in range(N):
    plt.plot(traj[i][:,0],traj[i][:,1])

plt.figure(2)
for i in range(N):
    plt.plot(np.arange(ctrl[i].shape[0]),ctrl[i])

plt.figure(3)
for i in range(N-1):
    plt.plot(np.arange(len(Vs[i])),Vs[i])

plt.figure(4)
plt.plot(np.arange(len(DeltaV[0])),DeltaV[0],'o-')
plt.show()