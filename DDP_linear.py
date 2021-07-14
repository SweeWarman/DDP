
import numpy as np
import DDPCore as DDP

def Model(X,U,dt=0.1):
    Xnew = np.dot(np.eye(2),X) + np.dot(dt*np.eye(2),U)
    return Xnew

def GenerateInitialTrajectory(X0):
    traj = [X0] 
    control = []
    X = np.array(X0).reshape(2,1)
    while (X[0]**2 + X[1]**2) > 1e-3:
        if X[0] > 3:
            U = np.array([-1.0,0.0])
        else:
            U = np.array([0.0,-1.0])
        Xnew = Model(X,U.reshape(2,1))
        traj.append(np.squeeze(Xnew,-1))
        control.append(U)
        X = Xnew
        if X[1] < 0:
            break
    return np.array(traj),np.array(control)
    

variations = {}
variations['L'] = lambda X,U: (X[0]**2 + X[1]**2 + U[0]**2 + U[1]**2)*0.5 
variations['Lx'] = lambda X,U: np.array([[X[0]],[X[1]]])
variations['Lxx'] = lambda X,U: np.array([[1.0, 0.0],[0.0,1.0]])
variations['Lu'] = lambda X,U: np.array([[U[0]],[U[1]]])
variations['Luu'] = lambda X,U: np.array([[1.0, 0.0],[0.0, 1.0]])
variations['Lux'] = lambda X,U: np.array([[0.0,0.0],[0.0,0.0]])
variations['Fx'] = lambda X,U:np.array([[1.0,0.0],
                                        [0.0,1.0]])
variations['Fu'] = lambda X,U:np.array([[1.0, 0.0],[0.0,1.0]])
variations['Fxx'] = lambda X,U: [np.zeros((2,2)), np.zeros((2,2))] 
variations['Fuu'] = lambda X,U: [np.zeros((2,2)), np.zeros((2,2))] 

trajectory0,control0 = GenerateInitialTrajectory([10.0,10.0])
traj,ctrl,Vs,DeltaV = DDP.Solve(Model,variations,trajectory0,control0,10)