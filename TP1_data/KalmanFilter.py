import numpy as np

class KalmanFilter():
    def __init__(self, dt : float, u_x : float, u_y : float, std_acc : float, x_sdt_meas : float, y_sdt_meas : float):
        self.u = np.array([[u_x], [u_y]])

        self.A = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1,0],
                  [0,0,0,1]])

        var_dt = 0.5 * (dt**2)
        self.B = np.array([[var_dt, 0],
                  [0, var_dt],
                  [dt, 0],
                  [0, dt]])

        self.H = np.array([[1,0,0,0],
                  [0,1,0,0]])

        self.xk = np.zeros((4,1))

        sig_a = std_acc**2
        self.Q = np.array([[dt ** 4/4, 0, dt**3/2,0],
                  [0,dt**4/4,0,dt**3/2],
                  [dt**3/2,0,dt**2,0],
                  [0,dt**3/2,0,dt**2]]) * sig_a

        self.R = np.array([[x_sdt_meas**2,0],
                          [0, y_sdt_meas**2]])

        self.P = np.identity(self.A.shape[0])

    def predict(self):
        self.xk = np.dot(self.A, self.xk) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.xk

    def update(self, zk : np.ndarray):
        Sk = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        Kk = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(Sk))

        # Update state estimate
        self.xk = self.xk + np.dot(Kk, (zk - np.dot(self.H, self.xk)))
        self.P = self.P - np.dot(np.dot(Kk, self.H), self.P)

        return self.xk
        