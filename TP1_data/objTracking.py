from KalmanFilter import KalmanFilter
import numpy as np
from Detector import detect
import cv2

def main():
    dt = 0.1
    u_x = 1
    u_y = 1
    std_acc = 1
    x_sdt_meas = 0.1
    y_sdt_meas = 0.1

    video_path = './randomball.mp4'
    cap = cv2.VideoCapture(video_path)

    # initialize the Kalman filter
    kalman = KalmanFilter(dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas)

    # for each frame, detect the object and predict its position
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect the object
        centers = detect(frame)

        if len(centers) > 0:

            # predict the position of the object
            x_pred = kalman.predict()
            # update the position of the object
            x_upd = kalman.update(centers[0])

            centers = np.array(centers).round().astype("int")
            real_center = [centers[0][0][0], centers[0][1][0]]

            # Draw detected circle (green color)
            cv2.circle(frame, real_center, 20, (0, 255, 0), 2)

            # Draw a blue rectangle as the predicted object position
            #print(x_pred)
            #print(x_pred[0])
            x_pred = x_pred.round().astype("int")
            cv2.rectangle(frame, (x_pred[0][0] - 20, x_pred[1][0] - 20), (x_pred[0][0] + 20, x_pred[1][0] + 20), (255, 0, 0), 2)

            # Draw a red rectangle as the estimated object position
            x_upd = x_upd.round().astype("int")
            cv2.rectangle(frame, (x_upd[0][0] - 20, x_upd[1][0] - 20), (x_upd[0][0] + 20, x_upd[1][0] + 20), (0, 0, 255), 2)

            # Draw the trajectory (tracking path ) in an image
            cv2.line(frame, (x_upd[0][0], x_upd[1][0]), (x_pred[0][0], x_pred[1][0]), (0, 255, 255), 2)

        cv2.imshow('frame', frame)

        #wait 0.1 second
        cv2.waitKey(100)


if __name__ == "__main__":
    main()
