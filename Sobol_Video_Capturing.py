import cv2
import numpy

if __name__ == '__main__':
    cv2.namedWindow("original")

    cap = cv2.VideoCapture(0)
    kernel = numpy.array([[-1, 0, 1],
                          [-3, 0, 3],
                          [-1, 0, 1]])
    kernel2 = numpy.array([[-2, 0, 2],
                          [-3, 0, 3],
                          [-2, 0, 2]])
    while True:
        flag, image = cap.read()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            dst = cv2.filter2D(img, -1, kernel)
            dst2 = cv2.filter2D(img, -1, kernel2)
            cv2.imshow('original', dst)
            cv2.imshow('kernel2', dst2)

        except:
            cap.release()
            raise

        ch = cv2.waitKey(5)

        if ch == 27:
            break
    cap.release()
    cv2.destroyAllWindows()