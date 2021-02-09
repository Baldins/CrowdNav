import numpy as numpy

def homography_eth(x, y):

  return x, y

# 2.8128700e-02   2.0091900e-03  -4.6693600e+00
# 8.0625700e-04   2.5195500e-02  -5.0608800e+00
# 3.4555400e-04   9.2512200e-05   4.6255300e-01

pts = np.array([[1,2],[3,4]], np.float32)
M = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
              [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
              [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]], \
              dtype=np.float32)

## (n, 1, 2)
pts1 = pts.reshape(-1,1,2).astype(np.float32)
dst1 = cv2.perspectiveTransform(pts1, M)

## (1, n, 2)
pts2 = np.array([pts], np.float32)
dst2 = cv2.perspectiveTransform(pts2, M)