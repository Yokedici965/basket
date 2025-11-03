import cv2, numpy as np

def point_in_poly(pt, poly):
    cnt = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0


def polyline(points):
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)
