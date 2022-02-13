import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img


def orb_feature_match(src, dest):
    orb = cv2.ORB_create()
    key_point1, description1 = orb.detectAndCompute(src, None)
    key_point2, description2 = orb.detectAndCompute(dest, None)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(description1, description2)
    matches = sorted(matches, key=lambda x: x.distance)
    recognized_match = cv2.drawMatches(src, key_point1, dest, key_point2, matches1to2=matches[:int(len(matches) / 2)],
                                       outImg=None, flags=2)
    return recognized_match


def get_best_matches(matches, ratio):
    best_matches = []
    for match1, match2 in matches:
        # the lower the distance, the better the match
        if match1.distance < ratio * match2.distance:
            best_matches.append([match1])
    return best_matches


def sift_feature_match(src, dest):
    sift = cv2.xfeatures2d.SIFT_create()
    key_point1, description1 = sift.detectAndCompute(src, None)
    key_point2, description2 = sift.detectAndCompute(dest, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(description1, description2, k=2)
    best_matches = get_best_matches(matches, 0.75)
    sift_matches = cv2.drawMatchesKnn(src, key_point1, dest, key_point2, matches1to2=best_matches, outImg=None, flags=2)
    return sift_matches


def flann_feature_match(src, dest):
    sift = cv2.xfeatures2d.SIFT_create()
    key_point1, description1 = sift.detectAndCompute(src, None)
    key_point2, description2 = sift.detectAndCompute(dest, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(description1, description2, k=2)
    best_matches = get_best_matches(matches, 0.75)
    draw_params = get_draw_params(best_matches)
    flann_matches = cv2.drawMatchesKnn(src, key_point1, dest, key_point2, matches1to2=best_matches, outImg=None,
                                       **draw_params)
    return flann_matches


def get_draw_params(matches):
    matches_mask = [[1, 0] for i in range(len(matches))]
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matches_mask, flags=0)
    return draw_params


target_cereal = cv2.imread('DATA/reeses_puffs.png')
cereals = cv2.imread('DATA/many_cereals.jpg')

# einstein = cv2.imread('DATA/einstein.jpg')
# conference = cv2.imread('DATA/solvay_conference.jpg')

matches_found = orb_feature_match(target_cereal, cereals)
matches_found_2 = sift_feature_match(target_cereal, cereals)
matches_found_3 = flann_feature_match(target_cereal, cereals)

# einstein_found = flann_feature_match(einstein, conference)

show_img('Cereals', matches_found_3)
