import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img


def chess_board_grid_detector(img, size):
    found, corners = cv2.findChessboardCorners(img, patternSize=size)
    if found:
        cv2.drawChessboardCorners(img, size, corners, found)
        return img
    else:
        print('Grid Not Found!')
        return


def dot_grid_detector(img, size):
    found, corners = cv2.findCirclesGrid(img, size, cv2.CALIB_CB_SYMMETRIC_GRID)
    if found:
        cv2.drawChessboardCorners(img, size, corners, found)
        return img
    else:
        print('Grid Not Found!')
        return


chess_board = cv2.imread('DATA/flat_chessboard.png')
dot_grid = cv2.imread('DATA/dot_grid.png')
chess_board_grid_detector(chess_board, (7, 7))
dot_grid_detector(dot_grid, (10, 10))
# show_img('Chess', chess_board)
# show_img('Dot Grid', dot_grid)
