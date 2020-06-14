##---------- IMPORTS ----------##

from cv2 import cv2
import numpy as np
from scipy import ndimage
import math
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
# import SudokuSolver
import copy

##---------- Solving Sudoku ----------##

# Solving Sudoku using Best-First search
# Best first search algorithms is an optimized version of Backtracking,
# where the “next cell” is the cell which has the least number of possibilities
# The 'number of possibilities' is calculated for each cell, by going through its corresponding
# row, column and 3x3 block and counting the number of have-not-chosen numbers
# This greedy heuristic increase the efficiency of the program substantially, as it minimizes the branching factor.

# Keep the "Best" cell data
class EntryData:
    def __init__(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

    def set_data(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

# Solve Sudoku using Best-first search
def solve_sudoku(matrix):
    cont = [True]
    # See if it is even possible to have a solution
    for i in range(9):
        for j in range(9):
            if not can_be_correct(matrix, i, j): # If it's not possible, stop
                return
    sudoku_helper(matrix, cont) # Otherwise try to solve the Sudoku puzzle

# Helper function - The heart of Best First Search
def sudoku_helper(matrix, cont):
    if not cont[0]: # Stopping point 1
        return

    # Find the best entry (The one with the least possibilities)
    best_candidate = EntryData(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0: # If it is unfilled
                num_choices = count_choices(matrix, i, j)
                if best_candidate.choices > num_choices:
                    best_candidate.set_data(i, j, num_choices)

    # If didn't find any choices, it means...
    # It has filled all board, Best-First Search done! 
    # Note, whether we have a solution or not depends on whether all Board is non-zero
    if best_candidate.choices == 100: 
        # Set the flag so that the rest of the recursive calls can stop at "stopping points"
        cont[0] = False 
        return

    row = best_candidate.row
    col = best_candidate.col

    # If found the best candidate, try to fill 1-9
    for j in range(1, 10):
        if not cont[0]: # Stopping point 2
            return

        matrix[row][col] = j

        if can_be_correct(matrix, row, col):
            sudoku_helper(matrix, cont)

    if not cont[0]: # Stopping point 3
        return
    matrix[row][col] = 0 # Backtrack, mark the current cell empty again

# Count the number of choices haven't been used
def count_choices(matrix, i, j):
    # From 0 to 9 - don't consider 0
    can_pick = [True,True,True,True,True,True,True,True,True,True]
    
    # Check row
    for k in range(9):
        can_pick[matrix[i][k]] = False

    # Check col
    for k in range(9):
        can_pick[matrix[k][j]] = False

    # Check 3x3 square
    r = i // 3
    c = j // 3
    for row in range(r*3, r*3+3):
        for col in range(c*3, c*3+3):
            can_pick[matrix[row][col]] = False

    # Count
    count = 0
    for k in range(1, 10):  # 1 to 9
        if can_pick[k]:
            count += 1

    return count

# Return true if the current cell doesn't create any violation
def can_be_correct(matrix, row, col):
    
    # Check row
    for c in range(9):
        if matrix[row][col] != 0 and col != c and matrix[row][col] == matrix[row][c]:
            return False

    # Check column
    for r in range(9):
        if matrix[row][col] != 0 and row != r and matrix[row][col] == matrix[r][col]:
            return False

    # Check 3x3 square
    r = row // 3
    c = col // 3
    for i in range(r*3, r*3+3):
        for j in range(c*3, c*3+3):
            if row != i and col != j and matrix[i][j] != 0 and matrix[i][j] == matrix[row][col]:
                return False
    
    return True

# This function returns true if the entire board is occupied by some non-zero number
# If true, the current board is the solution to the original Sudoku
def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True

##---------- Image Processing ----------##

# To Write the solution on "image"
def write_soln_on_image(image, grid, user_grid):
    # Write grid on the image
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if(user_grid[i][j] != 0): # If already filled
                continue
            # Convert to string
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseline = cv2.getTextSize(text, font, fontScale=1, thickness=3)

            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
            # Line_AA minimizes distortion in the text drawn
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), font, font_scale, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    return image

# Compare all elements of the 2 matrices and return if all corresponding entries are equal
def two_matrices_are_equal(matrix_1, matrix_2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix_1[i][j] != matrix_2[i][j]:
                return False
    return True

# First criteria for detecting sudoku board contours,
# Length of sides CANNOT be too different (very less difference in size) [sudoku has a square board]
# Returns if longest side is greater then shortest side * eps_scale
def side_lengths_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    largest = max(AB, AD, BC, CD)
    return largest > eps_scale * shortest

# Second criteria for detecting sudoku board contours,
# All 4 angles have to be approximately 90 deg
# Approximately 90 deg with epsilon tolerance (unit roundoff)
def approx_90_deg(angle, epsilon):
    return abs(angle - 90) < epsilon

# Divide the sudoku board into 9x9 small square images
# Each image will be a "crop_image"
# Seperating digit from noise in "crop_image"
def largest_connected_component(image):
    image = image.astype('uint8')

    # outputs labelled image, statistics and centroid for each label 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    
    max_label = 1

    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    
    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2

# Find angle between two vectors (to check if corners are 90 deg )
def angle_between(vector_1, vector_2):
    # linalg is used for linear algebra
    # norm function returns the length of vector
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    # Converting to degree
    return angle * 57.2958

# Calculate how to centralize using center of mass of image
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

# Shift image based on values returned by get_best_shift()
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img, M, (rows, cols))
    return shifted

# Get the 4 corners from contours
# These will be the contours of the sudoku board
def get_contour_corners(contours, corner_amt=4, max_iter=200):
    coeff = 1
    while max_iter > 0 and coeff >= 0:
        max_iter = max_iter - 1
        # Maximum distance from contour to approximated contour
        epsilon = coeff * cv2.arcLength(contours, True)
        
        # Approximation of a contour shape, True means curve is closed
        approx = cv2.approxPolyDP(contours, epsilon, True)
        
        # Checks curves for convexity defects
        hull = cv2.convexHull(approx)
        if len(hull) == corner_amt:
            return hull
        else:
            if len(hull) > corner_amt:
                coeff += .01
            else:
                coeff -= .01
    return None

# Prepare and normalize the image for digit recognition
def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype('float32')
    new_array = new_array / 255
    return new_array

def show_image(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

# Function to take a webcam image, find sudoku board, recognize digits, solve puzzle,
# print the result on image and return that image
def recognize_and_solve(image, model, old_sudoku):

    #Convert image to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian BLur to smoothen the image
    # 5 is kernel size, 0 for auto-completion of sigma value
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding
    athresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    
    # Find all contours
    # contours is python list of all contours in image (numpy array of (x,y), coordinates of boundary points)
    contours, _ = cv2.findContours(athresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    biggest = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            biggest = c
    
    # If no sudoku!
    if biggest is None:
        return image
    
    # Get the 4 corners of the biggest contour
    corners = get_contour_corners(biggest, 4)

    # If no sudoku!
    if corners is None:
        return image

    # Now as we have 4 corners, locate the top left, right and bottom left, right corners
    rect = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4,2)

    # Finding top left (sum of coordinates is the smallest)
    sum = 10000
    index = 0
    for i in range(4):
        if(corners[i][0] + corners[i][1] < sum):
            sum = corners[i][0] + corners[i][1]
            index = i

    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Finding bottom right (sum of coordinates is the largest)
    sum = 0
    for i in range(3):
        if(corners[i][0] + corners[i][1] > sum):
            sum = corners[i][0] + corners[i][1]
            index = i
    
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # Finding top right and bottom left (Should be easy)
    if(corners[0][0] > corners[1][0]):
        rect[1] = corners[0]
        rect[3] = corners[1]
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]
    
    rect = rect.reshape(4,2)

    # Now that we have found the 4 corners, check if ABCD is approximately a square
    # A------B
    # |      |
    # |      |
    # D------C

    # The four corners
    A = rect[0]
    B = rect[1]
    C = rect[2]
    D = rect[3]

    # Condition 1: If all four angles not approximately 90 deg, stop.
    # The 4 vectors - AB, AD, BC, CD
    AB = B - A
    AD = D - A
    BC = B - C
    CD = C - D
    eps_angle = 20
    if not (approx_90_deg(angle_between(AB, AD), eps_angle) and approx_90_deg(angle_between(AB, BC), eps_angle) and 
    approx_90_deg(angle_between(BC, CD), eps_angle) and approx_90_deg(angle_between(CD, AD), eps_angle)):
        return image
    
    # Condition 2: The lenghts of AB, AD, BC, CD have to be approximately equal
    # i.e largest and shortest sides have to be approximately equal
    # largest cannot be longer than eps_scale * shortest
    eps_scale = 1.2
    if(side_lengths_too_different(A, B, C, D, eps_scale)):
        return image
    
    ## Now we are sure that ABCD correspond to the 4 corners of the sudoku board

    # width of the sudoku board
    (tl, tr, br, bl) = rect
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # height of the sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # Taking the maximum of the width and height values to reach the final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    # contructing destination points which will be used to get a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")
    
    # calculating perspective transform matrix and warp it to grab the screen
    # perspective transform function is used to implement the top-down transform
    perspective_transformed_matrix = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, perspective_transformed_matrix, (max_width, max_height))
    original_warp = np.copy(warp)

    # Now, the warp only contains the chopped sudoku board
    # We need to do some image processing for recognizing digits
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    warp = cv2.GaussianBlur(warp, (5,5), 0)
    warp = cv2.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv2.bitwise_not(warp)
    _, warp = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)

    # Initializing grid to store the sudoku digits
    SIZE = 9
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)
    
    height = warp.shape[0] // 9 
    width = warp.shape[1] // 9

    # Offset used to get rid of the boundaries
    offset_width = math.floor(width / 10)
    offset_height = math.floor(height / 10)

    # Dividing the sudoku board into 9x9 square
    for i in range(SIZE):
        for j in range(SIZE):

            # Crop with offset to remove the boundaries
            crop_image = warp[height*i+offset_height:height*(i+1)-offset_height, width*j+offset_width:width*(j+1)-offset_width]

            # Boundary lines still left
            # Remove all black lines near edges
            # ratio = 0.6 => If 60% pixels are black, remove them
            # As soon as we reach a non black line, while loop will stop
            ratio = 0.6
            
            # Top
            while np.sum(crop_image[0]) <= (1-ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]

            # Bottom
            while np.sum(crop_image[:,-1]) <= (1-ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)
            
            # Left
            while np.sum(crop_image[:,0]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)
            
            # Right
            while np.sum(crop_image[-1]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]
            
            # Take the largestConnectedComponent (digit), and remove noises
            crop_image = cv2.bitwise_not(crop_image)
            crop_image = largest_connected_component(crop_image)

            # Resize it
            img_size = 28
            crop_image = cv2.resize(crop_image, (img_size, img_size))

            # If it is a white cell, set grid[i][j] to 0 and continue to next image

            # Condition 1: It has too little black pixels
            if crop_image.sum() >= img_size**2*255 - img_size * 1 * 255:
                grid[i][j] = 0
                continue

            # Condition 2: It has huge white area in the center
            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]

            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue

            # We are now certain that the crop_image contains a digit

            # Applying Binary threshold to make the digits more clear
            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY)
            crop_image = crop_image.astype(np.uint8)

            # Centralizing the image according to center of mass
            crop_image = cv2.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image, shift_x, shift_y)
            crop_image = shifted

            crop_image = cv2.bitwise_not(crop_image)

            # Crop image is good and clean
            # Check with
            # cv2.imshow(str(i)+str(j), crop_image)

            # Converting to proper format for recognition
            crop_image = prepare(crop_image)

            # Recognize digits
            # model is trained by digitRecognition.py
            prediction = model.predict([crop_image])
            
            # starts from 0, so add 1
            grid[i][j] = np.argmax(prediction[0]) + 1

    user_grid = copy.deepcopy(grid)

    # Solving the sudoku after recognizing each digit of the board:

    # If same board as last camera frame
    # print the same solution. No need to solve again
    
    if (not old_sudoku is None) and two_matrices_are_equal(old_sudoku, grid, 9, 9):
        if(all_board_non_zero(grid)):
            original_warp = write_soln_on_image(original_warp, old_sudoku, user_grid)
    else: # If its a different board
        solve_sudoku(grid)
        if(all_board_non_zero(grid)):
            original_warp = write_soln_on_image(original_warp, grid, user_grid)
            old_sudoku = copy.deepcopy(grid)
    
    # Applying inverse perspective transform and pasting the solution on top of original image
    result_sudoku = cv2.warpPerspective(original_warp, perspective_transformed_matrix, (image.shape[1], image.shape[0])
                                        , flags=cv2.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1,keepdims=True)!=0, result_sudoku, image)

    return result