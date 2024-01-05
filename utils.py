import cv2
from PIL import Image
import PIL
import numpy as np
import math

def imshow(a):
    a = a.clip(0, 255).astype("uint8")
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))


def video_from_path(path):
    chess = cv2.VideoCapture(path)
    if chess.isOpened():
        print("Video loaded")

    width = int(chess.get(3))
    height = int(chess.get(4))

    print(height, width)

    fps = chess.get(cv2.CAP_PROP_FPS)
    print(fps)

    return chess

def get_first_frame(video, if_display=True):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = video.read()
    if if_display:
        imshow(frame)
    return frame


def create_line_parameters(x1, x2, y1, y2):
    if x1 != x2 and y1 != y2:
        slope = (y2 - y1) / (x2 - x1)

        shift = y1 - slope * x1

        slope_y = (x2 - x1) / (y2 - y1) 

        shift_y = x1 - slope_y * y1
    elif x1 == x2:
        slope = float('inf')

        shift = 0

        slope_y = 0

        shift_y = x1
    else: 
        slope = 0 

        shift = y1

        slope_y = float('inf')

        shift_y = 0

    return shift, slope, shift_y, slope_y

def find_lines(image, if_display=True, minLineLength=50):
    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Perform Canny edge detection
    edges = cv2.Canny(thresh, 50, 70)  # Adjust thresholds (50 and 150) for different levels of edge detection

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=minLineLength, maxLineGap=10)

    line_parameters = list()

    for line in lines:
        x1, y1, x2, y2 = line[0]

        line_parameters.append(create_line_parameters(x1, x2, y1, y2))

    # Draw detected lines on the image
    if lines is not None:
        for shift, slope, shift_y, slope_y in line_parameters:
            if slope < 0.5:
                x1 = 0
                x2 = int(image.shape[1] - 1)
                y1 = int(shift)
                y2 = int(slope * x2 + shift)
            else:
                y1 = 0
                y2 = int(image.shape[0] - 1)
                x1 = int(shift_y)
                x2 = int(slope_y * y2 + shift_y)

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw lines on the original image


    if if_display:
        # Display the original and edge-detected images
        imshow(gray)
        imshow(edges)
        imshow(image)

    return line_parameters

def cluster_tuples(tuples_list, threshold):
    clusters = {}
    
    # Iterate through the tuples
    for tpl in tuples_list:
        second_val = math.atan(tpl[1]) % np.pi
        
        # Check if second value falls within any existing cluster
        assigned = False
        for key in clusters:
            if abs(second_val - key) <= threshold or abs(second_val - key - np.pi) <= threshold or abs(second_val - key + np.pi) <= threshold:
                clusters[key].append(tpl)
                assigned = True
                break
        
        # If the second value doesn't fall within any cluster, create a new cluster
        if not assigned:
            clusters[second_val] = [tpl]
    
    return clusters

def find_square_parameters(image, if_display=True, TARGET_AREA_LOWER = 1e2, TARGET_AREA_UPPER = 1e4):
    line_parameters = list()

    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Remove noise with morph operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(thresh, kernel=np.ones((5,5)))
    eroded  = cv2.erode(thresh, kernel=np.ones((5,5)))
    invert = 255 - dilated

    # Find contours and find squares with contour area filtering + shape approximation
    cnts1 = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
    cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]

    square_parameters = list()
    for cnts in [cnts1, cnts2]:
        for contour in cnts:
            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
            area = cv2.contourArea(approx)

            if len(approx) == 4 and TARGET_AREA_LOWER < area < TARGET_AREA_UPPER:
                s = np.zeros(4)
                for i in range(4):
                    s[i] = np.linalg.norm(approx[i] - approx[(i+1)%4])

                aspect_ratio1 = s[0] / s[2]
                aspect_ratio2 = s[1] / s[3]
                vertices = approx.reshape(-1, 2) 
                if 0.95 <= aspect_ratio1 <= 1.05 and 0.95 <= aspect_ratio2 <= 1.05:
                    if if_display:
                        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)  # Draw contours around squares

                    for i in range(4):
                        x1, y1 = approx[i][0]
                        x2, y2 = approx[(i+1)%4][0]
                        shift, slope, shift_y, slope_y = create_line_parameters(x1, x2, y1, y2)
                        line_parameters.append((shift, slope, shift_y, slope_y))
                        square_parameters.append((s[(i+1)%4], slope))

                        if abs(slope) > 1:
                            new_shift_y = shift_y
                            for j in range(-25, 25):
                                new_shift_y = shift_y - j * s[(i+1)%4]
                                if new_shift_y > 0 and new_shift_y < 90:
                                    y1_new = 0
                                    x1_new = int(new_shift_y)
                                    y2_new = 1000 
                                    x2_new = int(1000 * slope_y + new_shift_y)
                                    line_parameters.append(create_line_parameters(x1_new, x2_new, y1_new, y2_new))

                                    #cv2.line(image, (x1_new, y1_new), (x2_new , y2_new ), (0, 0, 255), 2)


    square_parameters = np.array(square_parameters)
    clusters = cluster_tuples(square_parameters, np.pi / 10)
    gap_lengths = dict()

    if if_display:
        imshow(image)

    for i in clusters:
        gap_length_angle = [item[0] for item in clusters[i]]
        gap_lengths[i] = np.median(gap_length_angle)

    return gap_lengths, line_parameters

def find_chessboard_lines(image, line_parameters, gap_lengths, if_display=True, aim_distance_correction=1.2):
    image = image.copy()

    clustered_line_parameters = cluster_tuples(line_parameters, np.pi / 80)

    chessboard_lines = list()

    for gap_length_angle in gap_lengths:
        #find set of lines closest in respect to slope with the found squares
        cluster_ind = np.argmin(np.abs(np.array(list(clustered_line_parameters.keys())) - gap_length_angle))
        cluster_slope = list(clustered_line_parameters.keys())[cluster_ind]
        aim_distance = gap_lengths[gap_length_angle] * aim_distance_correction

        sin_value = math.sin(np.pi / 2 - cluster_slope)
        
        line_shifts = np.array([item[0] for item in clustered_line_parameters[cluster_slope]])
        line_slopes = np.array([item[1] for item in clustered_line_parameters[cluster_slope]])
        line_shifts_y = np.array([item[2] for item in clustered_line_parameters[cluster_slope]])
        line_slopes_y = np.array([item[3] for item in clustered_line_parameters[cluster_slope]])

        #decide whether the lines are more vertical or horizontal
        if np.abs(np.median(line_slopes)) < 0.5:
            cross_points1 = line_shifts
            cross_points2 = 1000 * line_slopes + line_shifts
            cross_points = (cross_points1 + cross_points2) / 2

        else:
            cross_points1 = line_shifts_y
            cross_points2 = 1000 * line_slopes_y + line_shifts_y
            cross_points = (cross_points1 + cross_points2) / 2

        distances = cross_points[:, np.newaxis] - cross_points


        best_combination = None
        min_cost = float('inf')

        for i in range(len(distances)):
            combination = list()
            cost = 0
            for j in range(9):
                combination.append(np.argmin(np.abs(np.square(distances[i] - aim_distance * j))))
                cost += np.min(np.abs(distances[i] - aim_distance * j))

            if cost < min_cost:
                min_cost = cost
                best_combination = combination

        shifts = line_shifts[best_combination]
        slopes = line_slopes[best_combination]
        shifts_y = line_shifts_y[best_combination]
        slopes_y = line_slopes_y[best_combination]

        chessboard_lines.append(np.stack((shifts, slopes, shifts_y, slopes_y)))

        if if_display:
            if np.abs(np.median(line_slopes)) < 0.5:
                for shift, slope in zip(shifts, slopes):
                    x1 = 0
                    x2 = int(image.shape[1] - 1)
                    y1 = int(shift)
                    y2 = int(slope * x2 + shift)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            else:
                for shift_y, slope_y in zip(shifts_y, slopes_y):
                    y1 = 0
                    y2 = int(image.shape[0] - 1)
                    x1 = int(shift_y)
                    x2 = int(slope_y * y2 + shift_y)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if if_display:
        imshow(image)        

    return chessboard_lines


def find_line_intersection(parameters1, parameters2):
    if abs(parameters1[1]) <= 1 and abs(parameters2[1]) <= 1:
        x = (parameters2[0] - parameters1[0]) / (parameters1[1] - parameters2[1])
        y = parameters1[1] * x + parameters1[0] 
    if abs(parameters1[1]) > 1 and abs(parameters2[1]) > 1:
        y = (parameters2[2] - parameters1[2]) / (parameters1[3] - parameters2[3])
        x = parameters1[3] * y + parameters1[2]
    if abs(parameters1[1]) <= 1 and abs(parameters2[1]) > 1:
        x = (parameters1[0] * parameters2[3] + parameters2[2]) / (1 - parameters1[1] * parameters2[3])
        y = parameters1[1] * x + parameters1[0] 
    if abs(parameters1[1]) > 1 and abs(parameters2[1]) <= 1:
        x = (parameters2[0] * parameters1[3] + parameters1[2]) / (1 - parameters2[1] * parameters1[3])
        y = parameters2[1] * x + parameters2[0] 

    return np.array((x, y))


def find_square_contours(image, chessboard_lines, if_display=True):
    sorted_lines = list()
    image = image.copy()

    for line_set in chessboard_lines:
        to_add = line_set.copy()
        if np.abs(np.median(to_add[1])) < 1:
            sorted_indices = to_add[0].argsort()
        else:
            sorted_indices = to_add[2].argsort()

        sorted_lines.append(to_add[: , sorted_indices])

    lines1 = sorted_lines[0]
    lines2 = sorted_lines[1]

    chessboard_squares = [list() for _ in range(8)]


    for j in range(8):
        for i in range(8):  
            line11 = lines1[:, j]
            line12 = lines1[:, j+1]
            line21 = lines2[:, i]
            line22 = lines2[:, i+1]

            point1 = find_line_intersection(line11, line21)
            point2 = find_line_intersection(line21, line12)
            point3 = find_line_intersection(line22, line12)
            point4 = find_line_intersection(line22, line11)
            chessboard_squares[j].append((point1, point2, point3, point4))


    contours = list()

    for j in range(8):
        for i in range(8):  
            contours.append(np.array(chessboard_squares[i][j]).astype(np.int32))


    if if_display:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        imshow(image)

    return contours

def write_chessboard_detection(video, contours, filename):
    width = int(video.get(3))
    height = int(video.get(4))

    fps = video.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        f"chessboard_detection/{filename}.mp4",  # Updated filename
        cv2.VideoWriter_fourcc(*"mp4v"),  # Codec for MP4 format (codec might vary)
        fps,
        (width, height),
    )

    contours_for_frames = list()

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            writer.write(frame)

            contours_for_frames.append(contours)

        else:
            break

    save_contour_parameters(contours_for_frames, filename)

def draw_bbox(frame, bbox, color=(255, 255, 255)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)


def write_and_detect_chessboard(video, contours, filename):
    # find bottom-left and top-right corner
    points_array = np.array(contours)
    points_array = points_array.reshape(-1, 2)
    sum_diagonal = np.sum(points_array, axis=1)

    ind1 = np.argmin(sum_diagonal)
    ind2 = np.argmax(sum_diagonal)

    point1 = points_array[ind1]
    point2 = points_array[ind2]

    ret, frame = video.read()
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)


    #create tracker for the chessboard

    tracker = cv2.TrackerMIL_create()
    bbox = (point1[0], point1[1], point2[0] - point1[0], point2[1] - point1[1])

    if tracker.init(frame, bbox):
        print("MIL tracker initialized at bounding box:", bbox)



    #initialize writer

    width = int(video.get(3))
    height = int(video.get(4))

    fps = video.get(cv2.CAP_PROP_FPS)

    writer_chessboard = cv2.VideoWriter(
        f"chessboard_detection/{filename}.mp4",  # Updated filename
        cv2.VideoWriter_fourcc(*"mp4v"),  # Codec for MP4 format (codec might vary)
        fps,
        (width, height),
    )


    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    contours_series = list()

    while video.isOpened():
        ret, frame = video.read()
        new_contours = list()

        if ret:
            ok, bbox = tracker.update(frame)
            if ok:
                shift = point1 - bbox[:2]
                for i, _ in enumerate(contours):
                    new_contours.append(contours[i] - shift)
                
                contours_series.append(new_contours)
                cv2.drawContours(frame, new_contours, -1, (0, 255, 0), 2)

            writer_chessboard.write(frame)
        else:
            break

    save_contour_parameters(contours_series, filename)
    writer_chessboard.release()

def write_figure_detection(video, contours_series, filename, maxCorners=4500, qualityLevel=0.01):
    width = int(video.get(3))
    height = int(video.get(4))

    fps = video.get(cv2.CAP_PROP_FPS)

    writer_figures = cv2.VideoWriter(
        f"figure_detection/{filename}.mp4",  # Updated filename
        cv2.VideoWriter_fourcc(*"mp4v"),  # Codec for MP4 format (codec might vary)
        fps,
        (width, height),
    )

    i = 0
    points_in_squares = list()

    while video.isOpened():
        ret, frame = video.read()
        contours = contours_series[i]
        i += 1

        if ret:
            
                    
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            p2, square_count = find_figures(frame, contours, maxCorners, qualityLevel)

            for new in p2:
                a, b = new.ravel()
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (255, 255, 0), -1)



            points_in_squares.append(square_count)
            writer_figures.write(frame2)
        else:
            break

    writer_figures.release()


def find_figures(image, contours, maxCorners=4500, qualityLevel=0.01):
    feature_params = dict(maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=5, blockSize=7)

    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    square_count = np.zeros(64)

    p2 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    filtered_p2 = list()
    for point in p2:
        point_to_process = tuple(point.tolist()[0])

        distances = [cv2.pointPolygonTest(contour, point_to_process, measureDist=True) for contour in contours]

        if np.max(distances) > 10:
            filtered_p2.append(point)
            square_number = np.argmax(distances)
            square_count[square_number] += 1

    filtered_p2 = np.row_stack(filtered_p2)
    p2 = filtered_p2.reshape((-1, 2))

    return p2, square_count

def save_contour_parameters(parameters, path):
    np.save(f'contour_parameters/{path}.npy', parameters)

def load_contour_parameters(path):
    loaded_data = np.load(f'contour_parameters/{path}.npy')
    return loaded_data