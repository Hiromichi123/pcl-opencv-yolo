import cv2
import numpy as np

camera_num=0
frame_width=1280
frame_height=720
clipLimit=2.0
tileGridSize=(8, 8)
broder_width=10
broder_color=(255,255,255)
threshold_thresh=220
threshold_maxval=255
min_contour_area=20000
max_contour_area=50000
min_contour_proportion=0.8
max_contour_proportion=1.25
ROI_broder=5
fps_putPlace_x=20
fps_putPlace_y=20
fps_fontScale=1
fps_fontColor=(255, 0, 255)
fps_fontThickness=2
windows_name: str = 'Camera 1080p'

hsv_colorMask= {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "red": ([120, 50, 50], [180, 255, 250]),
        "yellow": ([20, 100, 100], [40, 255, 255])
    }
hsv_canny_min_thresh=100
hsv_canny_min_thresh=200
hsv_threshold_thresh=220
hsv_threshold_maxval=255
hsv_openKernel=(5,5)
hsv_contour_area_error=1000
HoughCircles_dp=1
HoughCircles_minDist=20
HoughCircles_param1=50
HoughCircles_param2=20
HoughCircles_minRadius=0
HoughCircles_maxRadius=0
HoughCircles_circle_broderColor=(0, 0, 255)
HoughCircles_circle_broderThickness=2
HoughCircles_circle_text_offset_x=40
HoughCircles_circle_text_offset_y=40
HoughCircles_circle_textFontScale=0.5
HoughCircles_circle_textColor=(255, 0, 0)
HoughCircles_circle_textThickness=1
hsv_approxPolyDP_precisionCoefficient=0.04
hsv_approxPolyDP_arcLength_contour_closed=True
hsv_approxPolyDP_contour_closed=True
HoughApprox_shapes_broderColor=(0, 0, 255)
HoughApprox_shapes_broderThickness=2
HoughApprox_shapes_text_offset_x=40
HoughApprox_shapes_text_offset_y=40
HoughApprox_shapes_textFontScale=0.5
HoughApprox_shapes_textColor=(255, 0, 0)
HoughApprox_shapes_textThickness=1

mark_boundingRect_color=(0,255,0)
mark_boundingRect_thickness=2
mark_circleCenterSzie=1
mark_coordinate_text_offset_x=0
mark_coordinate_text_offset_y=-10
mark_coordinate_fontScale=0.5
mark_coordinate_color=(255,0,255)
mark_coordinate_fontThickness=1

#超级函数,62参
def detect_system(camera_num,frame_width,frame_height,clipLimit,tileGridSize,broder_width,broder_color,
                  threshold_thresh,threshold_maxval,min_contour_area,max_contour_area,
                  min_contour_proportion,max_contour_proportion,ROI_broder,
                  fps_putPlace_x,fps_putPlace_y,fps_fontScale,fps_fontColor,fps_fontThickness,
                  windows_name,hsv_colorMask,hsv_canny_min_thresh,hsv_canny_max_thresh,
                  hsv_threshold_thresh,hsv_threshold_maxval,hsv_openKernel,hsv_contour_area_error,
                  HoughCircles_dp,HoughCircles_minDist,HoughCircles_param1,HoughCircles_param2,
                  HoughCircles_minRadius,HoughCircles_maxRadius,HoughCircles_circle_broderColor,
                  HoughCircles_circle_broderThickness,HoughCircles_circle_text_offset_x,
                  HoughCircles_circle_text_offset_y,HoughCircles_circle_textFontScale,
                  HoughCircles_circle_textColor,HoughCircles_circle_textThickness,
                  hsv_approxPolyDP_precisionCoefficient,hsv_approxPolyDP_arcLength_contour_closed,
                  hsv_approxPolyDP_contour_closed,HoughApprox_shapes_broderColor,
                  HoughApprox_shapes_broderThickness,HoughApprox_shapes_text_offset_x,
                  HoughApprox_shapes_text_offset_y,HoughApprox_shapes_textFontScale,
                  HoughApprox_shapes_textColor,HoughApprox_shapes_textThickness,mark_boundingRect_color,
                  mark_boundingRect_thickness,mark_circleCenterSzie,mark_coordinate_text_offset_x,
                  mark_coordinate_text_offset_y,mark_coordinate_fontScale,mark_coordinate_color,
                  mark_coordinate_fontThickness
                  ):
    capture = cv2.VideoCapture(camera_num)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    open, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while open:
        ret, frame = capture.read()
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        top_size, bottom_size, left_size, right_size = (broder_width, broder_width, broder_width, broder_width)
        frame = cv2.copyMakeBorder(frame, top_size, bottom_size, left_size, right_size,
                                   cv2.BORDER_CONSTANT, broder_color)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_thresh, threshold_maxval, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        frame_copy=frame.copy()

        if contours is not None:
            for import_contour in contours:
                area = cv2.contourArea(import_contour)
                x, y, w, h = cv2.boundingRect(import_contour)
                if area > min_contour_area and area < max_contour_area and w / h > min_contour_proportion and w / h < max_contour_proportion:
                    frame_ROI = frame[y-ROI_broder:y + h+ROI_broder,x-ROI_broder:x + w+ROI_broder]
                    if frame_ROI is not None and frame_ROI.shape[0] > 0 and frame_ROI.shape[1] > 0:
                        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        colors = hsv_colorMask

                        for color_name, (lower, upper) in colors.items():
                            mask = cv2.inRange(hsv_img, lower, upper)
                            result = cv2.bitwise_and(frame, frame, mask)
                            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones(hsv_openKernel, np.uint8))
                            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, hsv_canny_min_thresh, hsv_canny_max_thresh)
                            _, thresh = cv2.threshold(edges, hsv_threshold_thresh, hsv_threshold_maxval,
                                                      cv2.THRESH_TRUNC)
                            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                            x, y, w, h = cv2.boundingRect(import_contour)
                            shift_x = x - ROI_broder
                            shift_y = y - ROI_broder

                            for contour in contours:
                                for i in range(len(contour)):
                                    contour[i][0][0] += shift_x
                                    contour[i][0][1] += shift_y

                            for contour in contours:
                                contour_area = cv2.contourArea(contour)
                                if contour_area < area + hsv_contour_area_error and contour_area > area - hsv_contour_area_error:
                                    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, HoughCircles_dp,
                                                               HoughCircles_minDist,
                                                               HoughCircles_param1, HoughCircles_param2,
                                                               HoughCircles_minRadius, HoughCircles_maxRadius)

                                    if circles is not None:
                                        circles = np.uint16(np.around(circles))
                                        for i in circles[0, :]:
                                            center = (shift_x + i[0], shift_y + i[1])
                                            radius = i[2]
                                            cv2.circle(frame_copy, center, radius, HoughCircles_circle_broderColor,
                                                       HoughCircles_circle_broderThickness)
                                            cv2.putText(frame_copy, f"{color_name}_circle",
                                                        (shift_x + i[0] - HoughCircles_circle_text_offset_x,
                                                         shift_y + i[1] - HoughCircles_circle_text_offset_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, HoughCircles_circle_textFontScale,
                                                        HoughCircles_circle_textColor,
                                                        HoughCircles_circle_textThickness)

                                        x, y, w, h = cv2.boundingRect(import_contour)
                                        cv2.rectangle(frame_copy, (x - ROI_broder, y - ROI_broder),
                                                      (x + w + ROI_broder, y + h + ROI_broder),
                                                      mark_boundingRect_color, mark_boundingRect_thickness)
                                        M = cv2.moments(import_contour)
                                        center_x = int(M['m10'] / M['m00'])
                                        center_y = int(M['m01'] / M['m00'])
                                        cv2.circle(frame_copy, (center_x, center_y), mark_circleCenterSzie,
                                                   mark_coordinate_color,
                                                   mark_circleCenterSzie)
                                        cv2.putText(frame_copy, "[" + str(center_x) + "," + str(center_y) + "]",
                                                    (center_x + mark_coordinate_text_offset_x,
                                                     center_y + mark_coordinate_text_offset_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, mark_coordinate_fontScale,
                                                    mark_coordinate_color, mark_coordinate_fontThickness)

                                    approx = cv2.approxPolyDP(contour,
                                                              hsv_approxPolyDP_precisionCoefficient * cv2.arcLength(
                                                                  contour, hsv_approxPolyDP_arcLength_contour_closed),
                                                              hsv_approxPolyDP_contour_closed)
                                    if approx is not None:
                                        M = cv2.moments(contour)
                                        center_x = int(M['m10'] / M['m00'])
                                        center_y = int(M['m01'] / M['m00'])
                                        if len(approx) == 3:
                                            cv2.drawContours(frame_copy, [approx], 0, HoughApprox_shapes_broderColor,
                                                             HoughApprox_shapes_broderThickness)
                                            cv2.putText(frame_copy, f"{color_name}_triangle", (
                                            center_x - HoughApprox_shapes_text_offset_x,
                                            center_y - HoughApprox_shapes_text_offset_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, HoughApprox_shapes_textFontScale,
                                                        HoughApprox_shapes_textColor, HoughApprox_shapes_textThickness)
                                            x, y, w, h = cv2.boundingRect(import_contour)
                                            cv2.rectangle(frame_copy, (x - ROI_broder, y - ROI_broder),
                                                          (x + w + ROI_broder, y + h + ROI_broder),
                                                          mark_boundingRect_color, mark_boundingRect_thickness)
                                            M = cv2.moments(import_contour)
                                            center_x = int(M['m10'] / M['m00'])
                                            center_y = int(M['m01'] / M['m00'])
                                            cv2.circle(frame_copy, (center_x, center_y), mark_circleCenterSzie,
                                                       mark_coordinate_color,
                                                       mark_circleCenterSzie)
                                            cv2.putText(frame_copy, "[" + str(center_x) + "," + str(center_y) + "]",
                                                        (
                                                            center_x + mark_coordinate_text_offset_x,
                                                            center_y + mark_coordinate_text_offset_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, mark_coordinate_fontScale,
                                                        mark_coordinate_color, mark_coordinate_fontThickness)
                                        elif len(approx) == 4:
                                            cv2.drawContours(frame_copy, [approx], 0, HoughApprox_shapes_broderColor,
                                                             HoughApprox_shapes_broderThickness)
                                            cv2.putText(frame_copy, f"{color_name}_square", (
                                            center_x - HoughApprox_shapes_text_offset_x,
                                            center_y - HoughApprox_shapes_text_offset_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, HoughApprox_shapes_textFontScale,
                                                        HoughApprox_shapes_textColor, HoughApprox_shapes_textThickness)
                                            x, y, w, h = cv2.boundingRect(import_contour)
                                            cv2.rectangle(frame_copy, (x - ROI_broder, y - ROI_broder),
                                                          (x + w + ROI_broder, y + h + ROI_broder),
                                                          mark_boundingRect_color, mark_boundingRect_thickness)
                                            M = cv2.moments(import_contour)
                                            center_x = int(M['m10'] / M['m00'])
                                            center_y = int(M['m01'] / M['m00'])
                                            cv2.circle(frame_copy, (center_x, center_y), mark_circleCenterSzie,
                                                       mark_coordinate_color,
                                                       mark_circleCenterSzie)
                                            cv2.putText(frame_copy, "[" + str(center_x) + "," + str(center_y) + "]",
                                                        (
                                                            center_x + mark_coordinate_text_offset_x,
                                                            center_y + mark_coordinate_text_offset_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, mark_coordinate_fontScale,
                                                        mark_coordinate_color, mark_coordinate_fontThickness)

        fps = capture.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, "FPS:" + str(fps), (fps_putPlace_x, fps_putPlace_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_fontScale, fps_fontColor, fps_fontThickness)

        cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
        cv2.imshow(windows_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
        break