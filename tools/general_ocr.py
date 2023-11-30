'''
Requirements
paddleocr==2.5.0.3
paddlepaddle==2.2.2
'''


import os, cv2, time, math, shutil, inspect
import numpy as np
# import matplotlib.pyplot as plt
from unidecode import unidecode
from paddleocr import PaddleOCR
from paddleocr.tools.infer.predict_system import sorted_boxes

# replace file
this_dir = os.path.dirname(__file__)
# dst_path_paddleocr = inspect.getfile(unidecode).replace('unidecode/__init__.py', 'paddleocr/paddleocr.py')
# src_path_paddleocr = os.path.join(this_dir, 'models/paddleocr_mod/paddleocr.py')
# dst_path_predict_rec = inspect.getfile(unidecode).replace('unidecode/__init__.py',
#                                                           'paddleocr/tools/infer/predict_rec.py')
# src_path_predict_rec = os.path.join(this_dir, 'models/paddleocr_mod/predict_rec.py')
# dst_path_predict_det = inspect.getfile(unidecode).replace('unidecode/__init__.py',
#                                                           'paddleocr/tools/infer/predict_det.py')
# src_path_predict_det = os.path.join(this_dir, 'models/paddleocr_mod/predict_det.py')
# if os.path.exists(dst_path_paddleocr) and os.path.exists(dst_path_predict_rec):
#     shutil.copy(src_path_paddleocr, dst_path_paddleocr)
#     shutil.copy(src_path_predict_rec, dst_path_predict_rec)
#     shutil.copy(src_path_predict_det, dst_path_predict_det)
#
# time.sleep(10)


DEBUG_PRINT = True

ocr = PaddleOCR(use_angle_cls=False,
                use_gpu=False,
                det_db_box_thresh=0.45,
                rec_batch_num=4,
                use_onnx=False,
                # det_model_dir=os.path.join(this_dir, 'models/paddleocr_mod/det/text_dec_v20.onnx'),
                # cls_model_dir=os.path.join(this_dir, 'paddleocr_mod/cls'),
                # rec_model_dir=os.path.join(this_dir, 'models/paddleocr_mod/rec/vi/text_rec_v23.onnx'),
                rec_model_dir = '/home/misa/PycharmProjects/MISA.GKSExtraction/birthcert_services/core/models/paddleocr_mod/rec/vi/text_rec_v23.onnx',
                rec_char_dict_path='/home/misa/PycharmProjects/MISA.GKSExtraction/birthcert_services/core/models/paddleocr_mod/rec/vi/vi_dict.txt') # need to run only once to download and load model into memory
print('{:=^80}'.format('Done load paddleOCR model'))


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

def preprocessing(img, crop_ratio=(0.01, 0.01, 0.01, 0.01), max_side_len=480):
    """tiền xử lý ảnh --> tối ưu thời gian chạy và độ chính xác do chỉ cần lấy thông tin ngày cấp trên passport
    :param img: ảnh numpy array
    :param crop_ratio: tỉ lệ crop vào theo thứ tự left, top, right, bottom
    :param max_side_len: kích cỡ tối đa resize về
    :return:
    """
    w, h = img.shape[1], img.shape[0]

    left = int(crop_ratio[0] * w)
    right = int(w * (1 - crop_ratio[2]))
    top = int(crop_ratio[1] * h)
    bottom = int((1 - crop_ratio[3]) * h)

    crop_img = img[top:bottom, left:right]

    print(50*'-', 'crop_img', crop_img.shape)

    crop_w, crop_h = crop_img.shape[1], crop_img.shape[0]

    if crop_w > crop_h:
        new_w = max_side_len
        new_h = int(max_side_len * crop_h / crop_w)
    else:
        new_h = max_side_len
        new_w = int(max_side_len * crop_w / crop_h)

    final_img = cv2.resize(crop_img, (new_w, new_h))

    # cv2.imshow('crop img', final_img)
    return final_img



def order_points(pts):
    """Order a point's coordinates in a clockwise direction.
    """

    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect

def rotate_and_crop(img, points, debug=False, extend=True,
                    extend_x_ratio=1, extend_y_ratio=0.01,
                    min_extend_y=1, min_extend_x=2, wh_thres=15):
    '''

    :param img:
    :param points: list [4,2]
    :param debug:
    :param extend:
    :param extend_x_ratio:
    :param extend_y_ratio:
    :param min_extend_y:
    :param min_extend_x:
    :return:
    '''

    default_val = {'min_extend_x': 2,
                   'extend_x_ratio': 0.1,
                   'min_extend_y': 2,
                   'extend_y_ratio': 0.1}

    ### Warning: points must be sorted clockwise
    points = order_points(np.asarray(points))

    w = int(euclidean_distance(points[0], points[1]))
    h = int(euclidean_distance(points[1], points[2]))

    if w / h > wh_thres:
        min_extend_x = default_val['min_extend_x']
        extend_x_ratio = default_val['extend_x_ratio']
        min_extend_y = default_val['min_extend_y']
        extend_y_ratio = default_val['extend_y_ratio']

    # get width and height of the detected rectangle
    if extend:
        ex = min_extend_x if (extend_x_ratio * h) < min_extend_x else (extend_x_ratio * h)
        ey = min_extend_y if (extend_y_ratio * h) < min_extend_y else (extend_y_ratio * h)
        ex = int(round(ex))
        ey = int(round(ey))
    else:
        ex, ey = 0, 0
    src_pts = points.astype("float32")
    dst_pts = np.array([
        [ex, ey],
        [w - 1 + ex, ey],
        [w - 1 + ex, h - 1 + ey],
        [ex, h - 1 + ey]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(img, M, (w + 2 * ex, h + 2 * ey))

    if debug:
        print('wh_ratio, ex, ey', round(w / h, 2), ex, ey)
        cv2.imshow('rotate and extend', warped)
        cv2.waitKey(0)
    return warped


def run_ocr(img, img_fname):
    '''
    :param info_img:
    :return: tuple (name, issue_date, issue_date_conf)
    '''

    print(f'image shape: {img.shape}')                # image shape (h, w, c)

    # Text detection + image cropping 
    dt_bboxes, elapse = ocr.text_detector(img)        # a list of bbox detected, each bbox has a shape of (4, 2) with each point having format of (w, h) - equivalent to (x, y)

    if DEBUG_PRINT:
        print("dt_boxes num : {}, elapse : {}".format(len(dt_bboxes), elapse))

    if dt_bboxes is None:
        return None, None
    dt_bboxes = sorted_boxes(dt_bboxes)

    # Display original textbox detected 
    # disp_img = np.copy(img)
    # for i, bbox in enumerate(dt_bboxes):
    #     print(i, bbox)
    #     bbox = np.int64(bbox)
    #     cv2.drawContours(disp_img, [bbox], 0, (0, 0, 255), 2)
    #     cv2.putText(disp_img, str(i), tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # disp_img = cv2.resize(disp_img, (700, 1000))
    # cv2.imshow('original detected bbox', disp_img)
    # cv2.waitKey(0)
    
    crop_imgs = []
    for bbox in dt_bboxes:
        # if bbox[0][0] < img.shape[1] / 2:
        crop_img = rotate_and_crop(img, np.array(bbox).astype(np.int32).tolist(),     # each crop_img has a format of (h, w, c)
                                    debug=False,
                                    extend=True,
                                    min_extend_x=2,
                                    extend_x_ratio=0.05,
                                    min_extend_y=5,
                                    extend_y_ratio=0.25)
        crop_imgs.append(crop_img)

    # Text recognition 
    recog_res, elapse = ocr.text_recognizer(crop_imgs)

    if DEBUG_PRINT:
        print("rec_res num  : {}, elapse : {}".format(len(recog_res), elapse))

    # # Save results to text file 
    # with open(img_fname.split('.')[0] + '.txt', 'w') as f:
    #     f.write(img_fname + '\n')
    #     for i, res in enumerate(recog_res):
    #         f.write(str(i) + ' ' + str(res) + '\n')
    #         print(res)
    #         cv2.imshow('ocr_result', crop_imgs[i])
    #         cv2.waitKey(0)

    # Postprocessing
    ocr_res = []
    for bbox, recog_res_ in zip(dt_bboxes, recog_res):
        text, score = recog_res_
        if score >= 0.5:
            res = [bbox, text, score]    # format of (bbox, text, score)
            ocr_res.append(res)

    return ocr_res 

# DEBUG_PRINT = False
# run(np.ones((800, 800, 3)))
# DEBUG_PRINT = True

def test():
    def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG']):
        included_extensions = ext
        file_names = [fn for fn in os.listdir(dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]
        file_names = sorted(file_names)
        return file_names


if __name__ == '__main__':
    
    # img_dir = '/home/misa/Documents/eKYC_project/data/birth_certi_data/'
    # # # img = '/home/misa/Downloads/eKYC_text_det_1060/images/0b595d43-cccd_e366bc2603c04237a5f109064f7c8289638206075695305363_0.jpg
    # for img_fname in os.listdir(img_dir): 
    #     if img_fname.endswith('.jpg') or img_fname.endswith('.png'):
    #         img_fname = os.path.join(img_dir, img_fname)
    # #         run(cv2.imread(fname), fname) 
    # #         break
    # 
    # # img_fname = '/home/misa/Documents/eKYC_project/data/birth_certi_data/cam_3c.jpg' 
    #         ocr_res = run_ocr(cv2.imread(img_fname), img_fname)
    #         with open(img_fname.split('.')[0] + '.txt', 'w') as f: 
    #             for i, res in enumerate(ocr_res): 
    #                 f.write(str(i) + ' ' + res[1] + ' ' + str(res[2]) + '\n')
    # 
    #         with open(img_fname.split('.')[0] + '_bboxes.txt', 'w') as f:
    #             for i, res in enumerate(ocr_res): 
    #                 f.write(str(res[0]) + '\n')

    crop_imgs = [cv2.imread('/home/misa/Pictures/354773370_1593796987770999_8866583178461392364_n.jpg')]

    recog_res, elapse = ocr.text_recognizer(crop_imgs)
    print(recog_res)
