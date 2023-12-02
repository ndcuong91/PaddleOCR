import os, shutil

#1a887e3b-cccd_bc5e2fab34bb41fdb51c1334f699dcb4638182348616832506_1_0.jpg, "IDVNM1750068359001175006835<<0"

char_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz| 0123456789ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờùừỳảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ\'*:,@.-(#%")/~!^&_+={}[]\;<>?※”$€£¥₫°²™ā–'

def convert_crnn_data_to_paddle_data(crnn_dir, train_path, val_path, output_dir):
    '''
    Convert dữ liệu huấn luyện của crnn sang paddleOCR
    :param crnn_dir:
    :param train_path:
    :param val_path:
    :return:
    '''

    with open(train_path, mode = 'r', encoding='utf-8') as f:
        list_train = f.readlines()

    train_txt = []
    for idx, line in enumerate(list_train):
        img_path = line.replace('\n','')
        anno_path = os.path.join(crnn_dir, '.'.join(img_path.split('.')[:-1])+'.txt')

        with open(anno_path, mode = 'r', encoding='utf-8') as f:
            anno = f.read()
            anno = anno.replace('\n','')

        line = ','.join([img_path, '"{}"'.format(anno)])
        train_txt.append(line)

        dst_path = os.path.join(output_dir, img_path)
        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(os.path.join(crnn_dir,img_path), dst_path)

    train_txt = '\n'.join(train_txt)

    out_train_path = os.path.join(output_dir,'train.txt')
    with open(out_train_path, mode='w', encoding='utf-8') as f:
        f.write(train_txt)

    with open(val_path, mode='r', encoding='utf-8') as f:
        list_val = f.readlines()

    val_txt = []
    for idx, line in enumerate(list_val):
        img_path = line.replace('\n', '')
        anno_path = os.path.join(crnn_dir, '.'.join(img_path.split('.')[:-1]) + '.txt')

        with open(anno_path, mode='r', encoding='utf-8') as f:
            anno = f.read()
            anno = anno.replace('\n', '')

        line = ','.join([img_path, '"{}"'.format(anno)])
        val_txt.append(line)

        dst_path = os.path.join(output_dir, img_path)
        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(os.path.join(crnn_dir, img_path), dst_path)

    val_txt = '\n'.join(val_txt)

    out_val_path = os.path.join(output_dir, 'val.txt')
    with open(out_val_path, mode='w', encoding='utf-8') as f:
        f.write(val_txt)

    char = '\n'.join(char_list)
    with open(os.path.join(output_dir, 'char_dict.txt'), mode='w', encoding='utf-8') as f:
        f.write(char)

if __name__ =='__main__':
    convert_crnn_data_to_paddle_data(crnn_dir='/home/duycuong/home_data/ocr_data/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/cleaned_data_02Mar',
                                     train_path='/home/duycuong/home_data/ocr_data/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/cleaned_data_02Mar/train.txt',
                                     val_path='/home/duycuong/home_data/ocr_data/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/cleaned_data_02Mar/val.txt',
                                     output_dir='/home/duycuong/home_data/ocr_data/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/cleaned_data_02Mar_paddle')




