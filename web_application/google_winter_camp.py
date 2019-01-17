
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from play import beam_search
from PIL import Image, ImageDraw, ImageFont
from datetime import timedelta
from skimage import io
import numpy as np
from pylab import *

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


img_count = 0

# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    global img_count
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(str(img_count)))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        img_count += 1

        user_input = ''.join(beam_search(upload_path))

        # 使用Opencv转换一下图片格式和名称
        img = text_image(upload_path, user_input)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        return render_template('upload_ok.html', userinput=user_input)

    return render_template('upload.html')


def text_image(path,value):
    im = Image.open(path)
    im = im.resize((255, 255),Image.ANTIALIAS)
    font = ImageFont.truetype(os.getcwd() + '/NotoSerifCJKsc-SemiBold.otf', 20, index=0)
    fillColor = (255,255,255)
    strname = value
    length = len(value)
    draw = ImageDraw.Draw(im)
    if length<20:  #字符串很长的情况下
        an = (im.size[0] - length * 10) / 2.  #判断字符串到图片左侧的距离
        draw.text((20, 225), strname, fill=(255, 255, 255), font=font)  # 文字写入
    elif 20<=length & length<=40:
        an1 = (im.size[0] - 20 * 10) / 2   # 第一行
        an2=(im.size[0] - (length-20) * 10) / 2  #第二行
        #a1,a2=CutOut(strname)
        draw.text((20,205), strname[:10], fill=(255,255,255), font=font)
        draw.text((20,225), strname[10:], fill=(255,255,255), font=font)
    else:
        draw.text((20,205), strname[:10], fill=(255,255,255), font=font)
        contents='%s...'%strname[10:20]
        draw.text((20,225), contents, fill=(255,255,255), font=font)
    #draw.text(position, strname, font=font, fill=fillColor)
    img_OpenCV = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR) #PIL读取图片和OpenCV读取rgb图片的三个通道顺序不同
    return img_OpenCV


if __name__ == '__main__':
    # app.debug = True
    app.run('0.0.0.0', port=5001)
