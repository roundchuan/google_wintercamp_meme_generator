# -*- coding: UTF-8 -*-
from aip import AipOcr
import re
import base64
APP_ID='^.^'
API_KEY ='^.^'
SECRECT_KEY='^.^'
client=AipOcr(APP_ID,API_KEY,SECRECT_KEY)
li = open("lis.txt", "r")
for  line in  li.readlines():
    filename = line
#filename='aa/1.jpg'
    f = open(filename[:-1], 'rb')
    img=f.read()
    import io
    import os
 
    from PIL import Image 
    def IsValidImage(pathfile):
        bValid = True
        try:
            Image.open(pathfile).verify()
        except:
            bValid = False
        return bValid

    def is_valid_jpg(jpg_file):  
        #if jpg_file.split('.')[-1].lower() == 'jpg':  
       # with open(jpg_file, 'rb') as f:  
       #     f.seek(-2, 2)  
       #     return f.read() == '\xff\xd9' 
       # else:  
        return True

    def is_jpg(filename):
        try:
            i=Image.open(filename)
            return i.format =='JPEG'
        except IOError:
            return False
    fs_a = 'WW'
    fs_b = 'COM'
    fs_c = 'ww'
    fs_d = 'com'
    fs_e = 'net'
    fs_f = 'NET'
    import re
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    def is_Chinese(word):
        word_new=''
        for ch in word:
            global zh_pattern
            match = zh_pattern.search(ch)
            if match:
                word_new = word_new+ch
        return word_new

    def check_contain_chinese(check_str):
        for ch in check_str.decode('utf-8'):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
    if is_valid_jpg(f):
	if is_jpg(f):
            message=client.basicGeneral(img)
            if 'error_code' in message:
                continue
            with open("1.txt", "a") as fo:
                print(message)
                if len(message.get('words_result')) > 0:
                    c = " "
                    for text in message.get('words_result'):
                         #print text.get('words')
                         if fs_a in text.get('words'):
                             print 1
                             continue
                         if fs_b in text.get('words'):
                             print 1
                             continue
                         if fs_c in text.get('words'):
                             print 1
                             continue
                         if fs_d in text.get('words'):
                             print 1
                             continue
                         if fs_e in text.get('words'):
                             continue
                         if fs_f in text.get('words'):
                             continue
                         aa = u'网'
                         if aa in text.get('words'):
                             continue
                         new_words = is_Chinese(text.get('words'))
                         #    continue
                         c = c + new_words
                    print c
                    if len(c) > 1:
                         fo.write(filename[:-1] + c.encode('utf-8') + "\n")
