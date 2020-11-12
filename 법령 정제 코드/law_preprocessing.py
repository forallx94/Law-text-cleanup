import win32com.client as win32
import glob
import os
import re

# doc를 텍스트화 하여 받는 코드
def doc2txt(Data_path):
    files = glob.glob(Data_path)

    for fname in files:
        word = win32.Dispatch("Word.Application")
        # word.Visible = 0

        doc1 = word.Documents.Open(fname)

        DocContents = str(doc1.Content.Text)
        DocContents = DocContents.replace('\r', '\n')
        word.ActiveDocument.Close()
        word.Quit()
    return DocContents


def erase_(docs):
    re.compile(r'<.*?>')
    docs = re.sub(r'<.*?>','',docs)
    return docs


# 유니코드를 이용한 ① 1로 변경하는 과정
def uncircle(s):
    for i in range(1, 21):
        s = s.replace(chr(0x245f + i), str(i))
    return s.replace('\u24ea', '0')


def seperate_other_law(line, input_list = []):
    temp = [x for x in re.finditer('「',line)]
    if len(temp) == 0:
        input_list += [line]
        return input_list
    if len(temp) == 1:
        T = [line[:temp[0].span(0)[0]], line[temp[0].span(0)[0]:]]
        while True:
            try: T.remove('')
            except: break
        input_list += T
        return input_list
    line_num = temp[len(temp)//2].span(0)[0]
    seperate_other_law(line[:line_num],input_list)
    seperate_other_law(line[line_num:],input_list)
    return input_list
    

a = re.compile('(제\d+조(의\d+)?)(제\d+항(의\d+)?)(제\d+호(의\d+)?)|'+\
               '(제\d+조(의\d+)?)(제\d+호(의\d+)?)|'+
               '(제\d+조(의\d+)?)(제\d+항(의\d+)?)|'+\
               '(제\d+조(의\d+)?)|'+\
               '(제\d+항(의\d+)?)(제\d+호(의\d+)?)|'+\
               '(제\d+항(의\d+)?)|'+\
               '(제\d+호(의\d+)?)')

    
def seperate_clause(line,input_list = []):
    global a
    temp = [x for x in re.finditer(a,line)]
    if len(temp) == 0:
        input_list += [line]
        return input_list
    if len(temp) in [1,2]:
        T = [line[:temp[0].span(0)[1]], line[temp[0].span(0)[1]:]]
        while True:
            try: T.remove('')
            except: break
        input_list += T
        return input_list
    line_num = temp[len(temp)//2].span(0)[1]
    seperate_clause(line[:line_num],input_list)
    seperate_clause(line[line_num:],input_list)
    return input_list
