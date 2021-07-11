import win32com.client as win32
import pandas as pd
import re, os, glob


def seperate_Content(my_df,DocContents,law_name):
    # 절을 구분하는 과정
    # 대부분의 법이 맨 앞에 있는 것을 이용하여 구분한다.
    clause = re.compile(r'^[①-⑳] ?|^(\d{1,2}의*\d*\. )|^[가,나,다,라,마,바,사,아,자,차,카,타,파,하]\. ?')
    clause_jo = re.compile(r'^(제\d+조의*\d*)(\(.*?\))? ')
    clause_singularity_jo = re.compile(r'^(제\d+조의*\d*)(\(.*?\) )([①-⑳] ?)')


    docs = DocContents.split('\n\n')
    for doc in docs:
        lines = doc.split('\n')
        for line in lines:
            if clause_singularity_jo.search(line):
                temp = clause_singularity_jo.search(line)
                line = line[temp.span()[1]:]
                a = pd.DataFrame(data = [[temp.groups()[0] , temp.groups()[1]]], columns = ["index", "Content"])
                my_df = my_df.append(a)
                a = pd.DataFrame(data = [[temp.groups()[2] , line]], columns = ["index", "Content"])
                my_df = my_df.append(a)
                continue
            elif clause_jo.search(line):
                temp = clause_jo.search(line)
                line = line[temp.end(1):]
                a = pd.DataFrame(data = [[temp.groups()[0] , line]], columns = ["index", "Content"])
                my_df = my_df.append(a)
                continue            
            elif clause.search(line):
                temp = clause.search(line)
                line = line[temp.span()[1]:]
                a = pd.DataFrame(data = [[temp[0] , line]], columns = ["index", "Content"])
                my_df = my_df.append(a)
                continue
            else:
                a = pd.DataFrame(data = [[None,line]], columns = ["index","Content"])
                my_df = my_df.append(a)
        a = pd.DataFrame(data = [[None,'\n']], columns = ["index","Content"])
        my_df = my_df.append(a)
    return my_df


def full_index(my_df,law_name):
    #my_df의 index를 수정하는 작업
    jo = re.compile(r'^(제\d{1,}조)(의*\d*)')
    hang = re.compile(r'^[①-⑳]')
    ho = re.compile(r'^\d{1,3}의*\d*\.')
    mok = re.compile('^([가,나,다,라,마,바,사,아,자,차,카,타,파,하])\. ?')
    
    for num, i in enumerate(my_df["index"]):
        if my_df.iloc[num]["Content"] == '\n':
            clause_dict = {'법':law_name,'조': '','항':'','호':'','목':''}
            
        if i != None:
            if jo.search(i):
                clause_dict['조'] = jo.search(i)[0]
                my_df.iloc[num]["index"] = clause_dict['법'] + clause_dict['조']
                continue
            
            elif hang.search(i):
                clause_dict['항'] = '제' + uncircle(hang.search(i)[0]) + '항'
                my_df.iloc[num]["index"] = clause_dict['법'] + clause_dict['조'] + clause_dict['항']
                continue
            
            elif ho.search(i):
                clause_dict['호'] = '제' + ho.search(i)[0][:-1] + '호'
                my_df.iloc[num]["index"] = clause_dict['법'] + clause_dict['조'] + clause_dict['항'] + clause_dict['호']
                continue
            
            elif mok.search(i):
                clause_dict['목'] = mok.search(i).group(1) + '목'
                my_df.iloc[num]["index"] = clause_dict['법'] + clause_dict['조'] + clause_dict['항'] + clause_dict['호'] + clause_dict['목']
                continue
            
    return my_df
  
# 앞의 clause에서 clause_dict 을 재설정
def set_dict(line,clause_dict):
    a = re.compile('(제\d+조(의\d+)?)(제\d+항(의\d+)?)?(제\d+호(의\d+)?)?(\d+목(의\d+)?)?')
    temp = a.findall(line)
    for num, i in enumerate(clause_dict):
        if i == '법':
            continue
        clause_dict[i] = temp[0][2*(num-1)]
    return clause_dict

# Content 의 내용 수정 함수 
def self_Content(line, clause_dict):
    clause_list = [re.compile('(제\d+조(의\d+)?)(제\d+항(의\d+)?)(제\d+호(의\d+)?)'),\
         re.compile('(제\d+조(의\d+)?)(제\d+항(의\d+)?)'),\
         re.compile('(제\d+조(의\d+)?)'),\
         re.compile('(제\d+항(의\d+)?)(제\d+호(의\d+)?)'),\
         re.compile('(제\d+항(의\d+)?)'),\
         re.compile('(제\d+호(의\d+)?)')]

    if not any([x.search(line) for x in clause_list]):
        return line
    
    if re.search(clause_list[0],line) != None:        
        temp = clause_list[0].findall(line)
        for num, i in enumerate(clause_dict):
            if i == '법':
                continue
            if i == '조':
                clause_dict[i] = temp[0][0]
            if i == '항':
                clause_dict[i] = temp[0][2]
            if i == '호':
                clause_dict[i] = temp[0][4]
            if i == '목':
                clause_dict[i] = ''
        temp = ''
        for i in clause_dict:
            temp += clause_dict[i]
        new_line = re.sub(clause_list[0],temp,line)
        return new_line
    
    elif re.search(clause_list[1],line) != None:
        temp = clause_list[1].findall(line)
        for num, i in enumerate(clause_dict):
            if i == '법':
                continue
            if i == '조':
                clause_dict[i] = temp[0][0]
            if i == '항':
                clause_dict[i] = temp[0][2]
            if i == '호':
                clause_dict[i] = ''
            if i == '목':
                clause_dict[i] = ''
        temp = ''
        for i in clause_dict:
            temp += clause_dict[i]
        new_line = re.sub(clause_list[1],temp,line)
        return new_line
    
    elif re.search(clause_list[2],line) != None:
        temp = clause_list[2].findall(line)
        for num, i in enumerate(clause_dict):
            if i == '법':
                continue
            if i == '조':
                clause_dict[i] = temp[0][0]
            if i == '항':
                clause_dict[i] = ''
            if i == '호':
                clause_dict[i] = ''
            if i == '목':
                clause_dict[i] = ''
        temp = ''
        for i in clause_dict:
            temp += clause_dict[i]
        new_line = re.sub(clause_list[2],temp,line)
        return new_line

    elif re.search(clause_list[3],line) != None:
        temp = clause_list[3].findall(line)
        for num, i in enumerate(clause_dict):
            if i == '법':
                continue
            if i == '조':
                continue
            if i == '항':
                clause_dict[i] = temp[0][0]
            if i == '호':
                clause_dict[i] = temp[0][2]
            if i == '목':
                clause_dict[i] = ''
        temp = ''
        for i in clause_dict:
            temp += clause_dict[i]
        new_line = re.sub(clause_list[3],temp,line)
        return new_line

    elif re.search(clause_list[4],line) != None:
        temp = clause_list[4].findall(line)
        for num, i in enumerate(clause_dict):
            if i == '법':
                continue
            if i == '조':
                continue
            if i == '항':
                clause_dict[i] = temp[0][0]
            if i == '호':
                clause_dict[i] = ''
            if i == '목':
                clause_dict[i] = ''
        temp = ''
        for i in clause_dict:
            temp += clause_dict[i]
        new_line = re.sub(clause_list[4],temp,line)
        return new_line

    elif re.search(clause_list[5],line) != None:
        temp = clause_list[5].findall(line)
        for num, i in enumerate(clause_dict):
            if i == '법':
                continue
            if i == '조':
                continue
            if i == '항':
                continue
            if i == '호':
                clause_dict[i] = temp[0][0]
            if i == '목':
                clause_dict[i] = ''
        temp = ''
        for i in clause_dict:
            temp += clause_dict[i]
        new_line = re.sub(clause_list[5],temp,line)        
        return new_line



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

if __name__ == "__main__":

    Data_path = "./위험물안전관리법.doc"
    law_name = '위험물안전관리법'

    write_path = './법 정규편식'
    os.makedirs(write_path, exist_ok=true)

    # 꺽쇠 유지 변수, 0 이면 유지 1 이면 삭제
    brackets_exsit = 1

    DocContents = doc2txt(Data_path)
    DocContents = erase_(DocContents)

    my_df = pd.DataFrame(columns = ["index", "Content"])
    my_df = seperate_Content(my_df,DocContents,law_name)
    my_df = full_index(my_df,law_name)

    # my_df의 Content 수정 하는 작업. 
    Content_law = re.compile( r'「(.*?)」')

    clause = re.compile('(제\d+조(의\d+)?)(제\d+항(의\d+)?)(제\d+호(의\d+)?)|'+\
                '(제\d+조(의\d+)?)(제\d+호(의\d+)?)|'+
                '(제\d+조(의\d+)?)(제\d+항(의\d+)?)|'+\
                '(제\d+조(의\d+)?)|'+\
                '(제\d+항(의\d+)?)(제\d+호(의\d+)?)|'+\
                '(제\d+항(의\d+)?)|'+\
                '(제\d+호(의\d+)?)')

    for num, line in enumerate(my_df["Content"]):
        # 다른 법이 나왔으나 해당 법의 구체적인 조항이 언급되었는지 언급이 없는지 구별해야
        if re.search(r'\[.*?\]',line):
            my_df.iloc[num]["Content"] = ''
            continue
        clause_dict = {'법':law_name,'조': '','항':'','호':'','목':''}
        try:
            clause_dict = set_dict(my_df.iloc[num]['index'], clause_dict)
        except:
            pass
        lines = seperate_clause(line,[])
        new_line =''
        for i in lines:
            if Content_law.search(i) and clause.search(i):
                clause_dict['법'] = Content_law.search(i).group(brackets_exsit)
                i = re.sub(Content_law,'',i)
            if Content_law.search(i) and not clause.search(i):
                i = re.sub(Content_law,'\g<%d>'%brackets_exsit,i)
            new_line += self_Content(i, clause_dict)
        my_df.iloc[num]["Content"] = new_line
        
        
    my_df.to_excel('.\{}.xlsx'.format(law_name))

    f = open(os.path.join(write_path , '{}.txt'.format(law_name)), 'w', encoding='UTF8')
    for index, content in zip(my_df["index"],my_df["Content"]):
        if index != None:
            f.write(index)
            f.write('   ')
        f.write(content)
        
    f.close()

