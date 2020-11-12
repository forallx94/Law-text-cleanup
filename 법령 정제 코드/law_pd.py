from law_preprocessing import uncircle

import re
import pandas as pd

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

