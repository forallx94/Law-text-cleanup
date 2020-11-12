from law_preprocessing import seperate_clause, doc2txt ,erase_
from law_pd import seperate_Content, full_index,self_Content , set_dict

import pandas as pd
import re
import os


Data_path = r"C:\Users\POP\Desktop\work\5-4\법 정규편식\위험물안전관리법.doc"
law_name = '위험물안전관리법 '

write_path = r'C:\Users\POP\Desktop\work\5-4\법 정규편식'

# 꺽쇠 유지 변수, 0 이면 유지 1 이면 삭제
brackets_exsit = 1

DocContents = doc2txt(Data_path)
DocContents = erase_(DocContents)

my_df = pd.DataFrame(columns = ["index", "Content"])
my_df = seperate_Content(my_df,DocContents,law_name)
my_df = full_index(my_df,law_name)

# my_df의 Content 수정 하는 작업 (재작중). 
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
    
    
my_df.to_excel(r'C:\Users\POP\Desktop\work\5-4\법 정규편식\example.xlsx')

f = open(os.path.join(write_path , 'example.txt'), 'w', encoding='UTF8')
for index, content in zip(my_df["index"],my_df["Content"]):
    if index != None:
        f.write(index)
        f.write('   ')
    f.write(content)
    
f.close()

