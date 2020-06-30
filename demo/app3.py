import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import docx2txt
import PyPDF2 
from pyresparser import ResumeParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()
import glob
import textract
from pyresparser import ResumeParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
import numpy
import numpy as np
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx','doc'])


    #return render_template('multiple_files.html')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    #return('hello')
    return render_template('multiple_files.html')
    
@app.route('/dropdown', methods=['POST'])
def dropdown():
    skills={}
    ds3=[]
    final3={}
    main3={}
    education=[]
    edu_matching=[]
    exp_final=[]
    
    edu_abb={'MCA':['MCA','M.C.A','M.C.A.'],'BACHELORS_Tech':['B-TECH', 'B.TECH','BTECH','B.TECH.'],'MASTERS_Tech':['M-TECH', 'M.TECH','MTECH','M.TECH.'],'POST_GRADUATION':['M.E', 'M.E.','M-TECH', 'M.TECH', 'MTECH','MCA','MBA','M.TECH.'],'UNDER_GRADUATION':['B.E.', 'B.E', 'BS', 'B.S','B-TECH','BSC','B.SC','B.SC.', 'B.TECH','BTECH','B.TECH.'],'BACHELORS_Science':['BSC','B.SC','B.SC.'] }

    technologies=['java','python','.net','jquery','c#.net','mvc','jquery','javascript','python','flask','sql',
                   'asp.net','c#','azure','aws','agile','react','mysql','angular','kendo','machinelearning','artificialinteligence','ajax','vb.net','ado.net','neuralnetworks']
                   
    if request.method == 'POST':
        print('bjfbjksfkdjfj',request.form)
        resume_all_names = request.form.getlist('resume_files[]')
        education = request.form.getlist('edu')
        radio_rating = request.form.get('rating_radio')
        radio_exp = request.form.get('exp_radio')
        #print('radio_rating',radio_rating)
        experiences = []
        for k,v in request.form.items():
            if k.startswith( 'kk-' ):
                ll=k.split('-')
                experiences.append((ll[1],v))

        for key, value in request.form.items():
            if key in technologies:
                skills[key]=value
        ee=[]
        for i in edu_abb.keys():
            if i in education:
                for k in edu_abb[i]:
                    ee.append(k)
        #print('eeeeeeeeeeeeee',set(ee))

        exp=['experience','knowledge','idea','language','develped','involved','good','programme','technical','known']
        year=['year','month','years','months']
        words=['machine learning','artificial inteligence','java script','j query','neural networks','angular js','node js','react js']
        shortcuts={'ai':'artificialinteligence','ml':'machinelearning','nlp':'naturallanguageprocessing'}
        exp1=[]
        year1=[]
        technologies1=[]
        for k in exp:
            exp1.append(porter.stem(k))
        for k in year:
            year1.append(porter.stem(k))
        for k in technologies:
            technologies1.append(porter.stem(k))
        files=[]
        final={}
        final4={}
        for file in resume_all_names: #glob.glob("/Users/pradeep/Documents/jupyter/profilesforproject/1.docx"):
            result={}
            result3={}
            data1 = ResumeParser(file).get_extracted_data()
            #print('##########################',data1)
            ll=[]
            lll=[]
            text = textract.process("/Users/pradeep/Documents/demo/"+file)
            #print(text)
            name=(file.split('/')[-1]).split('.')[0]
            text = text.decode("utf-8")

            text=text.replace(',', ' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('+',' ').replace('|',' ')
        #print(text[15])
            my_text=text.lower().split('.\n')
            
            final3[name]=text
                  #print(text)
            ds3.append(text)
        #print(my_text[15])
            ll3=[]
            for i in my_text:
             kk=i.split()
             kk1=[]
             for value,term in enumerate(kk):
                  if term.upper() in ee:
                     print('jbjhjhjhj',term,'\n')
                     edu_matching.append((name ,term.upper()))
                  if value+2<len(kk):
                      if kk[value]+' '+kk[value+1] in words:
                        term=kk[value]+''+kk[value+1]
                        #print('kiiiiiiiiiiiii',term)
                  if term in shortcuts.keys():
                   kk1.append(porter.stem(shortcuts[term]))
                  else:
                   kk1.append(porter.stem(term))

                        
             ll3.append(kk1)
            #print('lllllllllllllllll',ll3)
            for ll2 in ll3:
             #print('ssssss',ll2)
             for k,j in enumerate(ll2):
                   #print('ooooo',j)
                   for l in skills.keys():
                    #print('oooooooooo',skills.keys())
                    if j==l:
                        flag1='TRUE'
                        flag2='TRUE'
                        front=ll2[:k+1]
                        rare=ll2[k+1:]
                        #print('kkkkkkkkkkk',front,rare)
                        for k1,k2 in enumerate(front[::-1]):
                            if k2=='year' and flag1=='TRUE':
                              #print(j,front[-k1-2],front[-k1-1])
                              lll.append((j,front[-k1-2]+' '+front[-k1-1]))
                              ll.append((j,int(float(front[-k1-2])*12)))
                              flag1='FLASE'
                            if k2=='month' and flag1=='TRUE':
                                #print(j,front[-k1-2],front[-k1-1])
                                lll.append((j,front[-k1-2]+' '+front[-k1-1]))
                                ll.append((j,int(float(front[-k1-2])*1)))
                                flag1='FLASE'
                                
                            if k2 in exp1 and flag2=='TRUE':
                              #print(j,front[front.index(k)-1],front[front.index(k)])
                              lll.append((j,'knowledge'))
                              #print(j,front[-k1-2])
                              ll.append((j,8))
                              flag2='FLASE'

                        for r,k4 in enumerate(rare):
                            #print('k',k)
                            if k4=='year' and flag1=='TRUE':
                              #print(j,rare[r])
                              lll.append((j,rare[r-1]+' '+rare[r]))
                              ll.append((j,int(float(rare[r-1])*12)))
                              flag1='FLASE'
                            if k4=='month' and flag1=='TRUE':
                              #print(j,rare[r])
                              ll.append((j,rare[r-1]+' '+rare[r]))
                              ll.append((j,int(float(rare[r-1])*1)))
                              flag1='FLASE'
                            if k4 in exp1 and flag2=='TRUE':
                              #print(j,front[front.index(k)-1],front[front.index(k)])
                              #print(j,rare[r-1])
                              lll.append((j,'knowledge'))
                              ll.append((j,8))
                              flag2='FLASE'
                              
                              
                                                             
            #print('jjjjjjjjjjjjjj',ll)
            for key,value in lll:
                if key not in result3.keys():
                    result3[key]=[value]
                else:
                    rr=result3[key]
                    rr.append(value)
                    result3[key]=list(set(rr))

            final4[name]=result3
            df=pd.DataFrame.from_dict(final4)
            final5 = df.replace(np.nan, 'Not Trained', regex=True).to_dict()
            result7=next(iter(final5.values()))
            result8=[]
            for i,v in result7.items():
                result8.append(i)
            #print(result8)

            
            #print('hellllloooooo',final5)
            for key,value in ll:
                if key not in result.keys():
                    result[key]=value
                else:
                    rr=result[key]
                    if rr > value:
                        pass
                    else:
                        result[key]=value
            #print(result)
            #result['name']=data1['name']
            #result['email']=data1['email']
            #result['mobile_number']=data1['mobile_number']
            final[name]=result
           
    final1=pd.DataFrame.from_dict(final)
    if radio_exp == 'and':
        final1 = final1.T
        final1 = final1.dropna()
        final1 = final1.T
    
    #print('mmmmm',final1)
    #print('kjjjgjhgjhhfhghg',final1.dropna(axis = 0, how = 'all', inplace = True))
    #print('hkgggghhgjhghhgjhghgj',modDf,final1)
    final2=final1.replace(np.nan, 'Not Trained', regex=True).to_dict()
    result2=next(iter(final2.values()))
    result6=[]
    for i,v in result2.items():
         result6.append(i)
    #print(result6)
    hello6={}
    for i,v in final.items():
        hello2={}
        for j,k in v.items():
           hello2[j]=1
        hello6[i]=hello2
    df=pd.DataFrame.from_dict(hello6)
    if radio_rating == 'and':
        df = df.T
        df = df.dropna()
        df = df.T
    df=df.replace(numpy.nan, 0 , regex=True)
    df = df.to_dict()
    hello3={}
    rating_result={}
    for i,v in df.items():
        ll=0
        for j,k in v.items() :
            if j in skills.keys():
             ll=ll+(k*int(skills[j]))
        hello3[i]=ll
    rating_result=sorted(hello3.items(), key=lambda x: x[1], reverse=True)
            
    # sorting dataframe based on experience nearest values
    sort_exp=pd.DataFrame.from_dict(final,orient='index')
    if radio_exp == 'and':
        sort_exp = sort_exp.dropna()
        
    for (k,v) in experiences:
        if k in sort_exp.columns.values:
            
            if v=='knowledge':
                sort_exp[k]=(sort_exp[k]-8).abs()
                
            else:
                sort_exp[k]=(sort_exp[k]-(float(v)*12)).abs()
                #sort_exp[1].fillna(0, inplace=True)
               # sort_exp.fillna(sort_exp.max())

                        
            #sort_exp[]
            #print(rating_result)
            #print('education',edu_matching,experiences)
    sort_exp=sort_exp.fillna(sort_exp.max()+10, downcast='infer')
    sort_exp["sum"] = sort_exp.sum(axis=1)
    sort_exp=sort_exp.sort_values('sum')
    exp_final=sort_exp.index.values
    print(sort_exp)
    return render_template('layer3.html',rating_result = rating_result,edu_matching=edu_matching,experiences=experiences,exp_final=exp_final,result=result2,final=final2)

    
    
    
    
    
@app.route('/multiple_file_upload', methods=['POST'])
def upload_file1():
    job_description_filename=""
    if request.method == 'POST':
        job_description_files = request.files.getlist('job_description_file')
        print(job_description_files)
        for job_description_file in job_description_files:
            if job_description_files and allowed_file(job_description_file.filename):
                job_description_filename = secure_filename(job_description_file.filename)
    EDUCATION = [
         'B.E.', 'B.E', 'BS', 'B.S',
         'M.E', 'M.E.', 'MS', 'M.S','BA','B.A','B.A.','BSC',
        'B-TECH', 'B.TECH','BTECH','M-TECH', 'M.TECH', 'MTECH','B.SC','B.SC.',
        'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII','MCA','MBA','BACHELORS','MASTERS','POST_GRADUATION','UNDER_GRADUATION','B.TECH.','M.TECH.'
    ]
    graduation=['POST_GRADUATION','UNDER_GRADUATION',]
    
    edu_abb={'MCA':['MCA','M.C.A','M.C.A.'],'BACHELORS_Tech':['B-TECH', 'B.TECH','BTECH','B.TECH.'],'MASTERS_Tech':['M-TECH', 'M.TECH','MTECH','M.TECH.'],'BACHELORS_Science':['BSC','B.SC','B.SC.'] }
    technologies=['java','python','.net','jquery','c#.net','mvc','jquery','javascript','python','flask','sql',
                  'asp.net','c#','azure','aws','agile','react','mysql','angular','kendo','machinelearning','artificialinteligence','ajax','vb.net','ado.net','neuralnetworks']
    for file in glob.glob("/Users/pradeep/Documents/demo/templates/"+job_description_filename):
        text = textract.process(file)
        text = text.decode("utf-8")

        text=text.replace(',', ' ').replace('\n', ' ').replace('\xa0', ' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('+',' ').replace('|',' ').replace('/',' ')
        my_text=text.lower().split(' ')
        hi=Counter(my_text)
    #print(hi,type(hi))
    #print(my_text)
    edu=[]
    for value in my_text:
        for k,v in edu_abb.items():
            for term in v:
                if value.upper()==term:
                    print('kkkkkkkkkkkkkkkkkk',value)
                    edu.append(k)
                    
    print('jnkkjkkjkj', edu)
           
    ll={}
    list1=[]
    for i,v in enumerate(my_text):
          if my_text[i]=="developer" and my_text[i-1] in technologies:
              list1.append(my_text[i-1])
    list1=list(set(list1))
    #print(list1)
    for k,v in hi.items():
        if k in technologies:
            ll[k]=v
    print(ll)
    exp={}
    for k,v in ll.items():
      exp[k]='knowledge'
      
    final5={}
    low=1
    min=[2,3]
    high=[4,5,6]
    hello=[1,2,3,4,5,6]
    for k,v in ll.items():
        if k in list1:
            final5[k]=5
        if v in high:
            final5[k]=4
        if v in min:
            final5[k]=2
        if v==1:
            final5[k]=1
        if v not in hello:
            final5[k]=5
        if k in list1:
            final5[k]=5
    #print(final5)
    list1=[1,2,3,4,5,6,7,8,9]
    list2=['knowledge',1,2,3,4,5,6,7,8,9]
    edu1=[]
    for i in edu_abb.keys():
        edu1.append(i)
    for i in graduation:
        edu1.append(i)
    print('edu1',edu1)

    return render_template('dropdown.html',edu_abb=edu1,edu=edu,final = final5,list1=list1,list2=list2,exp=exp)





@app.route('/multiple_file_upload1', methods=['POST'])
def upload_file():
    ds3=[]
    final3={}
    main3={}
    job_description_filename=""
    if request.method == 'POST':
        resume_all_names=[]
        resume_files = request.files.getlist('resume_files[]')

        job_description_files = request.files.getlist('job_description_file')
        for job_description_file in job_description_files:
            if job_description_files and allowed_file(job_description_file.filename):
                job_description_filename = secure_filename(job_description_file.filename)
        for resume_files1 in resume_files:
            if resume_files1 and allowed_file(resume_files1.filename):
                resume_files1 = secure_filename(resume_files1.filename)
                resume_all_names.append(resume_files1)


    #print(resume_files,'helllllll',job_description_filename,resume_all_names,'jellllllll')
    str1='i have 2 years off experience in python and 4 years of experience in java'
    str2='i have 3 years of experience in .net and jquery'
    str3='i have high knowledge in python and basic knowledge in java'
    str4='i have basic knowledge in c#.net but no knowledge in mvc'
    technologies=['java','python','.net','jquery','c#.net','mvc','jquery','javascript','python','flask','sql','asp.net','c#','django','pyramid','sqlite3','html5','machinelearning','artificialinteligence','javascript','jquery','neuralnetworks','angularjs','angular2','nodejs','reactjs']

    exp=['experience','knowledge','idea','language','develped','involved','good','programme','technical','known']
    year=['year','month','years','months']
    words=['machine learning','artificial inteligence','java script','j query','neural networks','angular js','node js','react js']
    shortcuts={'ai':'artificialinteligence','ml':'machinelearning','nlp':'naturallanguageprocessing'}
    exp1=[]
    year1=[]
    technologies1=[]
    for k in exp:
        exp1.append(porter.stem(k))
    for k in year:
        year1.append(porter.stem(k))
    for k in technologies:
        technologies1.append(porter.stem(k))
    #print('exp  :-',exp1,'\n','yeras :-',year1,'\n','technologies :-',technologies1,'\n\n\n')
    ll2=str4.split(' ')
    files=[]
    final={}
    final4={}
    job=textract.process("/Users/pradeep/Documents/demo/templates/"+job_description_filename)
    job = job.decode("utf-8")
    job=job.replace(',', ' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('+',' ')
    job1=[]
    job2=[]
    job=job.lower().split(' ')
    
    for value,term in enumerate(job):
         #print(value,term)
         if value+2<len(job):
             if job[value]+' '+job[value+1] in words:
               term=job[value]+''+job[value+1]
               #print('kiiiiiiiiiiiii',term)
         if term in shortcuts.keys():
          job2.append(porter.stem(shortcuts[term]))
         else:
          job2.append(porter.stem(term))
          
          
    for i in job2:
        if i in technologies1:
          job1.append(i)
    #print('kkkkkkkkkkkkkkkkk',job2,'kkkkkkkkkkkkkkllllllllllll',job1)
    for file in resume_all_names: #glob.glob("/Users/pradeep/Documents/jupyter/profilesforproject/1.docx"):
        result={}
        result3={}
        data1 = ResumeParser(file).get_extracted_data()
        #print('##########################',data1)
        ll=[]
        lll=[]
        text = textract.process("/Users/pradeep/Documents/demo/"+file)
        #print(text)
        name=(file.split('/')[-1]).split('.')[0]
        text = text.decode("utf-8")

        text=text.replace(',', ' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('+',' ').replace('|',' ')
    #print(text[15])
        my_text=text.lower().split('.\n')
        
        final3[name]=text
              #print(text)
        ds3.append(text)
    #print(my_text[15])
        ll3=[]
        for i in my_text:
         kk=i.split()
         kk1=[]
         for value,term in enumerate(kk):
              if value+2<len(kk):
                  if kk[value]+' '+kk[value+1] in words:
                    term=kk[value]+''+kk[value+1]
                    #print('kiiiiiiiiiiiii',term)
              if term in shortcuts.keys():
               kk1.append(porter.stem(shortcuts[term]))
              else:
               kk1.append(porter.stem(term))

                    
         ll3.append(kk1)
            
        for ll2 in ll3:
         #print('ssssss',ll2)
         for k,j in enumerate(ll2):
               #print('ooooo',j)
               for l in job1:
                #print('oooooooooo',k,j,l)
                if j==l:
                    flag1='TRUE'
                    flag2='TRUE'
                    front=ll2[:k+1]
                    rare=ll2[k+1:]
                    #print('kkkkkkkkkkk',front,rare)
                    for k1,k2 in enumerate(front[::-1]):
                        if k2=='year' and flag1=='TRUE':
                          #print(j,front[-k1-2],front[-k1-1])
                          lll.append((j,front[-k1-2]+' '+front[-k1-1]))
                          ll.append((j,int(float(front[-k1-2])*12)))
                          flag1='FLASE'
                        if k2=='month' and flag1=='TRUE':
                            #print(j,front[-k1-2],front[-k1-1])
                            lll.append((j,front[-k1-2]+' '+front[-k1-1]))
                            ll.append((j,int(float(front[-k1-2])*1)))
                            flag1='FLASE'
                            
                        if k2 in exp1 and flag2=='TRUE':
                          #print(j,front[front.index(k)-1],front[front.index(k)])
                          lll.append((j,'knowledge'))
                          #print(j,front[-k1-2])
                          ll.append((j,8))
                          flag2='FLASE'

                    for r,k4 in enumerate(rare):
                        #print('k',k)
                        if k4=='year' and flag1=='TRUE':
                          #print(j,rare[r])
                          lll.append((j,rare[r-1]+' '+rare[r]))
                          ll.append((j,int(float(rare[r-1])*12)))
                          flag1='FLASE'
                        if k4=='month' and flag1=='TRUE':
                          #print(j,rare[r])
                          ll.append((j,rare[r-1]+' '+rare[r]))
                          ll.append((j,int(float(rare[r-1])*1)))
                          flag1='FLASE'
                        if k4 in exp1 and flag2=='TRUE':
                          #print(j,front[front.index(k)-1],front[front.index(k)])
                          #print(j,rare[r-1])
                          lll.append((j,'knowledge'))
                          ll.append((j,8))
                          flag2='FLASE'
                                                         
        #print('jjjjjjjjjjjjjj',lll)
        for key,value in lll:
            if key not in result3.keys():
                result3[key]=[value]
            else:
                rr=result3[key]
                rr.append(value)
                result3[key]=list(set(rr))

        final4[name]=result3
        #result3['email']=data1['email']
        #result3['mobile_number']=data1['mobile_number']
        final4[name]=result3
        df=pd.DataFrame.from_dict(final4)
        final5 = df.replace(np.nan, 'Not Trained', regex=True).to_dict()
        result7=next(iter(final5.values()))
        result8=[]
        for i,v in result7.items():
            result8.append(i)
        #print(result8)

        
        #print('hellllloooooo',final5)
        for key,value in ll:
            if key not in result.keys():
                result[key]=[value]
            else:
                rr=result[key]
                if rr[0] > value:
                    pass
                else:
                    result[key]=[value]
        #print(result)
        #result['name']=data1['name']
        #result['email']=data1['email']
        #result['mobile_number']=data1['mobile_number']
        final[name]=result
       
        final1=pd.DataFrame.from_dict(final)
        final2=final1.replace(np.nan, 'Not Trained', regex=True).to_dict()
        result2=next(iter(final2.values()))
        result6=[]
        for i,v in result2.items():
             result6.append(i)
        #print(result2)
        
        
     ##### Tomorrow work on this percentages


    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english')

    description3=' '.join(job1)
    final3['description']=description3

    ds3.append(description3)
    percentage=[]
    for i,v in enumerate(final3.keys()):
        main3[i]=v
    tfidf_matrix = tf.fit_transform(final3.values())
    #print(tf.get_feature_names())
    cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix)
    #print(cosine_similarities[0][::-1])
    for i,v in enumerate(cosine_similarities[0]):
        if v > 0 and v < 0.99:
          percentage.append((main3[i],round(v*100,3)))
          
    #print(percentage)
          
    

    return render_template('dropdown.html',final = final2,percentage=percentage,result=result2,final4=final5,result8=result8)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8082)
