## Importing libraries and functions
import streamlit as st
import pandas as pd
from model import predict_score_and_accuracy

## Initialize streamlit app
st.set_page_config('EvalBot')
st.title('Find Score:female-technologist:')

st.sidebar.subheader('Put the text files of actual answers below :')
st.sidebar.write('NOTE : Actual answers should be in same sequence as student answers')
answers=st.sidebar.file_uploader('**Each file should contain only one actual answer',
                                 type=['docx','pdf','txt'],
                                 accept_multiple_files=True
                              )  
      
st.subheader('Put the text files of answers given by student below :')
answers_by_student=st.file_uploader('**Each file should contain only one answer given by student',
                                    type=['docx','pdf','txt'],
                                    accept_multiple_files=True
                                 )

## Defining a function for storing answers in a dataframe
def dataframe(actuals,answers):
   df=pd.DataFrame()
   list1=[]
   list2=[]
   for i in range(len(answers)):
      if answers[i] is not None:
         content1=answers[i].read().decode('latin-1')
      list1.append(content1)
   for i in range(len(actuals)):
      if actuals[i] is not None:
         content2=actuals[i].read().decode('latin-1')
      list2.append(content2)
   df['Actual_Answers']=list2
   df['Student_Answers']=list1
   return df

## Store maximum marks, marks obtained and similarities/accuracies in respective dataframes
marks=pd.DataFrame()
max_marks=pd.DataFrame()
max_marks['Student File No.']=[i+1 for i in range(0,len(answers_by_student))]
max_marks['Maximum Marks']=[5 for i in range(0,len(answers_by_student))]
marks['Marks Obtained']=['-' for i in range(0,len(answers_by_student))]

button1_clicked=st.session_state.get('button1_clicked',False)
button1=st.button('Next',disabled=button1_clicked)
button2_clicked=st.session_state.get('button2_clicked',True)
button2=st.button('Find Score',disabled=button2_clicked)

if 'table' not in st.session_state:
   st.session_state.table=False
   
## if button1 clicked...
if button1:
   if len(answers)==len(answers_by_student):
      st.session_state.button1_clicked=True
      st.session_state.button2_clicked=False
      st.session_state.table=True
   else:
      st.write("Number of actual answers should be equal to that of student answers!")
if st.session_state.table:
   max_marks=st.data_editor(max_marks)## Output after clicking 'Next' button
   
## if button2 clicked...
if button2:
   for i in range(0,len(max_marks['Student File No.'])):
         marks['Marks Obtained'][i]=int(round(predict_score_and_accuracy(dataframe(answers,answers_by_student))[i]*max_marks['Maximum Marks'][i],0))
   max_marks['Marks Obtained']=marks['Marks Obtained']
   st.table(max_marks)## Output after clicking 'Find Score' button
   st.write('Total Marks Obtained : ',max_marks['Marks Obtained'].sum())## Print total marks obtained
   
