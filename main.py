import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from tkinter import *

def welcome():
    print("Welcome in Diabetes or Non Diabetes predication system")
    print("Press ENTER key to proceed")
    input()

def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for csv_file in content_list:
        if csv_file.split('.')[-1]=='csv':
            csv_files.append(csv_file)
    if len(csv_files)==0:
        return 'No csv file in this directory'
    else:
        return csv_files

def display_and_select_csv(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'.......',file_name)
        i+=1
    return  csv_files[int(input("Select your File ..."))]     

def main():
    welcome()
    csv_files=checkcsv()
    try:
        if csv_files=='No csv file in this directory':
            raise FileNotFoundError("No csv file in this directory")
        csv_file=display_and_select_csv(csv_files)
        print(csv_file,'....csv file is selected')
        print("Reading csv file") 
        print("Creating dataset")
        dataset=pd.read_csv(csv_file)
        print("Dataset is created")
        print(dataset.head())  #display first 5 data
        print(dataset.shape)   #No of rows and colums
        print(dataset.describe()) #getting the stastical measure of data
        print(dataset['Outcome'].value_counts())  #0--> Non diabetes and 1-->diabetes
        print(dataset.groupby('Outcome').mean())
        x=dataset.drop(columns='Outcome',axis=1)
        y=dataset['Outcome']
        scaler=StandardScaler()
        scaler.fit(x)
        standardized_data=scaler.transform(x)
        x=standardized_data

        s=float(input("Enter test data size(between 0 and 1)"))
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s,stratify=y,random_state=2)
        classifier=svm.SVC(kernel='linear')
        classifier.fit(x_train,y_train)
        predication=classifier.predict(x_test)
        accuracy=accuracy_score(predication,y_test)
        print("Press ENTER key to know accuracy score of our model")
        input()
        print("our model accuracy score is %2.2f%% "%(accuracy*100))
        print("Now you can predict Please ENTER medical information")
        input_data=tuple(map(float,input().split(',')))
        #input_data=(8,183,64,0,0,23.3,0.672,32)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
        std_data=scaler.transform(input_data_reshape)
        pred=classifier.predict(std_data)
        print(pred)

        root=Tk()
        if(pred==[1]):
          #print("The Object Is Rock")
          un=Label(root,text="The person is Diebetes",font=('algerian',25))
          un.grid(pady=25)
        else:
          un=Label(root,text="The person is Non-Diebetes",font=('algerian',25))
          un.grid(pady=25)
        
        
              
    except FileNotFoundError:
        print("No csv file in the directory")
        print("Press ENTER key to exit")
        exit()
        
if __name__=="__main__":
    main()
