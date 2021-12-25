import pandas as pd
import string
import numpy as np
from scipy.linalg import svd 
from numpy import dot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def datset():
    #ambil database
    df = pd.read_excel('dataset.xlsx')
    # Membuang data kosong
    df_dropna = df.dropna(axis=0)
    # Seleksi Tabel yang digunakan
    selection = df_dropna.drop(['No.','Nama Obat','Harga','Diskon ','Pajak','Jumlah'], axis=1)
    a = df_dropna.drop(['No.','Nama Obat','Harga','Diskon ','Pajak','Jumlah'], axis=1)
    # Tahap Transformasi mencari sisa = total - quantity
    transform = a
    transform['sisa']= transform['Total Stok']-transform[' Quantity']
    transform.loc[transform['sisa'] <= transform['Stok Minimal'],'Keterangan'] = 'Order'
    transform.loc[transform['sisa'] > transform['Stok Minimal'],'Keterangan'] = 'Tidak Order'
    transform = transform.drop(['Total Stok',' Quantity'], axis=1)
    #menentukan nilai X dan Y untuk data test dan train KNN
    x = transform[['Stok Minimal','sisa']]
    y = transform.drop(['Stok Minimal','sisa'],axis=1)
    # Data Training
    x_train = knn(x,y)[1]
    y_train = knn(x,y)[3]
    # Nilai Akurasi Hasil KNN
    accuracy = knn(x,y)[4]
    # Hasil Matrik TF 
    matrix = knn(x,y)[5]

    # Data Training
    train = pd.merge(x_train,y_train, left_index=True,right_index=True)

    # Data Test
    test = knn(x,y)[6]
    no = pd.DataFrame(columns=['Obat'])
   


    return df,df_dropna,selection,transform,train,test,accuracy,matrix,no,x,y

#Fungsi Pengelolaan KNN
def knn(x,y):
    # Menentukan nilai K = 3
    knn=KNeighborsClassifier(n_neighbors=5)
    # Menentukan julah data training 70% dan testing 30%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=5)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    knn.fit(x_train,y_train.values.ravel())
    # Perhitungan data  testing dengan data training
    y_pred = knn.predict(x_test)
    knn.predict_proba(x_test)
    accuracy= accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    c=['Predict']
    panda_df = pd.DataFrame(y_pred,columns=c)
    a = x_test
    b = panda_df
    # Hasil Data Testing
    test = pd.merge(a,b)
    test = test.drop(['index_y'],axis = 1)
    test = test.rename(columns = {'index_x':'ID'})

    return x_test,x_train,y_pred,y_train,accuracy,matrix,test

#fungsi KNN Data uji baru
def knntest(x,y,x_train,y_train):
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x)
    knn.predict_proba(x)
    accuracy= accuracy_score(y, y_pred)
    matrix = confusion_matrix(y, y_pred)
    c=['Predict']
    panda_df = pd.DataFrame(y_pred,columns=c)
    a = x.reset_index()
    b = panda_df.reset_index()
    test = pd.merge(a,b, left_index=True,right_index=True)
    test = test.drop(['index_y'],axis = 1)
    return accuracy,matrix,test

def tests():
    df = pd.read_excel('test.xlsx')
    # Membuang data kosong
    df_dropna = df.dropna(axis=0)
    # Seleksi Tabel yang digunakan
    a = df_dropna.drop(['No.','Nama Obat','Harga','Diskon ','Pajak','Jumlah'], axis=1)
    # Tahap Transformasi mencari sisa = total - quantity
    transform = a
    transform['sisa']= transform['Total Stok']-transform[' Quantity']
    transform.loc[transform['sisa'] <= transform['Stok Minimal'],'Keterangan'] = 'Order'
    transform.loc[transform['sisa'] > transform['Stok Minimal'],'Keterangan'] = 'Tidak Order'
    transform = transform.drop(['Total Stok',' Quantity'], axis=1)
    #menentukan nilai X dan Y untuk data test dan train KNN
    x = transform[['Stok Minimal','sisa']]
    y = transform.drop(['Stok Minimal','sisa'],axis=1)
    return x,y

def tests1():
    return False