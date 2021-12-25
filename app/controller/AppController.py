from flask import request, jsonify
from app import app
from app.module.Engine import datset,knn,knntest,tests
import pandas as pd
import os
from flask import render_template

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

@app.route('/')
def index():
    jumlah = len(datset()[0])
    dataset = pd.DataFrame(datset()[0])
    return render_template('index.html', ses = [dataset.to_html()], countDataset = jumlah)

@app.route('/proses', methods=['POST','GET'])
def proses():
    
    jumlah = len(datset()[1])
    a = pd.DataFrame(datset()[1])
    selection = pd.DataFrame(datset()[2])
    a = pd.DataFrame(datset()[3])
    train = datset()[4].sort_index(ascending=True)
    jumlahTraining = len(train)
    test = datset()[5]
    jumlahTesting = len(test)
    accuracy = datset()[6]
    matrix = datset()[7]
    
    return render_template('proses.html',
        countDataset = jumlah, 
        ses = [a.to_html()],
        selection = [selection.to_html()],
        a = [a.to_html()],
        train = [train.to_html()],
        test = [test.to_html(index=False)],
        accuracy = accuracy,
        matrix = matrix,
        jumlahTraining = jumlahTraining,
        jumlahTesting = jumlahTesting,
    )
@app.route('/test', methods=['POST','GET'])
def test():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = pd.read_excel(file)
        d = pd.DataFrame(data[data['Klasifikasi'].isna()])
        a = d
        b = pd.DataFrame(data.dropna())
        knn=KNeighborsClassifier(n_neighbors=3)
        x_train = b[['minimal stok','sisa stok']]
        y_train = b[['Klasifikasi']]
        x_test = a[['minimal stok','sisa stok']]
        a.loc[a['sisa stok'] <= a['minimal stok'],'Klasifikasi'] = 'Order'
        a.loc[a['sisa stok'] > a['minimal stok'],'Klasifikasi'] = 'Tidak Order'
        y_test = a['Klasifikasi']
        knn.fit(x_train,y_train.values.ravel())
        y_pred = knn.predict(x_test)
        knn.predict_proba(x_test)

        accuracy= accuracy_score(y_test, y_pred)
        mt = confusion_matrix(y_test, y_pred)
        """df_dropna = data.dropna(axis=0)
        a = df_dropna.drop(['No.','Nama Obat','Harga','Diskon ','Pajak','Jumlah'], axis=1)
        # Tahap aasi mencari sisa = total - quantity
        a = a
        a['sisa']= a['Total Stok']-a[' Quantity']
        a.loc[a['sisa'] <= a['Stok Minimal'],'Keterangan'] = 'Order'
        a.loc[a['sisa'] > a['Stok Minimal'],'Keterangan'] = 'Tidak Order'
        a = a.drop(['Total Stok',' Quantity'], axis=1)
        #menentukan nilai X dan Y untuk data test dan train KNN
        x = a[['Stok Minimal','sisa']]
        y = a.drop(['Stok Minimal','sisa'],axis=1)
        xt = tests()[0]
        yt = tests()[1]
        x_train = knn(xt,yt)[1]
        y_train = knn(xt,yt)[3]
        test = knntest(x,y,x_train,y_train)[2]
        a = knntest(x,y,x_train,y_train)[0]
        mt = knntest(x,y,x_train,y_train)[1]"""
        return render_template('testing.html', 
            data=data.to_html(index=False),
            d = d.to_html(),
            b = b.to_html(),
            c = accuracy,
            #test = test.to_html(index=False),
            #a = a,
            mt = mt
        )
    return render_template('testing.html', data = '')