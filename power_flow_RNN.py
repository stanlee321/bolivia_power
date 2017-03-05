import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import pandas as pd
import numpy as np



# Loading the data from our  dataset

#dirname = ...
#dirname_1 = "/home/stanlee321/Desktop/SIN_TODO/newton/1.0/data.csv"
#dirname_2 = "/home/stanlee321/Desktop/SIN_TODO/newton/1.0/newton.csv"

dirname_1 = "/notebooks/Power/newton/1.0/data.csv"
dirname_2 = "/notebooks/Power/newton/1.0/newton.csv"

busd   = pd.read_csv(dirname_1)
solbus = pd.read_csv(dirname_2)
busd.fillna(0.0)
solbus.fillna(0.0)

busd["Bus2"] = busd['Bus']*busd['Bus']
busd["Type2"] = busd['Type']*busd['Type']
busd["Vsp2"] = busd['Vsp']*busd['Vsp']
busd["Pgi2"] = busd['Pgi']*busd['Pgi']
busd["Qgi2"] = busd['Qgi']*busd['Qgi']


busd["Qgi/Pgi"] = busd["Qgi"]/(busd["Pgi"]+1.0)


busd["Vsp/Pgi"] = busd["Vsp"]/(busd["Pgi"]+1.0)
busd["Vsp/Qgi"] = busd["Vsp"]/(busd["Qgi"]+1.0)
busd["Vsp/Pli"] = busd["Vsp"]/busd["Pli"]
busd["Vsp/Qli"] = busd["Vsp"]/busd["Qli"]


columns_busd = list(busd.columns.values)
columns_solbus = list(solbus.columns.values)

df1 = busd
df2 = solbus


# Pre-processing the data

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

# Function for Normalize the Datas and Split into Training and CV set...
def process_data(df1,df2,size,r_state):
    
    # Para normalizar los DataFrame, se extrae las columnas que no se quieren normalizar,
    # en nuestro caso para df1 es Index=[0,1] y para df2 es Index = [0]
    # Nombres de las columnas originales 
    columns_features = list(df1.columns.values)
    columns_labels   = list(df2.columns.values)
      
    
    
    
    
    ####Normalizando Busdata
    x = df1.values
    x_scaled = preprocessing.scale(x)
    df1_norm_all = pd.DataFrame(x_scaled)
   
    df1_norm_all = df1_norm_all.drop(df1_norm_all.columns[[3]], axis=1)
   
    # For Solution data set
    y = df2.values
    y_scaled = preprocessing.scale(y)
    df2_norm_all = pd.DataFrame(y_scaled)
    


    # Spliting
    X_train, X_test, y_train, y_test = train_test_split(df1_norm_all, df2_norm_all, test_size= size, random_state = r_state)
    
    X_cv, X_test, y_cv, y_test = train_test_split(X_test,y_test, test_size = 0.50, random_state = r_state)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_cv = np.array(X_cv)
    y_cv = np.array(y_cv)
    
    return (X_train, y_train, X_test, y_test, X_cv, y_cv, df1_norm_all,df2_norm_all)



X_train, y_train, X_test, y_test, X_cv, y_cv, df1_norm_all, df2_norm_all\
 = process_data(df1,df2, 0.30, r_state = None )




hm_epochs = 3
n_classes = y_train.shape[1]
batch_size = 128
chunk_size = 57
n_chunks = 57
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y,prediction), 1))
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < X_train.shape[0]/batch_size:

                start = i
                end = i + batch_size
               


                epoch_x, epoch_y = np.array(X_train[start:end]) ,  np.array(y_train[start:end])
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:X_test.reshape((-1, n_chunks, chunk_size)), y:y_test}))

train_neural_network(x)
