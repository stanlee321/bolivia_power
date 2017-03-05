import tensorflow as tf
#from libs import batch_norm
#from libs import data_pros

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


learning_rate = 0.001
batch_size = 1000
#n_iterations,\
total_batches = int(X_train.shape[0]/batch_size) 
n_neurons = 500
n_layers = 4
activation_fn = tf.nn.relu 
final_activation_fn = tf.nn.softmax
cost_type = 'l2_norm'
hm_epochs = 1000

logs_path = '/notebooks/DeepLearningTasks/proyect/tensorboard_logs'

# Linear model

def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    h, W : tf.Tensor, tf.Tensor
        Output of fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W








def build_model(Xs,ys,n_neurons, n_layers, activation_fn, final_activation_fn, cost_type):


	xs = Xs
	ys = ys



	n_xs = xs.shape[1]
	n_ys = ys.shape[1]


	X = tf.placeholder(name='X', shape=[None, n_xs], dtype=tf.float32)
	Y = tf.placeholder(name='Y', shape=[None, n_ys], dtype=tf.float32)


	current_input = X

	for layer_i in range(n_layers):
		current_input = linear(current_input, n_neurons,\
		 activation=activation_fn, name='layer{}'.format(layer_i))[0]

	Y_pred = linear(current_input, n_ys, activation=final_activation_fn, name='prediction')[0]
	
	if cost_type == 'l1_norm':
		cost = tf.reduce_mean(tf.reduce_sum(tf.abs(Y - Y_pred), 1))
	elif cost_type == 'l2_norm':
		cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(Y,Y_pred), 1))
	else:
		raise ValuError("Unkonw value, chosse between l1 or l2")

	saver = tf.train.Saver()

	tf_log = 'tf_log'

	return {'X': X,'Y': Y, 'Y_pred': Y_pred, 'cost': cost, 'saver': saver, 'tf_log': tf_log}
	#return {'X': X,'Y': Y, 'Y_pred': Y_pred, 'cost': cost}


def train_nn(X,y,X_cv,y_cv,X_test,y_test,\
		learning_rate,\
		n_neurons,\
		n_layers,\
		activation_fn,\
		final_activation_fn,\
		cost_type,\
		batch_size,\
		total_batches,\
		hm_epochs):
	


	Xs = np.array(X).reshape(-1,X.shape[1])
	ys = np.array(y).reshape(-1,y.shape[1])

	Xs_cv = np.array(X_cv).reshape(-1,X_cv.shape[1])
	ys_cv = np.array(y_cv).reshape(-1,y_cv.shape[1])

	Xs_test = np.array(X_test).reshape(-1,X_test.shape[1])
	ys_test = np.array(y_test).reshape(-1,y_test.shape[1])





	model = build_model(Xs, ys, n_neurons, n_layers,\
		activation_fn, final_activation_fn, cost_type)
	with tf.name_scope('Model'):
		prediction = model['Y_pred']
	with tf.name_scope('Loss'):
		cost = model['cost']
	with tf.name_scope('SGD'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	with tf.name_scope('Accuracy'):
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model['Y'],1))
		acc = tf.reduce_mean(tf.cast(correct, 'float'))
	# Tensorboard
	tf.scalar_summary('loss',cost)
	tf.scalar_summary('accuracy', acc)

	merged_summary_op = tf.merge_all_summaries()

	saver = model['saver']
	tf_log = model['tf_log']

	
	
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		summary_writer = tf.train.SummaryWriter(logs_path, graph = tf.get_default_graph())

		try:
			epoch = int(open(tf_log,'r').read().split('\n')[-2]) +1
			print('STARTING:', epoch)
		except:
			epoch = 1


		for epoch in range(hm_epochs):
			if epoch != 1:
				saver.restore(sess,"/notebooks/DeepLearningTasks/proyect/model.ckpt")


			epoch_loss = 0

			i = 0

			while i < (Xs.shape[0]):
				start = i
				end = i + batch_size
				batch_x = np.array(Xs[start:end])
				batch_y = np.array(ys[start:end])


				_, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={model['X']: batch_x, model['Y'] : batch_y})
				summary_writer.add_summary(summary, epoch * total_batches + 1)
				#epoch_loss += c
				epoch_loss = c / total_batches

				i +=batch_size
			saver.save(sess, "model.ckpt")

			with open(tf_log, 'a') as f:
				f.write(str(epoch)+'\n')

			print('Epoch', epoch + 1, 'completed out of ', hm_epochs,'loss:', epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model['Y'],1))
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Accuracy: ', accuracy.eval({model['X']:Xs_test, model['Y']:ys_test}))
		print("Run teh comand: \n" \
			"--> tensorboard -- logdir=/tmp/tensorflow_logs"\
			"\nThen open http://0.0.0.0:6666/ into your web browser")


train_nn(X_train,y_train,X_cv,y_cv,X_test,y_test,\
		learning_rate,\
		n_neurons,\
		n_layers,\
		activation_fn,\
		final_activation_fn,\
		cost_type,\
		batch_size,\
		total_batches,\
		hm_epochs)





