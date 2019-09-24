import tensorflow as tf
import numpy as np

#visualization part need matplot
import matplotlib.pyplot as plt

#Import classification training data
from tensorflow.examples.tutorials.mnist import input_data

#advanced_overfitting function needed testing datasets from sklearn

#Basis consist of gradient descent algorithm to optimizer the loss (pre - real)
def tensorflow_basis():
    # create data
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3 # standard data will be compared with the prediction one
    
    
    # create tensorflow structure start
    weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))
    biases = tf.Variable(tf.zeros([1]))
    
    # operation predication model start
    y = weights * x_data + biases
    
    #optimization weights starting comparison
    loss = tf.reduce_mean(tf.square(y - y_data))
    
    
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
    train  = optimizer.minimize(loss)
    
    # create ensor flow structure end
    
    sess = tf.compat.v1.Session()
    
    init = tf.compat.v1.global_variables_initializer()
    
    sess.run(init)
    
    for step in range(201):
        sess.run(train)
        
        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))
            
#Basis variable assignment 
def tensorflow_basis_assign():    
    
    state  = tf.Variable(0)
    
    one = tf.constant(1)
    
    new_value = tf.add(state, one)
    
    update = tf.compat.v1.assign(state, new_value)
    
    init = tf.compat.v1.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(5):
            sess.run(update)
            print(sess.run(state))
    
    
    return None

# layer


def tensorflow_basis_architecture():
    '''
        add one more layer and return the output of this layer
    '''
    def add_layer(inputs, in_size, out_size, activation_function=None):
        
        weights = tf.Variable(tf.random.normal([in_size,out_size]))
        biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        
        Wx_plus_b = tf.matmul(inputs, weights) + biases
        
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            
        return outputs
    
    
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]   # simulated with some real data
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1])
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1])
    
    # add hidden layer
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    
    # add output layer
    prediction = add_layer(l1, 10, 1, activation_function = None)
    
    #loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    init = tf.compat.v1.global_variables_initializer()
    
    sess = tf.compat.v1.Session()
    sess.run(init)
    
    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        
        if i % 50 ==0:
            
            print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
  
    return None


def tensorflow_basis_architecture_visualization():
    def add_layer(inputs, in_size, out_size, activation_function):
        weights = tf.Variable(tf.random.normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])+0.1)
        Wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    
    #display something
    
    
    #define placeholder for inputs to network
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1])
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1])
    
    l1 = add_layer(xs, 1, 10, activation_function= tf.nn.relu)
    prediction = add_layer(l1, 10, 1, None)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    sess = tf.compat.v1.Session()
    
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data)
    plt.ion()
    plt.show()
    
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs:x_data,ys:y_data})
            
            lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
            
            plt.pause(1)
                
    
    
    
    return None

def tensorflow_advanced_classification():
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    global prediction
    def add_layer(inputs, in_size, out_size, activation_function = None, ):
        weights = tf.Variable(tf.random.normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
        wx_plus_b = tf.matmul(inputs,weights) + biases
        
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        
        return outputs
    
    #only for indicate the accuracy after optimization
    def compute_accuracy(v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict = {xs: v_xs})
        correct_prediction = tf.equal(tf.math.argmax(y_pre, 1), tf.math.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict = {xs: v_xs, ys : v_ys})
        
        return result
    
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    
    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                                  reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.Session()
    
    init = tf.compat.v1.global_variables_initializer()
    
    sess.run(init)
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
        
    return None





def tensorflow_advanced_overfitting():
    digits = load_digits
    
    
    return None

def tensorflow_advanced_CNN():
    mnist = input_data.read_data_sets('MINST_data', one_hot = True)
    global prediction
    def compute_accuracy(v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict = {xs: v_xs, keep_prob:1})
        correct_prediction = tf.equal(tf.math.argmax(y_pre,1), tf.math.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys,keep_prob:1})
        
        return result
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape): 
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
    
    def conv2d(x,W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    xs = tf.placeholder(tf.float32, [None, 784])/255.
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    
    #conv1 layer
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    #conv2 layer
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    
    #fc1 layer
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #sovling over fitting
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc2 layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    #after deling with convolutionary layer optimization procedure
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 
                                                  reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    sess = tf.compat.v1.Session()
    
    init = tf.compat.v1.global_variables_initializer()
    
    sess.run(init)
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 ==0 :
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[0:1000]))
        
        
    
    
    
    
    
    
    
    return None

def tensorflow_davanced_RNN():
    tf.set_random_seed(1)
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    
    lr = 0.001
    training_iters = 100000
    batch_size = 128
    
    n_inputs = 28
    n_steps = 28
    n_hidden_units = 128
    n_classes = 10
    
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    
    Weights = {
            'in': tf.Variable(tf.random.normal([n_inputs, n_hidden_units])),
            'out': tf.Variable(tf.random.normal([n_hidden_units, n_classes]))

        }
    biases = {
            'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
            
            'out': tf.Variable(tf.constant(0.1, shape = [n_classes,]))
        }
    
    def RNN(X, weights, biases):
        X = tf.reshape(X, [-1, n_inputs])
        
        X_in = tf.matmul(X, weights['in']) + biases['in']
        
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
        
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
        
        init_state = cell.zero_state(batch_size, dtype = tf.float32)
        
        
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state = init_state, time_major=False)
        
        
        outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
        
        results = tf.matmul(outputs[-1], Weights['out'])+ biases['out']
        
        return results
    
    pred = RNN(x, Weights, biases)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with tf.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            
            sess.run([train_op],feed_dict = {
                    x:batch_xs,
                    y:batch_ys,
                })
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict ={
                        x: batch_xs,
                        y: batch_ys,
                    }))
            step += 1
    
    
    return None

if __name__ == '__main__':
#     tensorflow_basis()
    
#     tensorflow_basis_assign()
    tensorflow_basis_architecture()
#     tensorflow_basis_architecture_visualization()
#     tensorflow_advanced_classification() 

#     tensorflow_advanced_overfitting()

#     tensorflow_advanced_CNN()
#     tensorflow_davanced_RNN()
