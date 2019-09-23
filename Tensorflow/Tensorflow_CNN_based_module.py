import tensorflow as tf
import numpy as np

#visualization part need matplot
import matplotlib.pyplot as plt

def tensorflow_basis_20190923():
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
            
def tensorflow_basis_assign_20190923():
    
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
        weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros(1, out_size)+0.1)
        Wx_plus_b = tf.matmul(iputs, weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = n.squara(x_data) - 0.5 + noise
    
    #display something
    
    
    #define placeholder for inputs to network
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1])
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1])
    
    l1 = add_layer(xs, 1, 10, activation_function= tf.nn.relu)
    prediction = add_layer(l1, 10, 1, None)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.square(ys-prediction),reduction_indices[1])))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    sess = tf.Session()
    
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    fig = plt.figure()
    ax = gig.add_subplot(1,1,1)
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
            
            lines = ax.plot
                
    
    
    
    return None
if __name__ == '__main__':
    #tensorflow_basis_20190923()
    
#     tensorflow_basis_assign_20190923()
#     tensorflow_basis_architecture()

    tensorflow_basis_architecture_visualization()
 