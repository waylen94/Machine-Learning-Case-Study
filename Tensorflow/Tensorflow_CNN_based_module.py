import tensorflow as tf
import numpy as np

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
    
    def add_layer():
        
    return None

if __name__ == '__main__':
    #tensorflow_basis_20190923()
    
#     tensorflow_basis_assign_20190923()
    tensorflow_basis_architecture()
 