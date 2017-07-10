import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim = 784
z_dim = 10
#h_dim = 128
#fc_dim = 1024
lam = 10
n_disc = 5
lr = 1e-4

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
    
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

####
def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')
####

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

D_w1 = weight_variable([5, 5, 1, 32])
D_b1 = bias_variable([32])

D_w2 = weight_variable([5, 5, 32, 64])
D_b2 = bias_variable([64])

D_w3 = weight_variable([5, 5, 64, 128])
D_b3 = bias_variable([128])

D_w4 = weight_variable([28 ,28 , 128, 32])
D_b4 = bias_variable([32])

theta_D = [D_b1, D_b2, D_b3, D_b4, D_w1, D_w2, D_w3, D_w4]

G_w1 = weight_variable([z_dim, 7 * 7 * 32])
G_b1 = bias_variable([7 * 7 * 32])

G_w2 = weight_variable([4, 4, 64, 64])
G_b2 = bias_variable([32])

G_w3 = weight_variable([4, 4, 1, 32])
G_b3 = bias_variable([1])

#G_w4 = weight_variable([4, 4, 1, 16])
#G_b4 = bias_variable([1])
                    
theta_G = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3]#, G_w4, G_b4 ]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z):
    k1, k2, k4 =28, 28/2, 28/4
    G_h1 = tf.nn.relu(tf.matmul(z, G_w1) + G_b1)
    G_h1_reshape = tf.reshape(G_h1, [-1, 7, 7, 32])
    
    G_h2_output_shape = tf.stack([tf.shape(z)[0], 14, 14, 32])
    G_h2 = tf.nn.relu(deconv2d(G_h1_reshape, G_w2, G_h2_output_shape) + G_b2)
    
    G_h3_output_shape = tf.stack([tf.shape(z)[0], 28, 28, 1])
    G_h3 = tf.nn.tanh(deconv2d(G_h2, G_w3, G_h3_output_shape) + G_b3)
    
    #G_h4_output_shape = tf.stack([tf.shape(z)[0], 28, 28, 1])
    #G_h4 = tf.nn.tanh(deconv2d(G_h3, G_w4, G_h4_output_shape) + G_b4)
    
    return G_h3
    '''
        s2, s4, s8, s16 = \
                    OUTPUT_SIZE/2, OUTPUT_SIZE/4, OUTPUT_SIZE/8, OUTPUT_SIZE/16
    
        h1 = tf.reshape(fully_connected(z, GF*8*s16*s16, 'g_fc1'), 
                        [-1, s16, s16, GF*8], name = 'reshap')
        h1 = relu(batch_norm(h1, name = 'g_bn1', is_train = is_train))
        
        h2 = deconv2d(h1, [BATCH_SIZE, s8, s8, GF*4], name = 'g_deconv2d1')
        h2 = relu(batch_norm(h2, name = 'g_bn2', is_train = is_train))
        
        h3 = deconv2d(h2, [BATCH_SIZE, s4, s4, GF*2], name = 'g_deconv2d2')
        h3 = relu(batch_norm(h3, name = 'g_bn3', is_train = is_train))
        
        h4 = deconv2d(h3, [BATCH_SIZE, s2, s2, GF*1], name = 'g_deconv2d3')
        h4 = relu(batch_norm(h4, name = 'g_bn4', is_train = is_train))
        
        h5 = deconv2d(h4, [BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3], 
                      name = 'g_deconv2d4')    
        
        return tf.nn.tanh(h5)
        '''


def D(X): 
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    x_image = tf.reshape(X, [-1, 28, 28, 1])
    D_h1 = lrelu(ly.batch_norm(conv2d(x_image, D_w1)) + D_b1)
    D_h2 = lrelu(ly.batch_norm(conv2d(D_h1, D_w2)) + D_b2)
    D_h3 = lrelu(ly.batch_norm(conv2d(D_h2, D_w3)) + D_b3)

    out = tf.matmul(D_h3, D_w4) + D_b4
    return out
    '''if reuse:
        scope.reuse_variables()
    size = 64
    shared = tcl.conv2d(x, num_outputs=size, kernel_size=4,stride=2, activation_fn=lrelu) # bzx28x28x1 -> bzx14x14x64
    shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)# 7x7x128
    shared = tcl.flatten(shared)
    d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
    q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
    q = tcl.fully_connected(q, 10, activation_fn=None) # 10 classes
    return d, q'''
    
    


G_sample = G(z)
D_real = D(X)
D_fake = D(G_sample)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean(grad_norm - 1.)**2

#LSGAN
D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
G_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)

'''#WGAN
D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)
'''

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('WDCGANout/'):
    os.makedirs('WDCGANout/')

i = 0

for it in range(1000000):
    for _ in range(n_disc):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )

    if it % 1000 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('WDCGANout/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
