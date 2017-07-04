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
h_dim = 128
fc_dim = 1024
lam = 10
n_disc = 5
lr = 1e-4

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


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

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

####
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
####

X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_b1 = bias_variable(h_dim)
D_b2 = bias_variable(h_dim*2)
D_b3 = bias_variable(h_dim*4)

D_w1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_w2 = tf.Variable(xavier_init([h_dim, h_dim*2]))
D_w2 = tf.Variable(xavier_init([h_dim*2, h_dim*4]))

theta_D = [D_b1, D_b2, D_b3, D_w1, D_w2, D_w3]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

# HERE
def D(X): 
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    D_h1 = lrelu(ly.batch_norm(conv2d(X, D_w1)) + D_b1)
    D_h2 = lrelu(ly.batch_norm(conv2d(D_h1, D_w2)) + D_b2)
    D_h3 = lrelu(ly.batch_norm(conv2d(D_h2, D_w3)) + D_b3)
    out = linear(D_h3, 1)
    return tf.nn.sigmoid(out)


G_sample = G(z)
D_real = D(X)
D_fake = D(G_sample)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean(grad_norm - 1.)**2

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

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
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
