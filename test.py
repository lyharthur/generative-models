#data.py
import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

folder = '/data/projects/yhliuzzb/'
def get_img(img_path, crop_h, resize_h):
    img=scipy.misc.imread(img_path).astype(np.float) # mode L for grayscale
    #print(img.shape)
     
    #print(img.shape)
    #crop resize  Original Use
    crop_w = crop_h
    resize_h = resize_h
    resize_w = resize_h
    h, w = img.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    #cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])# cropp
    cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])# no cropp
    #print(cropped_image.shape)
    #img = cropped_image#for defect grayscale data 
    #img = np.dstack((cropped_image,cropped_image))[:,:,:1]
    if len(cropped_image.shape)==3:
        cropped_image = cropped_image[:,:,0]
    img = cropped_image.reshape((resize_h,resize_w,1))
    #print(img)
    #print(img.shape)
    return np.array(img)/255.0


class celebA():
    def __init__(self):
        datapath = prefix + 'celebA'
        self.z_dim = 100
        self.size = 64
        self.channel = 3
        self.data = glob(os.path.join(datapath, '*.jpg'))

        self.batch_count = 0

    def __call__(self,batch_size):
        batch_number = len(self.data)/batch_size
        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            self.batch_count = 0

        path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

        batch = [get_img(img_path, 128, self.size) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)
        '''
        print self.batch_count
        fig = self.data2fig(batch_imgs[:16,:,:])
        plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        '''
        return batch_imgs

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)
        return fig

class mydata():
    def __init__(self, size, defect, defect_num,test=False):

        data_list = []
        for i in range(defect_num):
            defect_sub = defect.split(',')[i]
            print(defect_sub)
            #datapath = '/data/projects/eyhsu/defect/images/PSD1_ASI/'+defect_sub+'/*.jpg'
            if test:
                datapath = '../defect/test'+defect_sub+'/*.jpg'
            else:
                datapath = '../defect/'+defect_sub+'/*.jpg'
            data_list.extend(glob(datapath))
        print(len(data_list))


        #datapath = folder+'data/'+defect+'/'
        self.z_dim = 512
        self.y_dim = defect_num
        self.size = size
        self.channel = 1 ##
        self.data = data_list
        #self.data = glob(os.path.join(datapath, '*.jpg'))
        #print(self.data)

        label = []
        check = []
        label_count = -1
        for path in self.data: 
            defect_id = path.split('/')[2]#7,5
            if defect_id not in check:
                check.append(defect_id)
                label_count+=1
            label.append(label_count)

        #print(label)  
        one_hot = np.zeros((len(label),self.y_dim))
        for i,val in enumerate(label):
            one_hot[i,val]=1
        self.label = one_hot
        self.batch_count = 0

    def __call__(self,batch_size):

        batch_number = len(self.data)/batch_size
        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            list_zip = list(zip(self.data,self.label))
            np.random.shuffle(list_zip)
            self.data, self.label = zip(*list_zip)           
            self.batch_count = 0

        path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        label_list = self.label[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        #print(path_list)

        batch = [get_img(img_path, self.size*3, self.size) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)
        
       

        '''
        print self.batch_count
        fig = self.data2fig(batch_imgs[:16,:,:])
        plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        '''
        return batch_imgs, label_list

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
   
        for i, sample in enumerate(samples):
            #print(sample)
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            #print(sample.shape)
            new_sample = np.concatenate((sample,sample),axis = 2)
            new_sample = np.concatenate((new_sample,sample),axis = 2)
            #print(new_sample.shape)
            sample = new_sample
            plt.imshow(sample)
        return fig

class mnist():
    def __init__(self, flag='conv', is_tanh = False):
        datapath = folder+'GAN_yhliu/MNIST_data'
        self.X_dim = 784 # for mlp
        self.z_dim = 100
        self.y_dim = 10
        self.size = 28 # for conv
        self.channel = 1 # for conv
        self.data = input_data.read_data_sets(datapath, one_hot=True)
        self.flag = flag
        self.is_tanh = is_tanh

    def __call__(self,batch_size):
        batch_imgs,y = self.data.train.next_batch(batch_size)
        if self.flag == 'conv':
            batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
        if self.is_tanh:
            batch_imgs = batch_imgs*2 - 1        
        return batch_imgs, y

    def data2fig(self, samples):
        if self.is_tanh:
            samples = (samples + 1)/2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
        return fig    

if __name__ == '__main__':
    data = face3D()
    print(data(17).shape)

#gan.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

from nets import *
from datas import *


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
 
def sample_y(m, n): # 16 , y_dim , fig count
    y = np.zeros([m,n])
    for i in range(m):
        y[i, i%n] = 1
    #y[:,7] = 1
    #y[-1,0] = 1
    #print(y)
    return y

def concat(z,y):
    return tf.concat([z,y],1)

class CGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data, loss_type):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data = data
        self.loss_type = loss_type

        self.lr = tf.placeholder(tf.float32)


        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.k = tf.placeholder(tf.float32)

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))

        self.D_real = self.discriminator(self.X, self.y)
        self.D_fake = self.discriminator(self.G_sample, self.y, reuse = True)
    
        self.C_real = self.classifier(self.X)
        self.C_fake = self.classifier(self.G_sample, reuse = True)

        self.lam = 10
        eps = tf.random_uniform([tf.shape(self.G_sample)[0], 1, 1, 1], minval=0., maxval=1.)#batch_size = 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = self.discriminator(self.X_inter, self.y, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        grad_pen = tf.reduce_mean(self.lam * tf.square(grad_norm- 1.))    
        # loss
        if self.loss_type == 'LS' :
            self.D_loss = 0.5 * (tf.reduce_mean((self.D_real - 1)**2) + tf.reduce_mean(self.D_fake**2))
            self.G_loss = 0.5 * tf.reduce_mean((self.D_fake - 1)**2)
        elif self.loss_type =='W' :
            self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake) + grad_pen
            self.G_loss = tf.reduce_mean(self.D_fake)

        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]

        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))  
        
        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(self.G_loss + self.C_fake_loss, var_list=self.generator.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(self.C_real_loss , var_list=self.classifier.vars)
        #self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_fake_loss, var_list=self.generator.vars)        

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train_classifier(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 64, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 2e-3

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    self.C_real_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                    )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_real_loss_curr = self.sess.run(
                        [self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b})
                print('Iter: {}; C_real_loss: {:.4}'.format(epoch,  C_real_loss_curr))

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)
     

    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 64, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 2e-4

        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 10 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, y_b = self.data(batch_size)
                self.sess.run(self.clip_D)
                self.sess.run(
                    [self.D_solver],
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim), self.lr: learning_rate}
                    )
            # update G
            for _ in range(1):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.y:y_b, self.z: sample_z(batch_size, self.z_dim), self.lr: learning_rate}
                )
            # update C
            for _ in range(1):
                # real label to train C
                self.sess.run(
                    self.C_real_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                    )
            
                # fake img label to train G
            '''for _ in range(3):
                self.sess.run(
                    self.C_fake_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})'''
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr, C_real_loss_curr = self.sess.run(
                        [self.D_loss, self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr, C_fake_loss_curr = self.sess.run(
                        [self.G_loss, self.C_fake_loss],
                        feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:}; C_fake_loss: {:}'.format(epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr))

                if epoch % 500 == 0:
                    y_s = sample_y(16, self.y_dim)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}.png'.format(sample_folder, str(fig_count).zfill(3)), bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                if self.loss_type == 'LS' :
                     self.saver.save(self.sess, ckpt_dir+'LS_GAN.ckpt', global_step=epoch)
                elif self.loss_type == 'W' :
                     self.saver.save(self.sess, ckpt_dir+'W_GAN.ckpt', global_step=epoch)
    def test(self, sample_folder, sample_num):
        
        y_s = sample_y(sample_num, self.y_dim)
        samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(sample_num, self.z_dim)})
        for i, sample in enumerate(samples):
            new_sample = np.concatenate((sample,sample),axis = 2)
            new_sample = np.concatenate((new_sample,sample),axis = 2)
            #fig = self.data.data2fig(samples)
            plt.imshow(new_sample)
            plt.axis('off')
            plt.savefig('{}/{}_{}.png'.format(sample_folder, i%self.y_dim, str(i).zfill(3)), bbox_inches='tight')
            plt.close()



    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class CBEGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data = data
        self.lr = 2e-4

        self.lam = 1e-3
        self.gamma = 0.5
        self.k_curr = 0.0

        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.k = tf.placeholder(tf.float32)

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))

        self.D_real = self.discriminator(self.X, self.y)
        self.D_fake = self.discriminator(self.G_sample, self.y, reuse = True)
    
        self.C_real = self.classifier(self.X)
        self.C_fake = self.classifier(self.G_sample, reuse = True)
    
        # loss BEGAN
        self.D_loss = tf.reduce_mean(self.D_real) - self.k * tf.reduce_mean(self.D_fake) 
        self.G_loss = tf.reduce_mean(self.D_fake)



        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))  

        # solver
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss + self.C_fake_loss, var_list=self.generator.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.C_real_loss, var_list=self.classifier.vars)
        #self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_fake_loss, var_list=self.generator.vars)        

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 32):
        fig_count = 0     

        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, y_b = self.data(batch_size)

                _, D_real_curr = self.sess.run(
                        [self.D_solver, self.D_real],
                        feed_dict={self.X: X_b, self.y:y_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr}
                        )
            # update G
            for _ in range(1):
                _, D_fake_curr = self.sess.run(
                        [self.G_solver, self.D_fake],
                        feed_dict={self.y:y_b, self.z: sample_z(batch_size, self.z_dim)}
                        )
            # update C
            for _ in range(1):
                # real label to train C
                self.sess.run(
                    self.C_real_solver,
                    feed_dict={self.X: X_b, self.y: y_b})
            
                # fake img label to train G
                '''self.sess.run(
                    self.C_fake_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})'''

            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr, C_real_loss_curr = self.sess.run(
                        [self.D_loss, self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr})
                G_loss_curr, C_fake_loss_curr = self.sess.run(
                        [self.G_loss, self.C_fake_loss],
                        feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr))

                if epoch % 100 == 0:
                    y_s = sample_y(16, self.y_dim)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)
            self.k_curr = self.k_curr + self.lam * (self.gamma * D_real_curr - D_fake_curr)
            #if epoch % 2000 == 0 and epoch != 0:
                #self.saver.save(self.sess, os.path.join(ckpt_dir, "CBE_gan_classifier.ckpt"))
    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, ckpt_dir)
        print("Model restored.")

class WorLS_GAN():
    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.tanh = True

        self.z_dim = self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        # nets
        self.G_sample = self.generator(self.z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)

        # loss
        # improved wgan
        self.lam = 10
        eps = tf.random_uniform([tf.shape(self.G_sample)[0], 1, 1, 1], minval=0., maxval=1.)#batch_size = 64 or lower than 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = self.discriminator(self.X_inter, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam *tf.reduce_mean(grad_norm- 1.)**2
        


        self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = tf.reduce_mean(self.D_fake)

        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self.generator.vars)
        
        # clip
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        

    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 64, restore = True):
        i = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, _ = self.data(batch_size)
                self.sess.run(self.clip_D)
                self.sess.run(
                        self.D_solver,
                        feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)}
                        )
            # update G
            self.sess.run(
                self.G_solver,
                feed_dict={self.z: sample_z(batch_size, self.z_dim)}
            )

            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                if epoch % 500 == 0:
                    samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
            if epoch % 1000 == 0 and epoch != 0:
                self.saver.save(self.sess, ckpt_dir+'W_GAN.ckpt', global_step=epoch)
    def test(self, sample_folder, sample_num):
        samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(sample_num, self.z_dim)})
        for i, sample in enumerate(samples):
            new_sample = np.concatenate((sample,sample),axis = 2)
            new_sample = np.concatenate((new_sample,sample),axis = 2)
            #fig = self.data.data2fig(samples)
            plt.imshow(new_sample)
            plt.axis('off')
            plt.savefig('{}/{}.jpg'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
            plt.close()

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class BEGAN():
    def __init__(self, generator, discriminator, data, flag=True):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        
        self.lam = 1e-2
        self.gamma = 0.75
        self.k_curr = 0.0

        self.z_dim = self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.k = tf.placeholder(tf.float32)
        # nets
        if flag:
            self.G_sample = self.generator(self.z, reuse = True)
           
        else:
            self.G_sample = self.generator(self.z, reuse = False)
        self.D_real = self.discriminator(self.X, reuse = False)

        #self.D_real, _ = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)

        # loss
        self.D_loss = tf.reduce_mean(self.D_real) - self.k * tf.reduce_mean(self.D_fake) 
        self.G_loss = tf.reduce_mean(self.D_fake)

        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        
     
        if flag:
            tf.get_variable_scope().reuse_variables()
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)
        
        # clip

        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #self.sess.run(tf.global_variables_initializer())

    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 128, restore = True):
        i = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())
        
            
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, _ = self.data(batch_size)

                _, D_real_curr = self.sess.run(
                        [self.D_solver, self.D_real],
                        feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr}
                        )
            # update G
            for _ in range(1):
                _, D_fake_curr = self.sess.run(
                        [self.G_solver, self.D_fake],
                        feed_dict={self.z: sample_z(batch_size, self.z_dim)}
                        )
                    
            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                if epoch % 500 == 0:
                    samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
            self.k_curr = self.k_curr + self.lam * (self.gamma * D_real_curr - D_fake_curr)
            if epoch % 1000 == 0 and epoch != 0:
                self.saver.save(self.sess, ckpt_dir+'BE_GAN.ckpt', global_step=epoch)
    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")
        return self.generator


class Classifier(object):
    def __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data

        self.lr = tf.placeholder(tf.float32)

        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
    
        self.C_real, self.C_softmax = self.classifier(self.X)

        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        
        # solver
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.9).minimize(self.C_real_loss , var_list=self.classifier.vars)

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train_classifier(self, sample_folder, ckpt_dir, training_epoches = 5000, batch_size = 32, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-3

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    self.C_real_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                    )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_real_loss_curr = self.sess.run(
                        [self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b})
                print('Iter: {}; C_real_loss: {}'.format(epoch,  C_real_loss_curr))

            if epoch % 500 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)
     
    def test(self):
        test_num = len(self.data.data)
        X_b, y_b = self.data(test_num)
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        C_out = self.sess.run(self.C_softmax, feed_dict={self.X: X_b})

        C_real_loss = self.sess.run([self.C_real_loss], feed_dict={self.X: X_b, self.y: y_b})
        for y in range(test_num):
            if C_out[y][0]>C_out[y][1]:
                if y_b[y][0]==1:
                    TP += 1 
                else:
                    FP += 1
            else:
                
                if y_b[y][0]==1:
                    FN += 1 
                else:
                    TN += 1
        print(TP,FN)   
        print(FP,TN) 
        #print(y_b)
        #print(C_out)
        print(C_real_loss)

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")
#test.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

from nets import *
from datas import *
from gans import *



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    folder = '/data/projects/yhliuzzb/'
    model = sys.argv[1]
    img_size = sys.argv[2]
    defect_id = sys.argv[3]
    defect_num = sys.argv[4]
    sample_num = sys.argv[5]
    
    print 'Model: '+model +'; Img_Resize: '+img_size +'; Defect_ID: '+defect_id +'; Defect_Num: '+defect_num +'; Sample_num: '+sample_num
    
    if model == 'wgan_adc' or model == 'lsgan_adc':
        if model == 'wgan_adc':
            loss_type = 'W'
        elif model == 'lsgan_adc':
            loss_type = 'LS'    	
    	sample_folder = folder+'Samples_single/adc_'+img_size+'_'+defect_id+'_'+loss_type+'gan_conv'
        ckpt_folder = folder+'ckpt/'+loss_type+'_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/'+loss_type+'_GAN_'+img_size+'_'+defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        generator = G_conv(size=int(img_size),is_tanh=True)
        discriminator = D_conv(size=int(img_size))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=1)
        
        # run
        GAN = WorLS_GAN(generator, discriminator, data)
        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))

    elif model == 'cwgan_adc' :#not finish

        defect_id_pre = defect_id.split(',')[0]
        for i in range(1,int(defect_num)):
            defect_id_post = defect_id.split(',')[i]
            #print(defect_id_pre,defect_id_post)
            defect_id_pre = defect_id_pre + '-' + defect_id_post
        new_defect_id = defect_id_pre
        print(new_defect_id)

        sample_folder = folder+'Samples/adc_'+img_size+'_'+new_defect_id+'_'+'cwgan_conv'
        ckpt_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_defect_id+'/'
        restore_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_defect_id+'/'

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv_condition(size=int(img_size))
        classifier = C_conv(size=int(img_size),class_num=int(defect_num))
         
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))

        # run
        GAN = CGAN_Classifier(generator, discriminator, classifier, data, loss_type = 'W')

        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))
    elif model == 'began_adc':
    
        sample_folder = folder+'Samples_single/adc_'+img_size+'_'+defect_id+'_'+'began_conv'
        ckpt_folder = folder+'ckpt/'+'BE_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/'+'BE_GAN_'+img_size+'_'+defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        generator = G_conv_BEGAN(size=int(img_size))
        discriminator = D_conv_BEGAN(size=int(img_size))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=1)
        
        # run
        GAN = BEGAN(generator, discriminator, data, flag=False)
        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))
    elif model == 'c_adc':
    
        defect_id_pre = defect_id.split(',')[0]
        for i in range(1,int(defect_num)):
            defect_id_post = defect_id.split(',')[i]
            #print(defect_id_pre,defect_id_post)
            defect_id_pre = defect_id_pre + '-' + defect_id_post
        new_defect_id = defect_id_pre
        print(new_defect_id)

        sample_folder = folder+'Samples/adc_'+img_size+'_'+new_defect_id+'_'+'classifier'
        ckpt_folder = folder+'ckpt/classifier_'+img_size+'_'+new_defect_id+'/'
        restore_folder = folder+'ckpt/classifier_'+img_size+'_'+new_defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        classifier = C_conv(size=int(img_size),class_num=int(defect_num))
         
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num), test=True)

        # run
        c = Classifier(classifier, data)
        c.restore_ckpt(restore_folder)
        c.test()

    else: print('wrong model')

#train.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

from nets import *
from datas import *
from gans import *



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    folder = '/data/projects/yhliuzzb/'
    model = sys.argv[1]
    img_size = sys.argv[2]
    batch = sys.argv[3]
    defect_id = sys.argv[4]
    defect_num = sys.argv[5]
    restore = sys.argv[6]
    print('------------------------------------------------------------------------------------------')
    print('Model: '+model +'; Img_Resize: '+img_size +'; Batch_Size: '+batch +'; Defect_ID: '+defect_id +'; Defect_num: '+defect_num +'; Restore_ckpt: '+restore)
    print('------------------------------------------------------------------------------------------')
    if model == 'cwgan_adc' :
        
        defect_id_pre = defect_id.split(',')[0]
        for i in range(1,int(defect_num)):
            defect_id_post = defect_id.split(',')[i]
            #print(defect_id_pre,defect_id_post)
            defect_id_pre = defect_id_pre + '-' + defect_id_post
        new_defect_id = defect_id_pre
        print(new_defect_id)

        sample_folder = folder+'Samples/adc_'+img_size+'_'+new_defect_id+'_'+'cwgan_conv'
        ckpt_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_defect_id+'/'
        restore_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=True)
        discriminator = D_conv_condition(size=int(img_size))
        classifier = C_conv(size=int(img_size),class_num=int(defect_num))
         
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))

        # run
        GAN = CGAN_Classifier(generator, discriminator, classifier, data, loss_type = 'W')
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=False)

    elif model == 'clsgan_c' :
        sample_folder = './Samples/mnist_clsgan_conv'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv_mnist(is_tanh=True)
        discriminator = D_conv_mnist()
        classifier = C_conv_mnist()
         
        data = mnist(is_tanh=True)

        # run
        clsgan_conv = CGAN_Classifier(generator, discriminator, classifier, data, loss_type = 'LS')
        clsgan_conv.train(sample_folder)
#######
    elif model == 'wgan_adc' :
        sample_folder = folder+'Samples/adc_'+img_size+'_'+defect_id+'_'+'wgan_conv'
        ckpt_folder = folder+'ckpt/'+'W_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/'+'W_GAN_'+img_size+'_'+defect_id+'/'
        
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=True)
        discriminator = D_conv(size=int(img_size))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))
        
        # run
        GAN = WorLS_GAN(generator, discriminator, data)
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=False)
        
    elif model == 'c_adc' :

        defect_id_pre = defect_id.split(',')[0]
        for i in range(1,int(defect_num)):
            defect_id_post = defect_id.split(',')[i]
            #print(defect_id_pre,defect_id_post)
            defect_id_pre = defect_id_pre + '-' + defect_id_post
        new_defect_id = defect_id_pre
        print(new_defect_id)

        sample_folder = folder+'Samples/adc_'+img_size+'_'+new_defect_id+'_'+'classifier'
        ckpt_folder = folder+'ckpt/classifier_'+img_size+'_'+new_defect_id+'/'
        restore_folder = folder+'ckpt/classifier_'+img_size+'_'+new_defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        classifier = C_conv(size=int(img_size),class_num=int(defect_num))
         
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))

        # run
        c = Classifier(classifier, data)
        if restore == 'True':
            c.restore_ckpt(restore_folder)
            c.train_classifier(sample_folder, ckpt_dir=ckpt_folder, restore=True)
        else:
            c.train_classifier(sample_folder, ckpt_dir=ckpt_folder, restore=False)


    elif model == 'began_adc' :
        sample_folder = folder+'Samples/adc_'+img_size+'_'+defect_id+'_BEgan_conv'
        ckpt_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+defect_id+'/'

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        # param
        generator = G_conv_BEGAN(size=int(img_size))
        discriminator_tmp = D_conv(size=int(img_size))
        discriminator = D_conv_BEGAN(size=int(img_size))
        
         
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))

        # run
        GAN = BEGAN(generator, discriminator, data, flag=False)
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            #GAN.restore_ckpt(folder+'ckpt/W_GAN_64_13/')
            #GAN.discriminator = discriminator
            #GAN.sess.run(tf.variables_initializer(GAN.discriminator.vars))
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, restore=False)

    elif model == 'began' :
        sample_folder = 'Samples/mnist_began_conv'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv_mnist(is_tanh=True)
        discriminator = D_conv_mnist_BEGAN()
        
         
        data = mnist(is_tanh=True)

        # run
        began_conv = BEGAN(generator, discriminator, data)
        
        began_conv.train(sample_folder)

    elif model == 'cbegan_c' :
        sample_folder = 'Samples/mnist_cbegan_conv'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv_mnist(is_tanh=True)
        discriminator = D_conv_mnist_BEGAN()
        classifier = C_conv_mnist()
         
        data = mnist(is_tanh=True)

        # run
        cbegan_conv = CBEGAN_Classifier(generator, discriminator, classifier, data)
        cbegan_conv.train(sample_folder)
    else:
        print('Wrong model')
#nets.py

import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

###############################################  mlp #############################################
class G_mlp(object):
    def __init__(self):
        self.name = 'G_mlp'

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64*64*3, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, tf.stack([tf.shape(z)[0], 64, 64, 3]))
            return g
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class D_mlp(object):
    def __init__(self):
        self.name = "D_mlp"

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            d = tcl.fully_connected(tf.flatten(x), 64, activation_fn=tf.nn.relu,normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            logit = tcl.fully_connected(d, 1, activation_fn=None)

        return logit

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#-------------------------------- MNIST for test ------
class G_mlp_mnist(object):
    def __init__(self):
        self.name = "G_mlp_mnist"
        self.X_dim = 784

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
        return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_mnist():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
            
        return d, q

    @property
    def vars(self):        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_mnist_BEGAN():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            #mse = tf.reduce_mean(tf.reduce_sum((x - d)**2, 1))
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
            
        return d, q

    @property
    def vars(self):        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Q_mlp_mnist():
    def __init__(self):
        self.name = "Q_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
        return q

    @property
    def vars(self):        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


###############################################  conv #############################################
class G_conv(object):
    def __init__(self, size, is_tanh=True):
        self.name = 'G_conv'
        self.size = size//16 #64//16
        self.channel = 1 #self.data.channel
        self.is_tanh = is_tanh
    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, self.size * self.size * 64, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm) #original 1024
            g = tf.reshape(g, (-1, self.size, self.size, 64))  # size
            g = tcl.conv2d_transpose(g, 32, 3, stride=2, # size*2
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 16, 3, stride=2, # size*4
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 8, 3, stride=2, # size*8
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            if self.is_tanh:
                g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                        activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            else:
                g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                        activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv(object):
    def __init__(self, size):
        self.name = 'D_conv'
        self.size = size  #64
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            shared = tcl.conv2d(x, num_outputs=self.size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=None)#tcl.batch_norm
            shared = tcl.conv2d(shared, num_outputs=self.size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=None)#tcl.batch_norm
            shared = tcl.conv2d(shared, num_outputs=self.size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=None)#tcl.batch_norm

            shared = tcl.flatten(shared)
            #shared = tcl.fully_connected(shared, 128, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))    
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            # q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            # q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d#, q
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv_condition(object):
    def __init__(self, size):
        self.name = 'D_conv_cond'
        self.size = size  #64
    def __call__(self, x, y, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            shared = tcl.conv2d(x, num_outputs=self.size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tf.concat([tcl.flatten(shared),y],1)
    
            #shared = tcl.fully_connected(shared, 128, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            #q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            #q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d#, q
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class G_conv_BEGAN(object):
    def __init__(self, size):
        self.name = 'G_conv'
        self.size = size//16 #64//16
        self.channel = 1 #self.data.channel
    def __call__(self, z, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            g = tcl.fully_connected(z, self.size * self.size * 64, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm) #original 1024
            g = tf.reshape(g, (-1, self.size, self.size, 64))  # size
            g = tcl.conv2d_transpose(g, 32, 3, stride=2, # size*2
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 16, 3, stride=2, # size*4
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 8, 3, stride=2, # size*8
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                    activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class D_conv_BEGAN(object):
    def __init__(self, size):
        self.name = 'D_conv_be'
        self.size = size
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
           
            encoder = tcl.conv2d(x, num_outputs=self.size, kernel_size=3, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=tf.nn.elu)
            encoder = tcl.conv2d(encoder, num_outputs=self.size * 8, kernel_size=3, # 16x16x128
                        stride=2, activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm)
            encoder = tcl.conv2d(encoder, num_outputs=self.size * 16, kernel_size=3, # 8x8x256
                        stride=2, activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm)
            encoder = tcl.conv2d(encoder, num_outputs=self.size * 32, kernel_size=3, # 4x4x512
                        stride=2, activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm)
            #shared = tcl.flatten(shared)
    
            encoder = tcl.fully_connected(encoder, self.size//16*self.size//16*4, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            decoder = tf.reshape(encoder, (-1, self.size//16, self.size//16, 64))  # size
            decoder = tcl.conv2d_transpose(decoder, 32, 3, stride=2, # size*2
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            decoder = tcl.conv2d_transpose(decoder, 16, 3, stride=2, # size*4
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            decoder = tcl.conv2d_transpose(decoder, 8, 3, stride=2, # size*8
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            decoder = tcl.conv2d_transpose(decoder, 1, 3, stride=2, # size*16
                                        activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            x_out = tf.reshape(decoder, (-1, self.size, self.size, 1)) 

            mse = tf.reduce_mean(tf.reduce_sum((x - x_out)**2, 1))

            return mse#, 0
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv(object):
    def __init__(self, size, class_num):
        self.name = 'C_conv'
        self.class_num = class_num
        self.size = size

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #size = 64
            shared = tcl.conv2d(x, num_outputs=self.size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            #d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
            #            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            
            q = tcl.fully_connected(tcl.flatten(shared), 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            p = tcl.fully_connected(q, self.class_num, activation_fn=None) # 10 classes
            class_out = tcl.fully_connected(q, self.class_num, activation_fn=tf.nn.softmax)
        
            return p, class_out
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class V_conv(object):
    def __init__(self):
        self.name = 'V_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=3, # 4x4x512
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            
            v = tcl.fully_connected(tcl.flatten(shared), 128)
            return v
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# -------------------------------- MNIST for test
class G_conv_mnist(object):
    def __init__(self, is_tanh):
        self.name = 'G_conv_mnist'
        self.is_tanh = is_tanh

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            #g = tcl.fully_connected(z, 1024, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                        weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 7*7*64, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tf.reshape(g, (-1, 7, 7, 64))  # 7x7
            g = tcl.conv2d_transpose(g, 32, 4, stride=2, # 14x14x64
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            if self.is_tanh:
                g = tcl.conv2d_transpose(g, 1, 4, stride=2, # 28x28x1
                                    activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            else:
                g = tcl.conv2d_transpose(g, 1, 4, stride=2, # 28x28x1
                                    activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class D_conv_mnist(object):
    def __init__(self):
        self.name = 'D_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 32
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)
            
            d = tcl.fully_connected(shared, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d, q
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv_mnist_BEGAN(object):
    def __init__(self):
        self.name = 'D_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 32
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)
            shared = tcl.fully_connected(shared, 28*28, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            x_out = tf.reshape(shared, (-1, 28, 28, 1)) 

            mse = tf.reduce_mean(tf.reduce_sum((x - x_out)**2, 1))
            q = tcl.fully_connected(shared, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return mse, q
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv_mnist(object):
    def __init__(self):
        self.name = 'C_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=5, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=5, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=tf.nn.relu)
            
            #c = tcl.fully_connected(shared, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            c = tcl.fully_connected(shared, 10, activation_fn=None) # 10 classes
            return c
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
