# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:04:27 2021

@author: Wang
"""

import tensorflow as tf
import numpy as np
from pyDOE import lhs
import time
import pandas as pd

# np.random.seed(1234)
# tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, Z_theta, Z_RE, theta, layers_theta, layers_k, layers_psi, lb, ub):

        self.lb = lb
        self.ub = ub

        self.z = Z_theta[:, 0:1]
        self.t = Z_theta[:, 1:2]
        self.z_re = Z_RE[:, 0:1]
        self.t_re = Z_RE[:, 1:2]

        self.theta = theta

        self.layers_theta = layers_theta
        self.layers_k = layers_k
        self.layers_psi = layers_psi

        # Initialize NNs
        self.weights_psi, self.biases_psi = self.initialize_NN('psi_', layers_psi)
        self.weights_k, self.biases_k = self.initialize_MNN('k_', layers_k)
        self.weights_theta, self.biases_theta = self.initialize_MNN('theta_', layers_theta)

        # var_list
        all_vars = tf.trainable_variables()
        psi_vars = [var for var in all_vars if 'psi_' in var.name]
        k_vars = [var for var in all_vars if 'k_' in var.name]
        theta_vars = [var for var in all_vars if 'theta_' in var.name]
        RRE_vars = psi_vars + k_vars + theta_vars

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])
        self.z_re_tf = tf.placeholder(tf.float32, shape=[None, self.z_re.shape[1]])
        self.t_re_tf = tf.placeholder(tf.float32, shape=[None, self.t_re.shape[1]])
        self.psi_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])

        tf.log_h = tf.math.log(-self.psi_tf)
        self.WRC_theta = self.net_theta(-tf.log_h, self.weights_theta, self.biases_theta)
        self.HCF_K = self.net_k(-tf.log_h, self.weights_k, self.biases_k)
        self.theta_pred, self.k_pred, self.psi_pred = self.net_RE(self.z_tf, self.t_tf)
        self.theta_t, self.psi_z, self.psi_zz, self.K_z, self.f_RE = self.net_f_RE(self.z_re_tf, self.t_re_tf)
        
        self.loss_theta = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred))
        self.loss_RE = tf.reduce_mean(tf.square(self.f_RE))
        self.loss_l2 = (self.cal(k_vars) + self.cal(theta_vars) +self.cal(psi_vars))/ tf.Variable([4500.0], dtype=tf.float32)
        self.loss_sm = self.loss_theta*tf.Variable([10.0], dtype=tf.float32)  + self.loss_RE  + self.loss_l2*tf.Variable([0.001], dtype=tf.float32)
        #learning rate
        self.learning_rate = tf.train.exponential_decay(learning_rate=0.0005, global_step=tf.Variable(0, trainable=False), decay_steps=10000, decay_rate=0.99, staircase=True)
        
        

        # L-BFGS-B method----------------------------------------------------------------------------------------------------------------------------------
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss_sm,
                                                        var_list = RRE_vars,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})

        # Adam method
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(loss=self.loss_sm, var_list=RRE_vars)
        
       
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # tf.saver
        self.saver = tf.train.Saver()
        print(tf.trainable_variables)

    def initialize_NN(self, name, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(name, size=[layers[l], layers[l + 1]])
            with tf.variable_scope(name):
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def cal(self, W):
        for i, each in enumerate(W):
            if i == 0:
                f = tf.reduce_sum(tf.reduce_sum(tf.square(each)))
            else:
                f = f + tf.reduce_sum(tf.reduce_sum(tf.square(each)))
        return f
                
    
    def initialize_MNN(self, name, layers): # monotonic neural network
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(name, size = [layers[l],layers[l+1]])
            W2 = W**2
            with tf.variable_scope(name):
                b = tf.Variable(tf.zeros([1, layers[l+1]], dtype = tf.float32), dtype = tf.float32)
            weights.append(W2)
            biases.append(b)
        return weights, biases

    def xavier_init(self, name, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        with tf.variable_scope(name):
            w = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        return w
    
    def net_k(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        K = tf.exp(Y)  # force psi to be positive
        return K

    def net_psi(self, z, t, weights, biases): 
        X = tf.concat([z, t],1)
        num_layers = len(weights) + 1
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        psi = -tf.exp(Y) 
        return psi
    
    def net_theta(self, X, weights, biases): 
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        theta = tf.sigmoid(Y)
        return theta
    
    
    def net_RE(self, z, t):
        psi = self.net_psi(z, t, self.weights_psi, self.biases_psi)
        log_h = tf.math.log(-psi)
        theta = self.net_theta(-log_h, self.weights_theta, self.biases_theta)
        k = self.net_k(-log_h, self.weights_k, self.biases_k)
        
        return theta, k, psi
    
    def net_f_RE(self, z, t):  
        theta, k, psi = self.net_RE(z, t)
        
        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(k, z)[0]
        f = theta_t - K_z*psi_z - k*psi_zz - K_z
        
        return theta_t, psi_z, psi_zz, K_z, f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, N_iter):
        
        start_time = time.time()
        tf_dict = {self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta,
                   self.z_re_tf: self.z_re, self.t_re_tf: self.t_re}
        # Adam
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_sm, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()
                
        # L-BFGS-B
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_sm],
                                loss_callback=self.callback)
        
        loss_value = self.sess.run(self.loss_sm, tf_dict)
        
        
           
    def predict(self, Z_star):

        tf_dict = {self.z_tf: Z_star[:, 0:1], self.t_tf: Z_star[:, 1:2], 
                   self.z_re_tf: Z_star[:, 0:1], self.t_re_tf: Z_star[:, 1:2]}
        
        theta_star = self.sess.run(self.theta_pred, tf_dict)
        psi_star = self.sess.run(self.psi_pred, tf_dict)
        K_star = self.sess.run(self.k_pred, tf_dict)
        f_star = self.sess.run(self.f_RE, tf_dict)
        theta_t_star = self.sess.run(self.theta_t, tf_dict)
        psi_z_star = self.sess.run(self.psi_z, tf_dict)
        psi_zz_star = self.sess.run(self.psi_zz, tf_dict)
        K_z_star = self.sess.run(self.K_z, tf_dict)
        flux_star = -K_star*(psi_z_star + 1.0)

        return theta_star, psi_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star
    
    def predict_T(self, Z_star):

        T_star = self.sess.run(self.T_pred, {self.z_T_tf: Z_star[:, 0:1], self.t_T_tf: Z_star[:, 1:2]})
        T_t_p = self.sess.run(self.T_t_pred, {self.z_T_tf: Z_star[:, 0:1], self.t_T_tf: Z_star[:, 1:2]})
        
        return T_star, T_t_p
    
    def save_model(self, name, path):
        name = name + '.ckpt'
        save_path = self.saver.save(self.sess, path + '/' + name)   
        print("Model saved in path: %s" % save_path)
        
    def load_model(self, name, path):
        name = name + '.ckpt'
        self.saver.restore(self.sess, path + '/' + name)
        
    def WRC_HCF(self, psi_star):
        tf_dict = {self.psi_tf: psi_star}

        theta = self.sess.run(self.WRC_theta, tf_dict)
        K = self.sess.run(self.HCF_K, tf_dict)
        return theta, K




if __name__ == "__main__":
    
    tf.reset_default_graph()
    
    noise = 0.02

    N_u = 100
    N_f = 1500
    N_re = 1000
    layers_psi = [2, 40, 40, 40, 40, 40, 40, 1]
    layers_k = [1, 10, 1]
    layers_theta = [1, 10, 1]

    # data
    data = pd.read_csv("D:/case311_siltloam.csv")
    savepath = "D:/"
    t_star = data['time'].values[:,None]
    z_star = data['depth'].values[:,None]
    psi_star = data['head'].values[:,None]
    K_star = data['K'].values[:,None]
    C_star = data['C'].values[:,None]
    theta_star = data['theta'].values[:,None]
    flux_star = data['flux'].values[:,None]
    T_star = data['T'].values[:,None]
    Z_star = np.hstack((z_star, t_star))
    
    depth_increment = 4
    fixed_position_full = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]  
    fixed_position = fixed_position_full[::depth_increment]
    fixed_list_p = [int(i * (-200)) for i in fixed_position]
    
    for i in range(251):
        if i == 0:
            fixed_list = fixed_list_p
        else:
            fixed_list_plus = [each + 1001*i for each in fixed_list_p]
            fixed_list = np.append(fixed_list, fixed_list_plus)
    
    # noise
    noise_theta = noise*np.random.randn(theta_star.shape[0], theta_star.shape[1])
    theta_noise = theta_star + noise_theta

    # fixed points
    Z_theta_train = Z_star[fixed_list,:]
    theta_train = theta_noise[fixed_list, :]
    
    lb = Z_star.min(0)
    ub = Z_star.max(0)
    lb[0] = -20
    lhs_f = lhs(2, N_f)
    Z_heat_train = lb + (ub - lb) * lhs_f
    lhs_re = lhs(2, N_re)
    Z_RE_train = lb + (ub - lb) * lhs_re

    tf.reset_default_graph()
    
    
    model = PhysicsInformedNN(Z_theta_train, Z_RE_train, theta_train, layers_theta, layers_k, layers_psi, lb, ub)
    start_time = time.time()
    
    model.train(100000)
    theta_pred, psi_pred, K_pred, f_pred, theta_t_pred, psi_z_pred, psi_zz_pred, K_z_pred, flux_pred = model.predict(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                    'theta_actual': theta_star.flatten(), 'theta_pred': theta_pred.flatten(),
                    'theta_noise': theta_noise.flatten(),
                    'psi_actual': psi_star.flatten(), 'psi_pred': psi_pred.flatten(),
                    'K_actual': K_star.flatten(), 'K_pred': K_pred.flatten(),
                    'flux_actual': flux_star.flatten(), 'flux_pred': flux_pred.flatten(),
                    'f_pred': f_pred.flatten(), 'theta_t_pred': theta_t_pred.flatten(),
                    'psi_z_pred': psi_z_pred.flatten(), 'psi_zz_pred': psi_zz_pred.flatten(),
                    'K_z_pred': K_z_pred.flatten()})
    dataset.to_csv(f"{savepath}data_rre.csv")
    
    #lookup
    log_h_look = np.arange(-5, 3.4, 0.021)
    h_look = 10**log_h_look
    psi_look = -h_look.reshape(400,1)
    theta_look, K_look = model.WRC_HCF(psi_look)

    lookup = pd.DataFrame({'theta': theta_look.flatten(), 'psi': psi_look.flatten()})
    lookup.to_csv(f"{savepath}lookup_rre.csv", index = False)                              
                                       