# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:04:27 2021

@author: Wang
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# np.random.seed(1234)
# tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, Z_theta, Z_RE, Z_T, Z_heat, theta, T, layers_theta, layers_k, layers_psi, layers_T, layers_lam, lb, ub):

        self.lb = lb
        self.ub = ub

        self.z = Z_theta[:, 0:1]
        self.t = Z_theta[:, 1:2]
        self.z_re = Z_RE[:, 0:1]
        self.t_re = Z_RE[:, 1:2]
        self.z_T = Z_T[:, 0:1]
        self.t_T = Z_T[:, 1:2]
        self.z_heat = Z_heat[:, 0:1]
        self.t_heat = Z_heat[:, 1:2]
        
        self.adaptive_constant_val = np.array(1.0)
        self.beta = 0.9
        self.theta = theta
        self.T = T

        self.layers_theta = layers_theta
        self.layers_k = layers_k
        self.layers_psi = layers_psi
        self.layers_T = layers_T

        # Initialize NNs
        self.weights_psi, self.biases_psi = self.initialize_NN('psi_', layers_psi)
        self.weights_T, self.biases_T = self.initialize_NN('Tem_', layers_T)
        self.weights_k, self.biases_k = self.initialize_MNN('k_', layers_k)
        self.weights_theta, self.biases_theta = self.initialize_MNN('theta_', layers_theta)
        self.weights_lam, self.biases_lam = self.initialize_MNN('lam_', layers_lam)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])
        self.T_tf = tf.placeholder(tf.float32, shape=[None, self.T.shape[1]])
        self.z_heat_tf = tf.placeholder(tf.float32, shape=[None, self.z_heat.shape[1]])
        self.t_heat_tf = tf.placeholder(tf.float32, shape=[None, self.t_heat.shape[1]])
        self.z_T_tf = tf.placeholder(tf.float32, shape=[None, self.z_T.shape[1]])
        self.t_T_tf = tf.placeholder(tf.float32, shape=[None, self.t_T.shape[1]])
        self.z_re_tf = tf.placeholder(tf.float32, shape=[None, self.z_re.shape[1]])
        self.t_re_tf = tf.placeholder(tf.float32, shape=[None, self.t_re.shape[1]])
        self.T_t_tf = tf.placeholder(tf.float32, shape=[None, self.T.shape[1]])
        self.psi_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])
        
        all_vars = tf.trainable_variables()
        psi_vars = [var for var in all_vars if 'psi_' in var.name]
        k_vars = [var for var in all_vars if 'k_' in var.name]
        theta_vars = [var for var in all_vars if 'theta_' in var.name]
        T_vars = [var for var in all_vars if 'Tem_' in var.name]
        lam_vars = [var for var in all_vars if 'lam_' in var.name]
        RRE_vars = psi_vars + k_vars + theta_vars
        heat_vars = lam_vars + T_vars
        
        tf.log_h = tf.math.log(-self.psi_tf)
        self.WRC_theta = self.net_theta(-tf.log_h, self.weights_theta, self.biases_theta)
        self.HCF_K = self.net_k(-tf.log_h, self.weights_k, self.biases_k)
        self.Horton_lam = self.net_lam(-tf.log_h, self.weights_lam, self.biases_lam)
        self.T_pred = self.net_T(self.z_T_tf, self.t_T_tf, self.weights_T, self.biases_T)
        self.theta_pred, self.k_pred, self.psi_pred = self.net_RE(self.z_tf, self.t_tf)
        self.theta_t, self.psi_z, self.psi_zz, self.K_z, self.f_RE = self.net_f_RE(self.z_re_tf, self.t_re_tf)
        self.f_heat, self.lam, self.f1, self.f2, self.f3= self.net_f_heat_net(self.z_heat_tf, self.t_heat_tf)  #拿出热导率
        
        self.loss_l2 = (self.cal(T_vars))/ tf.Variable([4500.0], dtype=tf.float32)
        self.loss_T1 = tf.reduce_mean(tf.square(self.T_tf - self.T_pred)) + self.loss_l2*tf.Variable([0.01], dtype=tf.float32)
        self.loss_theta = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred))
        self.loss_heat = tf.reduce_mean(tf.square(self.f_heat))
        self.loss_RE = tf.reduce_mean(tf.square(self.f_RE))
        self.loss_sm = self.loss_theta*tf.Variable([10.0], dtype=tf.float32)  + self.loss_RE
        self.loss_st = self.loss_heat + tf.reduce_mean(tf.square(self.T_tf - self.T_pred))*tf.Variable([100.0], dtype=tf.float32)
        #learning rate
        self.learning_rate = tf.train.exponential_decay(learning_rate=0.0005, global_step=tf.Variable(0, trainable=False), decay_steps=10000, decay_rate=0.99, staircase=True)
        self.learning_rate1 = tf.train.exponential_decay(learning_rate=0.0005, global_step=tf.Variable(0, trainable=False), decay_steps=5000, decay_rate=0.99, staircase=True)
        

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
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(loss=self.loss_sm, var_list=RRE_vars)
        
        # optimizer for T----------------------------------------------------------------------------------------------------------------------------------
        self.optimizer_T = tf.contrib.opt.ScipyOptimizerInterface(self.loss_T1,
                                                        var_list = T_vars,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})
        # Adam method for T
        self.optimizer_Adam_T = tf.train.AdamOptimizer()
        self.train_op_Adam_T = self.optimizer_Adam_T.minimize(loss=self.loss_T1, var_list=T_vars)
        
        # optimizer for heat----------------------------------------------------------------------------------------------------------------------------------
        self.optimizer_heat = tf.contrib.opt.ScipyOptimizerInterface(self.loss_st,
                                                        var_list = heat_vars,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})
        # Adam method for heat
        self.optimizer_Adam_heat = tf.train.AdamOptimizer(learning_rate=self.learning_rate1)
        self.train_op_Adam_heat = self.optimizer_Adam_heat.minimize(loss=self.loss_st, var_list=heat_vars)

        
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # tf.saver
        self.saver = tf.train.Saver()
        print(tf.trainable_variables)
    def cal(self, W):
        for i, each in enumerate(W):
            if i == 0:
                f = tf.reduce_sum(tf.reduce_sum(tf.square(each)))
            else:
                f = f + tf.reduce_sum(tf.reduce_sum(tf.square(each)))
        return f
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
        K = tf.exp(Y)
        return K
    
    def net_lam(self, X, weights, biases): 
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        lam = tf.exp(Y)  
        return lam
        
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
        psi = -tf.exp(Y)  # force psi to be negative
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
    
    def net_T(self, z, t, weights, biases): # net for T
        X = tf.concat([z, t],1)
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        T = tf.exp(Y)
        return T
    
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

    def net_f_heat_net(self, z, t): #loam
        beta = tf.constant(5, dtype=tf.float32)
        Cn_N = tf.constant(0.8456293, dtype=tf.float32)
        Cw = tf.constant(3.12035, dtype=tf.float32)     
        mult = tf.constant(100, dtype=tf.float32)
        # eps = tf.constant(0.0000001, dtype=tf.float32)
        
        theta, k, psi = self.net_RE(z, t)
        T = self.net_T(z, t, self.weights_T, self.biases_T)
        T_z = tf.gradients(T, z)[0]
        psi_z = tf.gradients(psi, z)[0]
        flux = -k*(psi_z + 1.0)
        log_h = tf.math.log(-psi)
        lam1 = self.net_lam(-log_h, self.weights_lam, self.biases_lam) 
        lam2 = beta*Cw*tf.abs(flux)
        lam = lam1 * mult + lam2
        Cp = Cn_N + Cw*theta  #10 14
        
        Cpt = Cp*T
        qT = flux*T
        lamT = lam*T_z

        f1 = tf.gradients(Cpt, t)[0]
        f2 = tf.gradients(lamT, z)[0]
        f3 = Cw * tf.gradients(qT, z)[0]
        f = (f1 - f2 + f3)
        return f, lam, f1, f2, f3

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
        
    def train_for_T_batch(self, max_epoch, batch_size):   
    
        def get_batch(z, t, T, batch_size, it_one_epoch):
            idx = np.random.choice(T.shape[0], T.shape[0], replace = False)
            z = z[idx]
            t = t[idx]
            T = T[idx]
            z_batch = []
            t_batch = []
            T_batch = []
            for it in range(it_one_epoch):
                if it == it_one_epoch-1:
                    z_batch.append(z[-batch_size:,:])
                    t_batch.append(t[-batch_size:,:])
                    T_batch.append(T[-batch_size:,:])
                else:
                    z_batch.append(z[it*batch_size:(it+1)*batch_size,:])
                    t_batch.append(t[it*batch_size:(it+1)*batch_size,:])
                    T_batch.append(T[it*batch_size:(it+1)*batch_size,:])
            return z_batch, t_batch, T_batch
        it_one_epoch = int(np.ceil(self.T.shape[0]/batch_size))
            
        start_time = time.time()
        
        
        # Adam
        for it in range(max_epoch):
            z_batch, t_batch, T_batch = get_batch(self.z_T, self.t_T, self.T, batch_size, it_one_epoch)
            for i in range(it_one_epoch):
                train_T = T_batch[i]
                train_z = z_batch[i]
                train_t = t_batch[i]
                
                tf_dict = {self.z_T_tf: train_z, self.t_T_tf: train_t, self.T_tf: train_T}
                self.sess.run(self.train_op_Adam_T, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_T1, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()     
        # L-BFGS-B
        tf_dict = {self.z_T_tf: self.z_T, self.t_T_tf: self.t_T, self.T_tf: self.T}
        self.optimizer_T.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_T1],
                                loss_callback=self.callback)
        loss_value = self.sess.run(self.loss_T1, tf_dict)
        self.T_t_star = self.sess.run(self.T_t_pred, tf_dict)

        
    def train_heat(self, N_iter):      
        start_time = time.time()

        lam_list = []
        loss_list = []
        f1_list = []
        f2_list = []
        f3_list = []
        # Adam
        for it in range(N_iter):
            tf_dict = {self.z_heat_tf: self.z_heat, self.t_heat_tf: self.t_heat, self.T_tf: self.T, 
                       self.z_T_tf: self.z_T, self.t_T_tf: self.t_T, 
                       self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta}
            self.sess.run(self.train_op_Adam_heat, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_st, tf_dict)
                lam_value = self.sess.run(self.lam, tf_dict)
                f1_value = self.sess.run(self.f1, tf_dict)
                f2_value = self.sess.run(self.f2, tf_dict)
                f3_value = self.sess.run(self.f3, tf_dict)
                loss_list.append(loss_value.flatten())
                lam_list.append(lam_value.flatten())
                f1_list.append(f1_value.flatten())
                f2_list.append(f2_value.flatten())
                f3_list.append(f3_value.flatten())
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()     
        # L-BFGS-B
        self.optimizer_heat.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_st],
                                loss_callback=self.callback)
        loss_value = self.sess.run(self.loss_st, tf_dict)
        return loss_list, lam_list, f1_list, f2_list, f3_list
          
    def predict(self, Z_star):

        tf_dict = {self.z_tf: Z_star[:, 0:1], self.t_tf: Z_star[:, 1:2], 
                   self.z_T_tf: Z_star[:, 0:1], self.t_T_tf: Z_star[:, 1:2],
                   self.z_heat_tf: Z_star[:, 0:1], self.t_heat_tf: Z_star[:, 1:2],
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
        h_star = self.sess.run(self.f_heat, tf_dict)
        T_star = self.sess.run(self.T_pred, tf_dict)
        lam_star = self.sess.run(self.lam, tf_dict)

        return theta_star, psi_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star, h_star, T_star, lam_star
    
    def predict_T(self, Z_star):

        T_star = self.sess.run(self.T_pred, {self.z_T_tf: Z_star[:, 0:1], self.t_T_tf: Z_star[:, 1:2]})
        
        return T_star
    
    def save_model(self, name, path):
        name = name + '.ckpt'
        save_path = self.saver.save(self.sess, path + '/' + name)   
        print("Model saved in path: %s" % save_path)
        
    def load_model(self, name, path):
        name = name + '.ckpt'
        self.saver.restore(self.sess, path + '/' + name)
        
    def WRC_HCF_lam(self, psi_star):
        tf_dict = {self.psi_tf: psi_star}

        theta = self.sess.run(self.WRC_theta, tf_dict)
        K = self.sess.run(self.HCF_K, tf_dict)
        lam = self.sess.run(self.Horton_lam, tf_dict)
        return theta, K, lam




if __name__ == "__main__":
    
    tf.reset_default_graph()
    
    noise = 0.0

    N_u = 100
    N_f = 1500
    N_re = 1000
    layers_psi = [2, 40, 40, 40, 40, 40, 40, 1]
    layers_k = [1, 10, 1]
    layers_theta = [1, 10, 1]
    layers_T = [2, 40, 40, 40, 40, 1] 
    layers_lam = [1, 20, 1]

    # data
    data = pd.read_csv("D:/case321_siltloam.csv")
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
            
    depth_increment_T = 3
    fixed_position_T = fixed_position_full[::depth_increment_T] 
    fixed_list_p_T = [int(i * (-200)) for i in fixed_position_T]
    
    for i in range(251):
        if i == 0:
            fixed_list_T = fixed_list_p_T
        else:
            fixed_list_plus_T = [each + 1001*i for each in fixed_list_p_T]
            fixed_list_T = np.append(fixed_list_T, fixed_list_plus_T)
    
    # noise
    noise_theta = noise*np.random.randn(theta_star.shape[0], theta_star.shape[1])
    theta_noise = theta_star + noise_theta
    noise_T = noise*np.random.randn(T_star.shape[0], T_star.shape[1])*np.std(T_star) 
    T_noise = T_star + noise_T

    # fixed points (dimensionless and raw)
    Z_theta_train = Z_star[fixed_list,:]
    theta_train = theta_noise[fixed_list, :]
    Z_T_train = Z_star[fixed_list_T,:]
    T_train = T_noise[fixed_list_T, :]
    
    # Doman bounds, collocation points
    lb = Z_star.min(0)
    ub = Z_star.max(0)
    lb[0] = -20
    lhs_f = lhs(2, N_f)
    Z_heat_train = lb + (ub - lb) * lhs_f
    lhs_re = lhs(2, N_re)
    Z_RE_train = lb + (ub - lb) * lhs_re
    
    model = PhysicsInformedNN(Z_theta_train, Z_RE_train, Z_T_train, Z_heat_train, theta_train, T_train, layers_theta, layers_k, 
                              layers_psi, layers_T, layers_lam, lb, ub)
    start_time = time.time()
    
    # train
    model.train_for_T_batch(2000, 16) 

    T_pred = model.predict_T(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                        'T_actual': T_star.flatten(), 'T_pred': T_pred.flatten()})
    dataset.to_csv(f"{savepath}data_T.csv")

    model.train(50)
    theta_pred, psi_pred, K_pred, f_pred, theta_t_pred, psi_z_pred, psi_zz_pred, K_z_pred, flux_pred, h_pred, T_pred, lam_pred = model.predict(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                    'theta_actual': theta_star.flatten(), 'theta_pred': theta_pred.flatten(),
                    'T_actual': T_star.flatten(), 'T_pred': T_pred.flatten(),
                    'theta_noise': theta_noise.flatten(),
                    'psi_actual': psi_star.flatten(), 'psi_pred': psi_pred.flatten(),
                    'K_actual': K_star.flatten(), 'K_pred': K_pred.flatten(),
                    'flux_actual': flux_star.flatten(), 'flux_pred': flux_pred.flatten(),
                    'f_pred': f_pred.flatten(), 'theta_t_pred': theta_t_pred.flatten(),
                    'psi_z_pred': psi_z_pred.flatten(), 'psi_zz_pred': psi_zz_pred.flatten(),
                    'K_z_pred': K_z_pred.flatten(), 'h_pred': h_pred.flatten(), 'lam_pred': lam_pred.flatten()})
    dataset.to_csv(f"{savepath}data_rre.csv")
    
    loss_list, lam_list, f1_list, f2_list, f3_list = model.train_heat(50)
    theta_pred, psi_pred, K_pred, f_pred, theta_t_pred, psi_z_pred, psi_zz_pred, K_z_pred, flux_pred, h_pred, T_pred, lam_pred = model.predict(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                    'theta_actual': theta_star.flatten(), 'theta_pred': theta_pred.flatten(),
                    'T_actual': T_star.flatten(), 'T_pred': T_pred.flatten(),
                    'theta_noise': theta_noise.flatten(),
                    'psi_actual': psi_star.flatten(), 'psi_pred': psi_pred.flatten(),
                    'K_actual': K_star.flatten(), 'K_pred': K_pred.flatten(),
                    'flux_actual': flux_star.flatten(), 'flux_pred': flux_pred.flatten(),
                    'f_pred': f_pred.flatten(), 'theta_t_pred': theta_t_pred.flatten(),
                    'psi_z_pred': psi_z_pred.flatten(), 'psi_zz_pred': psi_zz_pred.flatten(),
                    'K_z_pred': K_z_pred.flatten(), 'h_pred': h_pred.flatten(), 'lam_pred': lam_pred.flatten()})
    dataset.to_csv(f"{savepath}data_heat.csv")
      
    # lookup table
    log_h_look = np.arange(-5, 3.4, 0.021)
    h_look = 10**log_h_look
    psi_look = -h_look.reshape(400,1)
    theta_look, K_look, lam_look = model.WRC_HCF_lam(psi_look)
    
    lookup = pd.DataFrame({'theta': theta_look.flatten(), 'psi': psi_look.flatten(),
                           'K': K_look.flatten(), 'lam': lam_look.flatten()})
    lookup.to_csv(f"{savepath}lookup_heat.csv", index = False)                                   
                                       