"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../../Utilities/') # for plotting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
# from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, xe, u_xe, xf, layers, lb, ub):
        self.x0 = x0
        self.u0 = u0
        self.xe = xe
        self.u_xe = u_xe
        self.xf = xf
        self.lb = lb
        self.ub = ub
        self.layers = layers
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        # number of cols = 1
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]]) # (1, 1)
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]]) # (1, 1)
        self.xe_tf = tf.placeholder(tf.float32, shape=[None, self.xe.shape[1]]) # (1, 1)
        self.u_xe_tf = tf.placeholder(tf.float32, shape=[None, self.u_xe.shape[1]]) # (1, 1)
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]]) # (30, 1)

        # tf Graphs: return u, u_x, f
        self.u0_pred, _ , _ = self.net_all(self.x0_tf)
        _ , self.u_xe_pred, _ = self.net_all(self.xe_tf)
        self.uf_pred, self.u_xf_pred, self.f_pred = self.net_all(self.xf_tf) # used in predict (only call net_all once)
        
        ## u''(x) = sin(x), x in [0, pi]
        ## u(0) = 0
        ## u'(pi) = 1
        # Loss: initial + boundary + PDE
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_xe_tf - self.u_xe_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred)) # f_pred = u_xx - sin(x) = 0
                    # td.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        self.train_op_Adam = tf.train.AdamOptimizer().minimize(self.loss) # return a minimization Op (a graph node that performs computation on tensors) -> updates weights and biases

        # tf session: initiates a tf Graph(defines computations) in which tensors are processed through operations + allocates resources and holds intermediate values
        self.sess = tf.Session()
        init = tf.global_variables_initializer() # variables now hold the values from declarations: tf.Variable(tf.zeros(...)), tf.Variable(tf.random_normal(...)), etc
        self.sess.run(init) # required to initialize the variables
              
    def initialize_NN(self, layers):    
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
        # tf.Variable: for trainable variables/mutable tensor values that persist across multiple sesssion.run()
        # https://towardsdatascience.com/understanding-fundamentals-of-tensorflow-program-and-why-it-is-necessary-94cf5b60e255
            weights.append(self.xavier_init(size=[layers[l], layers[l+1]]))
            biases.append(tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)) # all zeros
        return weights, biases
        
    def xavier_init(self, size):
        # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        # Want activation outputs of each layer to have stddev around 1 -> repeat matrix mult across as many network layers as want, without activations exploding or vanishing
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        # random values from a truncated normal distribution (values whose magnitude>2 staddev from mean are dropped and re-picked)
        # Shape of the output tensor: [layers[l], layers[l+1]]
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1 # 6 in this case
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # Initializing first input: mapping to [-1, 1] 
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) # passing along networks
            # NOTE: H*W=(50, 20) + B(1, 20) -> tf does broadcasting: B becomes (50, 20)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # passed 5 times in total
        return Y
    
    def net_all(self, x): # x = (50,1)
        u = self.neural_net(x, self.weights, self.biases) # (50, 1)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_xx - tf.sin(x) # f = u_xx - sin(x) = 0 # (50, 1)
        return u, u_x, f
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self): # one iteration: uses all training data from tf_dict and updates weights and biases
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0,
                   self.xe_tf: self.xe, self.u_xe_tf: self.u_xe,
                   self.xf_tf: self.xf}
        self.sess.run(self.train_op_Adam, tf_dict) # feeding training examples during training and running the minimization Op of self.loss
        loss_value = self.sess.run(self.loss, tf_dict)
        return loss_value

    def predict(self, xt):
        tf_dict = {self.xf_tf: xt}
        u, f = self.sess.run([self.uf_pred, self.f_pred], tf_dict)
        # self.uf_pred, self.u_xf_pred, self.f_pred = self.net_all(self.xf_tf)
        return u, f
    
if __name__ == "__main__": 
    ## u''(x) = sin(x), x in [0, pi]
    ## u(0) = 0
    ## u'(pi) = 1
    ## analytical solution: u(x) = -sin(x)

    ###########################
    ## PART 1: setting parameters and getting accurate data for evaluation 

    # Domain bounds
    lb = np.array([0])
    ub = np.array([np.pi])
    
    layers = [1, 20, 20, 20, 1] # 5-layer deep NN with 100 neurons/layer & hyperbolic tangent act. func.
    
    ## Getting ground truth data based on analytical solution 
    # analytical solution: u(x) = -sin(x)
    # N = 1000 # Number of total data points
    # x = np.reshape(np.linspace(0, np.pi, N), (-1, 1)) # [[0] [pi/2] [pi]]
    # u = np.sin(x) # [[0] [1] [1.2246468e-16]]
    
    ###########################
    ## PART 2：randomly picking the training set from full analytical solution for uniform grid
    # people perfer deterministic way now: want to make sure points cover whole domain/dense everywhere
    # random: better for complex geometrical problem
    x0 = np.array([[0]])
    u0 = np.array([[0]])
    xe = np.array([[np.pi]])
    u_xe = np.array([[1]])
    N_f = 30 # Number of collocation points
    # idx = np.random.choice(x.shape[0], N_f, replace=False) 
    # x_f = x[idx,:] # N_f rows from x
    # u_f = u[idx,:] # corresponding N_f rows from u (matching sin(x))
    xf = np.reshape(np.linspace(0, np.pi, N_f), (-1, 1)) # [[0] [pi/2] [pi]]

    ###########################  
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(x0, u0, xe, u_xe, xf, layers, lb, ub)
             
    N_t = 50
    xt = np.reshape(np.linspace(0, np.pi, N_t), (-1, 1)) # [[0] [pi/2] [pi]]
    ut = -1 * np.sin(xt)

    start_time = time.time()
    # Loss: 10^-3/-4 should be about good
    loss_values = []
    u_preds = [] # 4 * (50, 1)
    f_preds = [] # 4 * (50, 1)
    N_iter = 4000
    for i in range(N_iter):
        loss_value = model.train() 
        if (i+1) % 10 == 0:  # wouldn't plot in beginning, but plot at the end
            loss_values.append(loss_value)
            # print('Iter: %d, Loss: %.3e, Time: %.2f' % (i+1, loss_value, time.time() - start_time))
        if (i+1) % 1000 == 0: # wouldn't plot in beginning, but plot at the end
            u_pred, f_pred = model.predict(xt) # f = u_xx - tf.sin(x) # xt: (50, 1)
            u_preds.append(u_pred)
            f_preds.append(f_pred)
            # print('     u_pred: %.3e, f_pred: %.3e' % (u_pred , f_pred)
            # print('u_pred:', u_pred.shape) # (50, 1)
            # print('f_pred:', f_pred.shape) # (50, 1)
            ## analytical solution: u(x) = -sin(x)
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)

    print("loss_values:", loss_values)
    print('Training time: %.4f' % (time.time() - start_time))
    u_pred, f_pred = model.predict(xt)
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weigths & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network
   
   ###########################  
    ## PART 4: calculating errors  
    error_u = np.linalg.norm(u_pred  - ut, 2) / np.linalg.norm( ut, 2)
    print('Error u: %e' % (error_u))
    
    ###########################  
    ## PART 5: Plotting
    # 1. plot loss_values (loss vs. iteration) 
    fig = plt.figure()
    x_coords = 10*(np.array(range(len(loss_values))) + 1)
    plt.semilogy(x_coords, loss_values)  # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss during Training")
    init_tuple = (10,loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, textcoords='data', fontsize=7)
    last_tuple = (N_iter,loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=7)
    fig.subplots_adjust(right=0.86)
    # Oscillation: 1. overshoot (fixed -> decaying learning rate) 
    # 2/ Adam: gradient descent + momentum (sometime parameter change makes the loss go up)
    plt.savefig('./figures/forward_1d_loss.pdf')

    # 2. plot u vs x
    print("u_preds.shape", u_preds.shape)
    print("f_preds.shape", f_preds.shape)
    fig = plt.figure()
    for i in range(u_preds.shape[0]): # 4
        ax = plt.subplot(2, 2, i+1)
        exact_plot, = ax.plot(xt, ut, 'b-', label = 'Exact') # tuple unpacking   
        pred_plot, = ax.plot(xt, u_preds[i], 'r--', label = 'Prediction')    
        ax.set_xlabel('$x$',fontsize=6) # $: mathematical font like latex
        ax.set_ylabel('$u$',fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        ax.set_title('$Iteration = %d$' % ((i+1)*1000), fontsize = 8)
    plt.figlegend(handles=(exact_plot, pred_plot),labels=('Exact', 'Prediction'), loc='upper center', ncol=2, fontsize=7) # from last subplot
    fig.subplots_adjust(wspace=0.35, hspace=0.45)
    plt.savefig('./figures/forward_1d.pdf')

