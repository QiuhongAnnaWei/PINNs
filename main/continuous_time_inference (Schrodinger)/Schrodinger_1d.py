"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0) (append a second argument 0 to each sampled x)
            # x0 = [[1] [0.5]] (initial points' x); [[0] [0]]
            # [[1   0]
            #  [0.5 0]]
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb) 
            # [[-5] [-5]]; tb = [[1] [0]] (boundary points' t)
            # [[-5 1]
            #  [-5 0]]
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
            # [[5 1]
            #  [5 0]]
        
        self.lb = lb
        self.ub = ub
        self.x0 = X0[:,0:1] # 2d: [[1] [0.5]]
        self.t0 = X0[:,1:2]
        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]
        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u0 = u0
        self.v0 = v0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders: tf.placeholder inserts a placeholder for a tensor that'll always be fed -> used to feed actual training examples/able to run the same model on different problem set
        # https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]]) # N0 = 50
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]]) #  N_b = 50
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]]) # N_f = 20000
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.u0_pred, self.v0_pred, _ , _ = self.net_uv(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        
        # Loss: initial(file) + periodic boundary + PDE
        # TODO: change
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) # td.reduce_mean: computes the mean of elements across dimensions of a tensor
        
        # Optimizers: QUESTION
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        # Wrapper allowing scipy.optimize.minimize() to perform minimization by controlling a tf.Session.: used in predict() # QUESTION
        self.train_op_Adam = tf.train.AdamOptimizer().minimize(self.loss) # return a minimization Op (a graph node that performs computation on tensors)

        # tf session: initiates a tf Graph(defines computatinos) in which tensors are processed through operations + allocates resources and holds intermediate values
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer() # variables now hold the values from declarations: tf.Variable(tf.zeros(...)), tf.Variable(tf.random_normal(...)), etc
        self.sess.run(init) # required to initialize the variables
              
    def initialize_NN(self, layers): # layers = [2, 100, 100, 100, 100, 2]       
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
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # Initializing first input # QUESTION
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) # passing along networks
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # passed 5 times in total
        return Y
    
    def net_uv(self, x, t):
        X = tf.concat([x,t],1)  # input
        # x = [[-0.5], [0.5]] # t = [[0], [1]]
        # [[-0.5, 0]
        #  [0.5,  1]]
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]
        v = uv[:,1:2]
        
        u_x = tf.gradients(u, x)[0] # automatic differentiation
        v_x = tf.gradients(v, x)[0] # automatic differentiation

        return u, v, u_x, v_x

    def net_f_uv(self, x, t): # PDE
        u, v, u_x, v_x = self.net_uv(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v # QUESTION
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u # QUESTION
        
        return f_u, f_v
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        # feeding training examples during training
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict) # feeding training examples during training and running the minimization Op of self.loss
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict, # specifies the placeholder values for the fetches computation
                                fetches = [self.loss], # indicate what it is we want to compute
                                loss_callback = self.callback)      
        # Variables subject to optimization are updated in-place at the end of optimization.  
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)  
        v_star = self.sess.run(self.v0_pred, tf_dict)  
        
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
               
        return u_star, v_star, f_u_star, f_v_star
    
if __name__ == "__main__": 
     
    noise = 0.0  # for inverse (to assume some error in observation data) 

    ###########################
    ## PART 1: setting parameters and getting accurate data for evaluation 
    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000

    layers = [2, 100, 100, 100, 100, 2] # 5-layer deep NN with 100 neurons/layer & hyperbolic tangent act. func.
    
    ## Extracting ground truth data from numerical method from file
    data = scipy.io.loadmat('../Data/NLS.mat')
    t = data['tt'].flatten()[:,None] # probably just 't' [[0] [1]]
    x = data['x'].flatten()[:,None] # [0. 0.5 1. ] -> [[0] [0.5] [1]]
    # NOTE: equivalent to reshape
        # a = [1 2 3 4] -> shape: (4,)
        # a[:,None] = [[1] [2] [3] [4]] -> shape: (4, 1)
    Exact = data['uu'] # should have been 'h' (probably 6 entries)
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    ## Reformatting the ground truth data for calculating errors
    X, T = np.meshgrid(x,t) # making 2-d grid from two 1-d arrays
    # [[0 0.5 1 ]
    #  [0 0.5 1 ]]
    # [[0 0 0]]
    #  [1 1 1]]
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # each row: one point (spatial, temporal) (ab, 2)
        # [[0] [0.5] [1] [0] [0.5] [1]]      # [[0] [0] [0] [1] [1] [1]]
        # [[0 0] [0.5 0] [1 0] [0 1] [0.5 1] [1 1]]
    u_star = Exact_u.T.flatten()[:,None] # vectors (T: transpose) 
    v_star = Exact_v.T.flatten()[:,None] # vectors
    h_star = Exact_h.T.flatten()[:,None] # vectors
    
    ###########################
    ## PART 2：randomly picking the training set from the data (no full analytical solution for uniform grid)
    # people perfer deterministic way now: want to make sure points cover whole domain/dense everywhere
    # random: better for complex geometrical problem
    # TODO: use analytical solution for uniform grid
    idx_x = np.random.choice(x.shape[0], N0, replace=False) # random indices
    x0 = x[idx_x,:] # initial data from file -> used in training # [[1] [0.5]]
    u0 = Exact_u[idx_x,0:1] # matching u  # [[u0] [u1]] ?exact format
    v0 = Exact_v[idx_x,0:1] # matching v  # [[v0] [v1]] ?xact format
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False) # random indices
    tb = t[idx_t,:] # boundary data from file -> used in training
    
    X_f = lb + (ub-lb)*lhs(2, N_f) # dense collocation points from near-random sampling
        # [[ 2.28434627  0.58958455]
        #  [-4.3445426   1.25654662]
        #  [ 4.28407126  1.37822203]...] (N_f rows of (x, t))
    # TODO: uniform meshgrid is fine for rectangular domain

    ###########################  
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
             
    start_time = time.time()                
    model.train(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
        
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star) # predicting on X_star
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    ###########################  
    ## PART 4: calculating errors  
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    ###########################  
    # QUESTION: not used?
    ## PART 5: In order to plot, interpolate from prediction data(generated for X_f? QUESTION) to match ground truth grid(X_star)
    # TODO: for us, can use analytical solution to generate values for X_f directly
    # X_star: coordinates. 2nd entry: values, (X, T): points at which to interpolate data)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')     
    
    # NOTE: interpolation -> for consistency (compare result) or structure
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    # 75, 100, 125: timestamp
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    # savefig('./figures/NLS')  
    
