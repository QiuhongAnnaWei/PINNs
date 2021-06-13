# Forward: given model/pde parameters λ -> u(t, x)

import time, sys, os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
# from plotting import newfig, savefig
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import scipy.io
# from scipy.interpolate import griddata
# from pyDOE import lhs

sys.path.insert(0, '../../Utilities/') # for plotting

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, xb, u_xb, xf, layers, lowerbound, upperbound):
        self.x0 = x0
        self.u0 = u0
        self.xb = xb
        self.u_xb = u_xb
        self.xf = xf
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        # number of cols = 1
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]]) # (1, 1)
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]]) # (1, 1)
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]]) # (1, 1)
        self.u_xb_tf = tf.placeholder(tf.float32, shape=[None, self.u_xb.shape[1]]) # (1, 1)
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]]) # (30, 1)

        # tf Graphs: u, u_x, f = net_all(x)
        self.u0_pred, _ , _ = self.net_all(self.x0_tf)
        _ , self.u_xb_pred, _ = self.net_all(self.xb_tf)
        self.uf_pred, self.u_xf_pred, self.f_pred = self.net_all(self.xf_tf) # used in predict (only call net_all once)
        
        ## u''(x) = sin(x), x in [0, pi]
        ## u(0) = 0
        ## u'(pi) = 1
        # Loss: initial + boundary + PDE
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_xb_tf - self.u_xb_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred)) # f_pred = u_xx - sin(x) = 0
                    # td.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        # return a minimization Op (a graph node that performs computation on tensors) -> updates weights and biases
        self.train_op_Adam = tf.train.AdamOptimizer().minimize(self.loss)

        ## tf session: initiates a tf Graph (defines computations) that processes tensors through operations + allocates resources + holds intermediate values
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
        # Want each layer's activation outputs to have stddev around 1 -> repeat matrix mult across as many layers without activations exploding or vanishing
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        # random values from a truncated normal distribution (values whose magnitude>2 staddev from mean are dropped and re-picked)
        # Shape of the output tensor: [layers[l], layers[l+1]]
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1 # 6 in this case
        H = 2.0*(X - self.lowerbound)/(self.upperbound - self.lowerbound) - 1.0 # Initializing first input: mapping to [-1, 1] 
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) # passing along networks (tanh range: -1 to 1)
            # NOTE: H*W=(50, 20) + B(1, 20) -> tf does broadcasting: B becomes (50, 20)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # passed 5 times in total
        return Y # u
    
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
                   self.xb_tf: self.xb, self.u_xb_tf: self.u_xb,
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
    ## u_xx = sin(x), x in [0, pi]
    ## u(0) = 0
    ## u_x(pi) = 1
    ## analytical solution: u(x) = -sin(x)

    ###########################
    ## PART 1: setting parameters and getting accurate data for evaluation 

    # Domain bounds
    lowerbound = np.array([0])
    upperbound = np.array([np.pi])
    
    layers = [1, 20, 20, 20, 1] # 4-layer deep NN with 20 neurons/layer & hyperbolic tangent act. func.
    
    ## Getting ground truth data based on analytical solution 
    # analytical solution: u(x) = -sin(x)
    # N = 1000 # Number of total data points
    # x = np.reshape(np.linspace(0, np.pi, N), (-1, 1)) # [[0] [pi/2] [pi]]
    # u = np.sin(x) # [[0] [1] [1.2246468e-16]]
    
    ###########################
    ## PART 2：randomly picking/preping the training set from full analytical solution for uniform grid
    # people perfer deterministic way now: want to make sure points cover whole domain/dense everywhere
    # random: better for complex geometrical problem

    # boundary condition
    x0 = np.array([[0]])
    u0 = np.array([[0]])
    xb = np.array([[np.pi]])
    u_xb = np.array([[1]])

    # collocation points for enforcing f=0
    N_f = 30
    # idx = np.random.choice(x.shape[0], N_f, replace=False) 
    # x_f = x[idx,:] # N_f rows from x
    # u_f = u[idx,:] # corresponding N_f rows from u (matching sin(x))
    xf = np.reshape(np.linspace(0, np.pi, N_f), (-1, 1)) # [[0] [pi/2] [pi]]

    # testing data
    N_test = 50
    xt = np.reshape(np.linspace(0, np.pi, N_test), (-1, 1)) # [[0] [pi/2] [pi]]
    ut = -1 * np.sin(xt)

    ###########################  
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(x0, u0, xb, u_xb, xf, layers, lowerbound, upperbound)
             
    start_time = time.time()

    # settings for plots 
    dirpath = f'./main/1d/forward_1d_figures/{start_time}' # where figures are stored
    os.mkdir(dirpath)
    ticksize = 8.5
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    plt.rcParams['axes.labelsize'] = 9.5
    plt.rcParams['axes.titlesize'] = 10.5
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['legend.handlelength'] = 0.4
    annotatesize = 9.5
    dataDict = {
        'boundary points':{
            'xb': xb.tolist(),
            'u_xb': u_xb.tolist(),
            'x0': x0.tolist(),
            'u0': u0.tolist(),
        },
        'collocation points':{
            'N_f': N_f,
            'xf': xf.tolist()
        },
        'testing data':{
            "N_test": N_test,
            "xt": xt.tolist(),
            "ut": ut.tolist()
        }
    }
    with open(f'{dirpath}/data.json', 'w') as f:
        json.dump(dataDict, f)
    # Note: loss around 10^-3/-4 should be about good
    loss_values, u_preds, f_preds = ([] for i in range(3))
    N_iter = 4000
    loss_value_step = 10
    pred_step = 1000
    for i in range(N_iter):
        loss_value = model.train() 
        if (i+1) % loss_value_step == 0:  # wouldn't plot in beginning, but plot at the end
            loss_values.append(float(loss_value))
            print('Iter: %d, Loss: %.3e, Time: %.2f' % (i+1, loss_value, time.time() - start_time))
        if (i+1) % pred_step == 0: # wouldn't plot in beginning, but plot at the end
            u_pred, f_pred = model.predict(xt) # f = u_xx - tf.sin(x) # xt: (50, 1)
            u_preds.append(u_pred) # (N_test, 1)
            f_preds.append(f_pred) # (N_test, 1)
    training_time = time.time() - start_time
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)
    u_pred, f_pred = model.predict(xt)
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weights & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network
   
   ###########################  
    ## PART 4: calculating errors
    error_u = np.linalg.norm(u_pred  - ut, 2) / np.linalg.norm(ut, 2) # scalar
    print('Error u: %e' % (error_u))
    
    ###########################  
    ## PART 5: Plotting
    # Plot 1. loss vs. iteration) 
    fig = plt.figure(figsize=(5.25, 5.25))
    plt.ticklabel_format(axis='x', style="sci", scilimits=(2, 2))
    x_coords = loss_value_step * (np.array(range(len(loss_values))) + 1)
    plt.semilogy(x_coords, loss_values)  # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.gca().set(xlabel='Iteration', ylabel='Loss', title='Loss during Training')
    init_tuple = (loss_value_step,loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, fontsize=annotatesize, ha='left')
    last_tuple = (N_iter,loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, fontsize=annotatesize, ha='right', va='top')
    plt.plot([init_tuple[0], last_tuple[0]], [init_tuple[1], last_tuple[1]], '.', c='#3B75AF')
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.1, top=0.95) 
    # Oscillation: actually very small nummerical difference because of small y scale
    # 1. overshoot (fixed -> decaying learning rate) 
    # 2. Adam: gradient descent + momentum (sometime parameter change makes the loss go up)
    plt.savefig(f'{dirpath}/forward_1d_loss.pdf')
    plt.close(fig)
    with open(f'{dirpath}/forward_1d_loss.json', 'w') as f:
        json.dump({"x_coords": x_coords.tolist(), "loss_values": loss_values}, f)

    # Plot 2. u vs x (exact, prediction)
    fig = plt.figure(figsize=(6.8, 5.25))
    for i in range(u_preds.shape[0]): # 4
        ax = plt.subplot(2, 2, i+1)
        exact_plot, = ax.plot(xt, ut, 'b-', label = 'Exact') # tuple unpacking   
        pred_plot, = ax.plot(xt, u_preds[i], 'r--', label = 'Prediction')  
        plt.gca().set(xlabel='$x$', ylabel='$u$', title=f'Snapshot at Iteration = {(i+1)*pred_step}')  
    plt.figlegend(handles=(exact_plot, pred_plot),labels=('Exact', 'Prediction'), loc='upper center', ncol=2, fontsize=ticksize) # from last subplot
    fig.subplots_adjust(wspace=0.26, hspace=0.32, left=0.08, right=0.98, bottom=0.08, top=0.89)
    plt.savefig(f'{dirpath}/forward_1d_u.pdf')
    plt.close(fig)
    with open(f'{dirpath}/forward_1d_u_preds.json', 'w') as f:
        json.dump(u_preds.tolist(), f)

    ###########################
    ## PART 6: Saving information
    infoDict = {
        'problem':{
            'pde form': 'u_xx = sin(x)',
            'boundary (x)': [float(lowerbound[0]),float(upperbound[0])], 
            'boundary condition (u)': 'u(0) = 0',
            'boundary condition (u_x)': 'u_x(pi) = 1',
            'analytical solution': 'u(x) = -sin(x)'
        },
        'model':{
            'layers': str(layers),
            'training iteration': N_iter,
            'loss_value_step':loss_value_step,
            'pred_step': pred_step,
            'training_time': training_time,
            'error_u': error_u,
        },
        'training data':{
            'x0': str(x0),
            'u0': str(u0),
            'xb': str(xb),
            'u_xb': str(u_xb),
            'N_f': N_f,
            'xf': str(xf)
        }
    }
    with open(f'{dirpath}/info.json', 'w') as f:
        json.dump(infoDict, f, indent=4)
