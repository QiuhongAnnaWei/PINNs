"""
@author: Maziar Raissi
"""
# Inverse: given observed data of u(t, x) -> model/pde parameters λ

import time, sys, os, json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sys.path.insert(0, '../../Utilities/')  # for plotting

# from plotting import newfig, savefig

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    def __init__(self, x0, u0, xb, u_xb, xo, uo, xf, lambda0, layers, lowerbound, upperbound):
        self.x0 = x0
        self.u0 = u0
        self.xb = xb
        self.u_xb = u_xb
        self.xo = xo
        self.uo = uo
        self.xf = xf

        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.lambda_1 = tf.Variable([lambda0[0]], dtype=tf.float32)
        self.lambda_2 = tf.Variable([lambda0[1]], dtype=tf.float32)

        # number of cols = 1
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])  # (1, 1)
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])  # (1, 1)
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])  # (1, 1)
        self.u_xb_tf = tf.placeholder(tf.float32, shape=[None, self.u_xb.shape[1]])  # (1, 1)
        self.xo_tf = tf.placeholder(tf.float32, shape=[None, self.xo.shape[1]])  # N_train x 1
        self.uo_tf = tf.placeholder(tf.float32, shape=[None, self.uo.shape[1]])  # N_train x 1
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])  # N_f x 1

        # tf Graphs: u, u_x, f = net_all(x)
        self.u0_pred, _, _ = self.net_all(self.x0_tf)
        _, self.u_xb_pred, _ = self.net_all(self.xb_tf)
        self.uo_pred, _, _ = self.net_all(self.xo_tf)
        _, _, self.f_pred = self.net_all(self.xf_tf)

        # Loss: initial + boundary + observed data + PDE
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_xb_tf - self.u_xb_pred)) + \
                    tf.reduce_mean(tf.square(self.uo_tf - self.uo_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))  # NOTE: different from the observed data
        # tf.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        # return a minimization Op (a graph node that performs computation on tensors) -> updates weights, biases, lambdas
        self.train_op_Adam = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        # NOTE: default: learning_rate=0.001 (typically 0.001 is the max, can make smaller)
        
        # tf session: initiates a tf Graph (defines computations) that processes tensors through operations + allocates resources + holds intermediate values
        self.sess = tf.Session()
        init = tf.global_variables_initializer() # variables now hold the values from declarations: tf.Variable(tf.zeros(...)), tf.Variable(tf.random_normal(...)), etc
        self.sess.run(init)  # required to initialize the variables

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            # tf.Variable: for trainable variables/mutable tensor values that persist across multiple sesssion.run()
            # https://towardsdatascience.com/understanding-fundamentals-of-tensorflow-program-and-why-it-is-necessary-94cf5b60e255
            weights.append(self.xavier_init(size=[layers[l], layers[l+1]]))
            biases.append(tf.Variable(
                tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32))  # all zeros
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
        num_layers = len(weights) + 1  # 6 in this case
        H = 2.0*(X - self.lowerbound)/(self.upperbound - self.lowerbound) - 1.0 # Initializing first input: mapping to [-1, 1]
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # passing along networks
            # NOTE: H*W=(50, 20) + B(1, 20) -> tf does broadcasting: B becomes (50, 20)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  # passed 5 times in total
        return Y

    def net_all(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        u = self.neural_net(x, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_xx - lambda_1 * tf.sin(lambda_2 * x)
        return u, u_x, f

    def train(self):  # one iteration: uses all training data from tf_dict and updates weights, biases, lambdas
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0,
                   self.xb_tf: self.xb, self.u_xb_tf: self.u_xb,
                   self.xo_tf: self.xo, self.uo_tf: self.uo, self.xf_tf: self.xf}
        # feeding training examples during training and running the minimization Op of self.loss
        self.sess.run(self.train_op_Adam, tf_dict)
        loss_value, lambda_1, lambda_2 = self.sess.run([self.loss, self.lambda_1, self.lambda_2], tf_dict) # lambda_1/2 do not need tf_dict
        return loss_value, lambda_1, lambda_2

    def predict(self, x):
        tf_dict = {self.xo_tf: x, self.xf_tf: x}  # no need for u
        # want to use the values in Session
        u, f = self.sess.run([self.uo_pred, self.f_pred], tf_dict)
        return u, f


if __name__ == "__main__":
    # u''(x) = lambda_1 * sin(lambda_2 * x), x in [0, pi] -> u''(x) = 9 * sin(3 * x)
        # NOTE: lambda defined in f/pde, not in u(x)
        # NOTE: to uniquely identify lambda, typically need boundary condition
    # u(0) = 0
    # u'(pi) = 3, u'(x) = - 3 cos(3x)
        ## NOTE: given initial and boundary -> used in loss (identical to forward implementation)
    # analytical solution: u(x) = - sin(3x)

    # global settings for all subplots 
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8

    ###########################
    ## PART 1: initialization
    
    # 5-layer deep NN with 100 neurons/layer & hyperbolic tangent act. func.
    layers = [1, 20, 20, 20, 1]

    # Domain bounds
    lowerbound = np.array([0])
    upperbound = np.array([np.pi])

    ###########################
    ## PART data
    # initial condition
    x0 = np.array([[lowerbound[0]]])
    u0 = np.array([[0]])

    # boundary condition
    xb = np.array([[upperbound[0]]])
    u_xb = np.array([[3]])

    # observed u based on analytical solution
    N_observed = 5
    xo = np.reshape(np.linspace(lowerbound[0]+0.2, upperbound[0]-0.2, N_observed), (-1, 1))
    # xo = np.array([[1.1], [2.1]])
    uo = -1 * np.sin(3 * xo)

    # collocation points for enforcing f=0
    # NOTE: separate from observed data: want f/residual = 0 everywhere
    N_f = 30
    xf = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_f), (-1, 1)) # collocation points

    # testing data
    N_test = 50
    xt = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_test), (-1, 1))  # [[0] [pi/2] [pi]]
    ut = -1 * np.sin(3 * xt)

    # initial values for lambdas
    lambda0 = np.array([5, 2])

    ###########################
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(x0, u0, xb, u_xb, xo, uo, xf, lambda0, layers, lowerbound, upperbound)

    start_time = time.time()
    dirpath = f'./1d/inverse_1d_figures/{start_time}' # where figures are stored
    os.mkdir(dirpath)
    # Loss: 10^-3/-4 should be about good
    loss_values, u_preds, f_preds = ([] for i in range(3))
    lambda_1s = [lambda0[0]] # 1d - initial
    lambda_2s = [lambda0[1]] # 1d - initial
    N_iter = 18000
    loss_value_step = 10
    pred_step = 2000
    for i in range(N_iter):
        loss_value, lambda_1, lambda_2 = model.train() # from last iteration
        if (i+1) % loss_value_step == 0:  # start with i=9 and end with i=3999 (last iter)
            loss_values.append(loss_value)
            lambda_1s.append(lambda_1)
            lambda_2s.append(lambda_2)
            print('Iter: %d, Loss: %.3e, Lambda_1: %.5f, lambda_2: %.5f, Time: %.2f' %
                  (i+1, loss_value, lambda_1, lambda_2, time.time() - start_time))
        if (i+1) % pred_step == 0:  # start with i=999 and end with i=3999 (last iter)
            u_pred, f_pred = model.predict(xt)
            u_preds.append(u_pred) # (N_test, 1)
            f_preds.append(f_pred) # (N_test, 1)
            # print('     u_pred: %.3e, f_pred: %.3e' % (u_pred , f_pred)
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)
    u_pred, f_pred = model.predict(xt)
    print("Initial lambda_1: %.5f, Final lambda_1: %.5f" % (lambda0[0], lambda_1))
    print("Initial lambda_2: %.5f, Final lambda_2: %.5f" % (lambda0[1], lambda_2))
    # print('Training time: %.4f' % (time.time() - start_time))
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weights & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network
    
    ###########################
    ## PART 4: calculating errors
    error_u = np.linalg.norm(u_pred - ut, 2) / np.linalg.norm(ut, 2) # scalar
    error_lambda_1 = np.abs(lambda_1 - 9)/9 * 100  # Ground truth: lambda_1=9 # 1d np array
    error_lambda_2 = np.abs(lambda_2 - 3)/3 * 100  # Ground truth: lambda_2=3 # 1d np array
    print('Error u: %e | Error lambda_1: %.5f%% | Error lambda_2: %.5f%%' % (error_u, error_lambda_1, error_lambda_2))
    
    ###########################
    ## PART 5: Plotting

    # 1. loss vs. iteration, lambda_1 vs iteration, lambda_2 vs iteration
    fig = plt.figure(figsize=(8, 4.8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    fig.add_subplot(gs[0, :])
    x_coords = loss_value_step * (np.array(range(len(loss_values))) + 1)
    plt.semilogy(x_coords, loss_values) # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.gca().set(xlabel="Iteration", ylabel="Loss", title="Loss during Training")
    init_tuple = (loss_value_step, loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, textcoords='data', fontsize=6)
    last_tuple = (N_iter, loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=6)
    
    x_coords = loss_value_step * (np.array(range(len(lambda_1s)))) # has an additional entry than loss_values: initial
    fig.add_subplot(gs[1, 0])
    plt.plot(x_coords, lambda_1s)
    plt.gca().set(xlabel="Iteration", ylabel="Lambda_1", title="Lambda_1 during Training")
    init_tuple = (0, lambda_1s[0])
    plt.annotate('(%d, %.3f)' % init_tuple, xy=init_tuple,textcoords='data', fontsize=6)
    last_tuple = (N_iter, lambda_1s[-1])
    plt.annotate('(%d, %.3f)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=6)
    
    fig.add_subplot(gs[1, 1])
    plt.plot(x_coords, lambda_2s)
    plt.gca().set(xlabel="Iteration", ylabel="Lambda_2", title="Lambda_2 during Training")
    init_tuple = (0, lambda_2s[0])
    plt.annotate('(%d, %.3f)' % init_tuple, xy=init_tuple,textcoords='data', fontsize=6)
    last_tuple = (N_iter, lambda_2s[-1])
    plt.annotate('(%d, %.3f)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=6)
    
    fig.subplots_adjust(wspace=0.3, hspace=0.35)
    plt.savefig(f'{dirpath}/inverse_1d_loss.pdf')

    # 2. plot u vs x (exact, prediction)
    plt.rcParams['axes.labelpad'] = -1 # default = 4
    fig = plt.figure()
    for i in range(u_preds.shape[0]):
        ax = plt.subplot(3, 3, i+1)
        exact_plot, = ax.plot(xt, ut, 'b-', label='Exact')  # tuple unpacking
        pred_plot, = ax.plot(xt, u_preds[i], 'r--', label='Prediction')
        plt.gca().set(xlabel="$x$", ylabel="$u$", title='$Iteration = %d$' % ((i+1)*pred_step))
    plt.figlegend(handles=(exact_plot, pred_plot), labels=('Exact', 'Prediction'), loc='upper center', ncol=2, fontsize=7)  # from last subplot
    fig.subplots_adjust(wspace=0.3, hspace=0.51)
    plt.savefig(f'{dirpath}/inverse_1d.pdf')

    infoDict = {
        'problem':{
            'pde form': 'u''(x) = lambda_1 * sin(lambda_2 * x)',
            'pde value': 'u''(x) = 9 * sin(3 * x)',
            'boundary (x)': [float(lowerbound[0]),float(upperbound[0])], 
            'initial condition': 'u(0) = 0',
            'boundary condition': "u'(pi) = 3",
            'analytical solution': 'u(x) = - sin(3x)'
        },
        'model':{
            'layers': str(layers),
            'training iteration': N_iter,
            'initial lambda_1': float(lambda0[0]),
            'final lambda_1': float(lambda_1[0]),
            'error_lambda_1 percentage': float(error_lambda_1[0]),
            'initial lambda_2': float(lambda0[1]),
            'final lambda_2': float(lambda_2[0]),
            'error_lambda_2 percentage': float(error_lambda_2[0]),
            'error_u': error_u
        }, 
        'training data':{
            'initial': (float(x0[0][0]), float(u0[0][0])),
            'boundary': (float(xb[0][0]), float(u_xb[0][0])) ,
            'N_observed': N_observed,
            'xo': str(xo),
            'uo': str(uo),
            'N_f': N_f,
            'xf': str(xf),
        },
        'testing data':{
            'N_test': N_test,
            'xt': str(xt),
            'ut': str(ut)
        }
    }
    with open(f'{dirpath}/info.json', 'w') as f:
        json.dump(infoDict, f, indent=4)

    # TODO: noisy data (to reflect practiacl situation)
    # noise = 0.01  # (to assume some error in observation data)
    # u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1]) # samples of specified size from standard normal
    # v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
    # retrain model
    # obtain error_lambda_1_noisy and print
