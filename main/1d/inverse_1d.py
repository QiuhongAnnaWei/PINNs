"""
@author: Maziar Raissi
"""
# Inverse: given observed data of u(t, x) -> model/pde parameters λ

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '../../Utilities/')  # for plotting

# from plotting import newfig, savefig

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class (x_train, u_train, layers)
    def __init__(self, x0, u0, xe, u_xe, xo, uo, xr, layers):
        self.x0 = x0
        self.u0 = u0
        self.xe = xe
        self.u_xe = u_xe
        self.xo = xo
        self.uo = uo
        self.xr = xr  # NOTE: separate from observed data: want residual = 0 everywhere
        self.lb = np.array([0])
        self.ub = np.array([np.pi])
        self.layers = layers
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        # number of cols = 1
        self.x0_tf = tf.placeholder(
            tf.float32, shape=[None, self.x0.shape[1]])  # (1, 1)
        self.u0_tf = tf.placeholder(
            tf.float32, shape=[None, self.u0.shape[1]])  # (1, 1)
        self.xe_tf = tf.placeholder(
            tf.float32, shape=[None, self.xe.shape[1]])  # (1, 1)
        self.u_xe_tf = tf.placeholder(
            tf.float32, shape=[None, self.u_xe.shape[1]])  # (1, 1)
        self.xo_tf = tf.placeholder(
            tf.float32, shape=[None, self.xo.shape[1]])  # N_train x 1
        self.uo_tf = tf.placeholder(
            tf.float32, shape=[None, self.uo.shape[1]])  # N_train x 1
        self.xr_tf = tf.placeholder(
            tf.float32, shape=[None, self.xr.shape[1]])  # N_res x 1

        # tf Graphs: return u, u_x, f
        self.u0_pred, _, _ = self.net_all(self.x0_tf)
        _, self.u_xe_pred, _ = self.net_all(self.xe_tf)
        self.uo_pred, _, _ = self.net_all(self.xo_tf)
        _, _, self.f_pred = self.net_all(self.xr_tf)

        # u''(x) = sin(x), x in [0, pi]
        # u(0) = 0
        # u'(pi) = 1
        # Loss: initial + boundary + observed data + PDE
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
            tf.reduce_mean(tf.square(self.u_xe_tf - self.u_xe_pred)) + \
            tf.reduce_mean(tf.square(self.uo_tf - self.uo_pred)) + \
            tf.reduce_mean(tf.square(
                self.f_pred))  # f_pred = u_xx - sin(x) = 0 # NOTE: different fromm the observed data
        # tf.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        # return a minimization Op (a graph node that performs computation on tensors) -> updates lambdaxxx
        self.train_op_Adam = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(self.loss)
        # NOTE: default: learning_rate=0.001 (typically 0.001 is the max, can make smaller)
        # tf session: initiates a tf Graph(defines computations) in which tensors are processed through operations + allocates resources and holds intermediate values
        self.sess = tf.Session()
        # variables now hold the values from declarations: tf.Variable(tf.zeros(...)), tf.Variable(tf.random_normal(...)), etc
        init = tf.global_variables_initializer()
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
        # Want activation outputs of each layer to have stddev around 1 -> repeat matrix mult across as many network layers as want, without activations exploding or vanishing
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        # random values from a truncated normal distribution (values whose magnitude>2 staddev from mean are dropped and re-picked)
        # Shape of the output tensor: [layers[l], layers[l+1]]
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1  # 6 in this case
        # Initializing first input: mapping to [-1, 1]
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
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
        u = self.neural_net(x, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_xx - lambda_1 * tf.sin(x)  # f = u_xx - lambda_1 * sin(x) = 0
        return u, u_x, f

    def callback(self, loss, lambda_1):
        print('Loss: %.3e, l1: %.5f' % (loss, lambda_1))

    def train(self):  # one iteration: uses all training data from tf_dict and updates lambda
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0,
                   self.xe_tf: self.xe, self.u_xe_tf: self.u_xe,
                   self.xo_tf: self.xo, self.uo_tf: self.uo, self.xr_tf: self.xr}
        # feeding training examples during training and running the minimization Op of self.loss
        self.sess.run(self.train_op_Adam, tf_dict)
        # self.lambda_1 does not need tf_dict
        loss_value, lambda_1 = self.sess.run(
            [self.loss, self.lambda_1], tf_dict)
        return loss_value, lambda_1

    def predict(self, x):
        tf_dict = {self.xo_tf: x, self.xr_tf: x}  # no need for u
        # want to use the values in Session
        u, f = self.sess.run([self.uo_pred, self.f_pred], tf_dict)
        return u, f


if __name__ == "__main__":
    # u''(x) = lambda_1 * sin(lambda_2 * x), x in [0, pi] [u''(x) = 9 * sin(3x)]
    # NOTE: lambda defined in f/pde, not in u(x)
    # NOTE: to uniquely identify lambda, typically need boundary condition
    # u(0) = 0
    # u'(pi) = 1, u'(x) = - 3cos(3x)
    # NOTE: analytically given initial and boundary -> used in loss (identical to forward implementation)
    # analytical solution: u(x) = - sin(3x)
    ###########################
    # PART 1: initialization
    # 5-layer deep NN with 100 neurons/layer & hyperbolic tangent act. func.
    layers = [1, 20, 20, 20, 1]

    ###########################
    # PART 2：training data
    x0 = np.array([[0]])
    u0 = np.array([[0]])
    xe = np.array([[np.pi]])
    u_xe = np.array([[1]])
    # N_train = 4
    # x_train = np.reshape(np.linspace(0.3, np.pi-0.3, N_train), (-1, 1))
    # u_train = -1 * np.sin(x_train) # analytical solution
    # [[0.1][1.57079633][3.04159265]]
    # [[0.3] [1.14719755] [1.9943951] [2.84159265]]
    x_train = np.array([[2.1]])
    u_train = -1 * np.sin(x_train)  # analytical solution
    N_res = 20
    x_res = np.reshape(np.linspace(0, np.pi, N_res), (-1, 1))

    ###########################
    # PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(
        x0, u0, xe, u_xe, x_train, u_train, x_res, layers)

    N_t = 50
    xt = np.reshape(np.linspace(0, np.pi, N_t), (-1, 1))  # [[0] [pi/2] [pi]]
    ut = -1 * np.sin(xt)

    start_time = time.time()
    # Loss: 10^-3/-4 should be about good
    loss_values = []
    lambda_1s = []
    u_preds = []  # 4 * (50, 1)
    f_preds = []  # 4 * (50, 1)
    N_iter = 4000
    for i in range(N_iter):
        loss_value, lambda_1 = model.train()
        if i == 0:
            initial_loss = loss_value
            initial_lambda_1_value = lambda_1
        if (i+1) % 10 == 0:  # wouldn't plot in beginning, but plot at the end
            loss_values.append(loss_value)
            lambda_1s.append(lambda_1)
            print('Iter: %d, Loss: %.3e, Lambda_1: %.5f, Time: %.2f' %
                  (i+1, loss_value, lambda_1, time.time() - start_time))
        if (i+1) % 1000 == 0:  # wouldn't plot in beginning, but plot at the end
            # f = u_xx - tf.sin(x) # xt: (50, 1)
            u_pred, f_pred = model.predict(xt)
            u_preds.append(u_pred)
            f_preds.append(f_pred)
            # print('     u_pred: %.3e, f_pred: %.3e' % (u_pred , f_pred)
            # print('u_pred:', u_pred.shape) # (50, 1)
            # print('f_pred:', f_pred.shape) # (50, 1)
            # analytical solution: u(x) = -sin(x)
    # plot loss_values (loss-iteration)
    # 6 graphs
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)

    print("loss_values:", loss_values)  # [-3:]
    print('Training time: %.4f' % (time.time() - start_time))
    u_pred, f_pred = model.predict(xt)
    lambda_1_value = model.sess.run(model.lambda_1)
    print("Initial lambda_1: %.5f; Final lambda_1: %.5f" %
          (initial_lambda_1_value, lambda_1_value))
    # print("N_train:", N_train)
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weigths & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network
    # print("u_preds.shape", u_preds.shape) # (10, 50, 1)
    # print("f_preds.shape", f_preds.shape) # (10, 50, 1)
    print("x_train:", x_train)
    print("u_train:", u_train)
    ###########################
    # PART 4: calculating errors
    error_u = np.linalg.norm(u_pred - ut, 2) / np.linalg.norm(ut, 2)
    error_lambda_1 = np.abs(lambda_1_value - 1.0) * \
        100  # Ground truth: l1 = 1.0
    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1))

    ###########################
    # PART 5: Plotting
    # plot loss_values (loss vs. iteration) and lambda vs iteration
    # fig = plt.figure()
    # x_coords = 10*(np.array(range(len(loss_values))) + 1)
    # plt.semilogy(x_coords, loss_values)  # linear X axis, logarithmic y axis(log scaling on the y axis)
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Loss during Training")
    # init_tuple = (10,loss_values[0])
    # plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, textcoords='data', fontsize=7)
    # last_tuple = (N_iter,loss_values[-1])
    # plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=7)
    # fig.subplots_adjust(right=0.8)
    # plt.savefig('./figures/inverse_1d_loss.pdf')

    fig = plt.figure(figsize=(8, 4.8))
    plt.subplot(1, 2, 1)
    x_coords = 10*(np.array(range(len(loss_values))) + 1)
    # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.semilogy(x_coords, loss_values)
    plt.xlabel("Iteration", fontsize=6)
    plt.ylabel("Loss", fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title("Loss during Training", fontsize=8)
    init_tuple = (10, loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple,
                 textcoords='data', fontsize=6)
    last_tuple = (N_iter, loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple,
                 textcoords='data', fontsize=6)
    plt.subplot(1, 2, 2)
    plt.plot(x_coords, lambda_1s)
    plt.xlabel("Iteration", fontsize=6)
    plt.ylabel("Lambda", fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title("Lambda during Training", fontsize=8)
    init_tuple = (10, lambda_1s[0])
    plt.annotate('(%d, %.3f)' % init_tuple, xy=init_tuple,
                 textcoords='data', fontsize=6)
    last_tuple = (N_iter, lambda_1s[-1])
    plt.annotate('(%d, %.3f)' % last_tuple, xy=last_tuple,
                 textcoords='data', fontsize=6)
    fig.subplots_adjust(wspace=0.3)
    plt.savefig('./figures2/inverse_1d_loss.pdf')

    #
    # 2. plot u vs x (exact, prediction) # label training data
    print("u_preds.shape", u_preds.shape)
    print("f_preds.shape", f_preds.shape)
    fig = plt.figure()
    for i in range(u_preds.shape[0]):
        ax = plt.subplot(2, 2, i+1)
        exact_plot, = ax.plot(xt, ut, 'b-', label='Exact')  # tuple unpacking
        pred_plot, = ax.plot(xt, u_preds[i], 'r--', label='Prediction')
        ax.set_xlabel('$x$', fontsize=6)  # $: mathematical font like latex
        ax.set_ylabel('$u$', fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        ax.set_title('$Iteration = %d$' % ((i+1)*1000), fontsize=8)
    plt.figlegend(handles=(exact_plot, pred_plot), labels=(
        'Exact', 'Prediction'), loc='upper center', ncol=2, fontsize=7)  # from last subplot
    fig.subplots_adjust(wspace=0.3, hspace=0.51)
    plt.savefig('./figures2/inverse_1d.pdf')

    # with open('myfile.txt', 'w') as f:
    #     file1.writelines(L)

    # PDE table?

    # TODO: noisy data (to reflect practiacl situation)
    # noise = 0.01  # (to assume some error in observation data)
    # u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1]) # samples of specified size from standard normal
    # v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
    # retrain model
    # obtain error_lambda_1_noisy and print
