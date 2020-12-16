"""
@author: Maziar Raissi
"""
# Forward: given model/pde parameters λ -> u(t, x)

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy.io
import numpy as np
import tensorflow as tf
import sys, os
sys.path.insert(0, '../../Utilities/')  # for plotting

# from plotting import newfig, savefig

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, xb, yb, ul, ur, ub, ut, xf_grid, yf_grid, layers, lowerbound, upperbound):
        self.xb = xb
        self.yb = yb
        self.x0 = 0 * yb + lowerbound[0]  # left edge
        self.xe = 0 * yb + upperbound[0]  # right edge
        self.y0 = 0 * xb + lowerbound[1]  # bottom edge
        self.ye = 0 * xb + upperbound[1]  # top edge
        self.ul = ul
        self.ur = ur
        self.ub = ub
        self.ut = ut

        self.xf_grid = xf_grid
        self.yf_grid = yf_grid

        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        # shape: (N_b, 1)
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.yb_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.xe_tf = tf.placeholder(tf.float32, shape=[None, self.xe.shape[1]])
        self.y0_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.ye_tf = tf.placeholder(tf.float32, shape=[None, self.ye.shape[1]])
        self.ul_tf = tf.placeholder(tf.float32, shape=[None, self.ul.shape[1]])
        self.ur_tf = tf.placeholder(tf.float32, shape=[None, self.ur.shape[1]])
        self.ub_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.ut_tf = tf.placeholder(tf.float32, shape=[None, self.ut.shape[1]])
        # shape: (N_f * N_f, 1) because in net_all: X = tf.concat([x,y],1)
        self.xf_grid_tf = tf.placeholder(
            tf.float32, shape=[None, self.xf_grid.shape[1]])
        self.yf_grid_tf = tf.placeholder(
            tf.float32, shape=[None, self.yf_grid.shape[1]])

        # tf Graphs: u, f = net_all(x, y)
        self.ul_pred, _ = self.net_all(self.x0_tf, self.yb_tf)
        self.ur_pred, _ = self.net_all(self.xe_tf, self.yb_tf)
        self.ub_pred, _ = self.net_all(self.xb_tf, self.y0_tf)
        self.ut_pred, _ = self.net_all(self.xb_tf, self.ye_tf)

        # used in predict (only call net_all once)
        self.uf_pred, self.f_pred = self.net_all(
            self.xf_grid_tf, self.yf_grid_tf)

        # u_xx + u_yy = 0, x in [0, 1] & y in [0, 1]
        # u(0, y) = -y^2
        # u(1, y) =  1 - y^2 + 3y
        # u(x, 0) = x^2
        # u(x, 1) = x^2 - 1 + 3x
        # analytical solution: u(x, y) = x^2 - y^2 + 3xy
        # Loss: boundary + PDE
        self.loss = tf.reduce_mean(tf.square(self.ul_tf - self.ul_pred)) + \
            tf.reduce_mean(tf.square(self.ur_tf - self.ur_pred)) + \
            tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
            tf.reduce_mean(tf.square(self.ut_tf - self.ut_pred)) + \
            tf.reduce_mean(tf.square(self.f_pred))  # f_pred = u_xx + u_yy = 0
        # td.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        # return a minimization Op (a graph node that performs computation on tensors) -> updates weights and biases
        self.train_op_Adam = tf.train.AdamOptimizer().minimize(self.loss)

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
        H = 2.0*(X - self.lowerbound)/(self.upperbound - self.lowerbound) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # passing along networks
            # NOTE: H*W=(50, 20) + B(1, 20) -> tf does broadcasting: B becomes (50, 20)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  # passed 5 times in total
        return Y

    def net_all(self, x, y):
        X = tf.concat([x, y], 1)  # input
        # x = [[-0.5], [0.5]] # y = [[0], [1]]
        # [[-0.5, 0]
        #  [0.5,  1]]
        u = self.neural_net(X, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        f = u_xx + u_yy  # f = u_xx + u_yy = 0
        return u, f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):  # one iteration: uses all training data from tf_dict and updates weights and biases
        tf_dict = {self.x0_tf: self.x0, self.xe_tf: self.xe, self.xb_tf: self.xb,
                   self.y0_tf: self.y0, self.ye_tf: self.ye, self.yb_tf: self.yb,
                   self.ul_tf: self.ul, self.ur_tf: self.ur,
                   self.ub_tf: self.ub, self.ut_tf: self.ut,
                   self.xf_grid_tf: self.xf_grid, self.yf_grid_tf: self.yf_grid}
        # feeding training examples during training and running the minimization Op of self.loss
        self.sess.run(self.train_op_Adam, tf_dict)
        loss_value = self.sess.run(self.loss, tf_dict)
        return loss_value

    def predict(self, x, y):
        tf_dict = {self.xf_grid_tf: x, self.yf_grid_tf: y}
        u, f = self.sess.run([self.uf_pred, self.f_pred], tf_dict)
        # self.uf_pred, self.f_pred = self.net_all(self.xf_grid_tf, self.yf_grid_tf)
        return u, f


if __name__ == "__main__":
    # u_xx + u_yy = 0, x in [0, 1] & y in [0, 1]
    # u(0, y) = -y^2
    # u(1, y) =  1 - y^2 + 3y
    # u(x, 0) = x^2
    # u(x, 1) = x^2 - 1 + 3x
    # analytical solution: u(x, y) = x^2 - y^2 + 3xy
    ## u_x(0, y)
    # TODO: try a mix of edges + boundary
    # NOTE: need at least one edge to be u(x, y), otherwise solution have arbitrary constant

    # global settings for all subplots 
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8

    ###########################
    ## PART 1: setting parameters and getting accurate data for evaluation

    # Domain bounds
    lowerbound = np.array([0, 0])
    upperbound = np.array([1, 1])

    # 4-layer deep NN with 20 neurons/layer & hyperbolic tangent act. func.
    layers = [2, 50, 50, 50, 1]

    ###########################
    ## PART 2：setting training and testing data from full analytical solution for uniform grid

    # boundary data (t) ftom file -> used in training
    N_b = 20
    xb = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_b), (-1, 1)) # [[0] [0.5] [1]]
    yb = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_b), (-1, 1))
    ul = -1 * yb**2
    ur = 1 - yb**2 + 3 * yb
    ub = xb**2
    ut = xb**2 - 1 + 3 * xb

    # collocation points for enforcing f=0 from uniform grid -> used in training
    # NOTE: want PDE satisfied at positions arbitrarily close to boundary -> add boundary points to collocation points
    N_f = 20 
    xf = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_f), (-1, 1)) # [[0] [0.5] [1]] (N_f, 1)
    yf = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_f), (-1, 1)) # [[0] [0.5] [1]] (N_f, 1)
    xf_mesh, yf_mesh = np.meshgrid(xf, yf) # [[0 0.5 1] [0 0.5 1] [0 0.5 1]], [[0 0 0] [0.5 0.5 0.5] [1 1 1]]
    # shape: (N_f * N_f, 1) because in net_all: X = tf.concat([x,y],1)
    xf_grid = np.reshape(xf_mesh.flatten(), (-1, 1)) # [[0] [0.5] [1] [0] [0.5] [1] [0] [0.5] [1]]
    yf_grid = np.reshape(yf_mesh.flatten(), (-1, 1)) # [[0] [0] [0] [0.5] [0.5] [0.5] [1] [1] [1]]

    # testing data
    N_test = 50  # NOTE: different from collocation points
    xtest = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_test), (-1, 1))  # (N_test, 1)
    ytest = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_test), (-1, 1))  # (N_test, 1)
    xtest_mesh, ytest_mesh = np.meshgrid(xtest, ytest)  # (N_test, N_test)
    xtest_grid = np.reshape(xtest_mesh.flatten(),(-1, 1))  # (N_test * N_test, 1)
    ytest_grid = np.reshape(ytest_mesh.flatten(), (-1, 1))  # (N_test * N_test, 1)
    u_test = xtest_grid**2 - ytest_grid**2 + 3 * xtest_grid * ytest_grid  # (N_test * N_test, 1)

    ###########################
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(xb, yb, ul, ur, ub, ut,xf_grid, yf_grid, layers, lowerbound, upperbound)

    start_time = time.time()
    dirpath = f'./2d/forward_2d_figures/{start_time}' # where figures are stored
    os.mkdir(dirpath)
    # Note: loss around 10^-3/-4 should be about good
    loss_values, u_preds, f_preds = ([] for i in range(3))
    N_iter = 9000
    loss_value_step = 10
    pred_step = 1000
    for i in range(N_iter):
        loss_value = model.train()
        if (i+1) % loss_value_step == 0:  # wouldn't plot in beginning, but plot at the end
            loss_values.append(loss_value)
            print('Iter: %d, Loss: %.3e, Time: %.2f' %
                  (i+1, loss_value, time.time() - start_time))
        if (i+1) % pred_step == 0:  # start with i=999 and end with i=8999 (last iter)
            # f = u_xx - tf.sin(x) # xt: (50, 1)
            u_pred, f_pred = model.predict(xtest_grid, ytest_grid)
            u_preds.append(u_pred)  # (N_test * N_test, 1)
            f_preds.append(f_pred)  # (N_test * N_test, 1)
            fig = plt.figure()
            ax = plt.subplot(1,2,1)
            cset1 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(u_test, (N_test, N_test)), levels=30, cmap='winter')
            plt.gca().set(xlabel='$x$', ylabel='$y$', title='Exact')
            ax = plt.subplot(1,2,2)
            cset2 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(u_pred, (N_test, N_test)), levels=30, cmap='winter')
            plt.gca().set(xlabel='$x$', ylabel='$y$', title='Prediction')
            fig.subplots_adjust(right=0.8)
            cbar_ax = plt.axes([0.85, 0.12, 0.05, 0.76]) # Axes into which the colorbar will be drawn
            fig.colorbar(cset2, cax=cbar_ax)
            plt.savefig(f'{dirpath}/forward_2d_contour_iter{i+1}.pdf')
 
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)

    # print("loss_values:", loss_values)
    print('Training time: %.4f' % (time.time() - start_time))
    # print('final u_pred:', u_pred)
    # print('final f_pred:', f_pred)
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weigths & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network

   ###########################
    ## PART 4: calculating errors
    error_u = np.linalg.norm(u_pred - u_test, 2) / np.linalg.norm(u_test, 2)
    print('Error u: %e' % (error_u))  # 5.558028e-04

    # print('final f_pred:', f_pred)

    ###########################
    ## PART 5: Plotting
    # 1. Exact vs Prediction of u (contour) during training
    # 2. loss vs. iteration + MSE of u_pred and u_test vs. iteration
    fig = plt.figure()
    plt.subplot(2,1,1)
    x_coords = loss_value_step * (np.array(range(len(loss_values))) + 1)
    plt.semilogy(x_coords, loss_values) # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.gca().set(xlabel='Iteration', ylabel='Loss', title='Loss during Training')
    init_tuple = (loss_value_step, loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, textcoords='data', fontsize=7)
    last_tuple = (N_iter, loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=7)
    fig.subplots_adjust(right=0.86)
    # NOTE: Oscillation: actually very small nummerical difference because of small y scale
    # 1. overshoot (fixed -> decaying learning rate)
    # 2. Adam: gradient descent + momentum (sometime parameter change makes the loss go up)
    
    plt.subplot(2,1,2)
    u_mses = [] # 2d array [ [mse1] [mse2] [mse3]]
    for u_pred in u_preds:  # (N_test * N_test, 1)
        u_mses.append(((u_pred - u_test)**2).mean(axis=0)) # append [mse for that iteration]
    x_coords = pred_step * (np.array(range(len(u_preds))) + 1)
    u_mses = np.array(u_mses)
    annots = list(zip(x_coords, u_mses.flatten())) # [(1000, 4.748), (2000, 9.394)]
    print("annot:", annots)
    plt.semilogy(x_coords, u_mses, '.-')
    for annot in annots:
        plt.annotate('(%d, %.3e)' % annot, xy=annot, textcoords='data', fontsize=6)
    plt.gca().set(xlabel='Iteration', ylabel='MSE of u', title='MSE of u during Training')

    fig.subplots_adjust(hspace=0.4)
    plt.savefig(f'{dirpath}/forward_2d_loss.pdf')

    # 3. plot u vs (x, y) and |u_pred-u_test| vs (x,y): contour
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig, width_ratios=[6, 6, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    cset1 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(u_test, (N_test, N_test)), levels=30, cmap='winter')
    plt.gca().set(xlabel='$x$', ylabel='$y$', title='Exact') # $: mathematical font like latex

    ax = fig.add_subplot(gs[0, 1])
    cset2 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(u_pred, (N_test, N_test)), levels=30, cmap='winter')
    plt.gca().set(xlabel='$x$', ylabel='$y$', title='Prediction') # $: mathematical font like latex

    ax = fig.add_subplot(gs[0, 2])
    fig.colorbar(cset2, cax=ax)

    ax = fig.add_subplot(gs[1, 0:2])
    cset3 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(np.abs(u_pred-u_test), (N_test, N_test)), levels=30, cmap='autumn')
    plt.gca().set(xlabel='$x$', ylabel='$y$', title='|Prediction - Exact|')
    ax = fig.add_subplot(gs[1, 2])
    fig.colorbar(cset3, cax=ax)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'{dirpath}/forward_2d_contour.pdf')

    # 4. plot u vs (x, y): surface
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xtest_mesh, ytest_mesh, np.reshape(u_test, (N_test, N_test)), label='Exact', cmap='winter')  # Data values as 2D arrays: (N_test * N_test, 1)
    plt.gca().set(xlabel='$x$', ylabel='$y$', zlabel='$u$', title='Exact')
    # ax.set_zlabel('$u$', fontsize=6) 
    ax.tick_params(labelsize=6)

    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot_surface(xtest_mesh, ytest_mesh, np.reshape(u_pred, (N_test, N_test)), label='Prediction', cmap='winter')  # Data values as 2D arrays: (N_test * N_test, 1)
    plt.gca().set(xlabel='$x$', ylabel='$y$', zlabel='$u$', title='Prediction')
    # ax.set_zlabel('$u$', fontsize=6)
    ax.tick_params(labelsize=6)

    plt.savefig(f'{dirpath}/forward_2d_surface.pdf')