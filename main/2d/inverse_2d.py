"""
@author: Maziar Raissi
"""
# Inverse: given observed data of u(t, x) -> model/pde parameters λ

import time, sys, os, json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy.io
import numpy as np
import tensorflow as tf
sys.path.insert(0, '../../Utilities/')  # for plotting

# from plotting import newfig, savefig

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, xb, yb, x0, xe, y0, ye, ul, ur, ub, ut, xo_grid, yo_grid, uo, xf_grid, yf_grid, lambda0, layers, lowerbound, upperbound):
        self.xb = xb
        self.yb = yb
        self.x0 = x0
        self.xe = xe
        self.y0 = y0
        self.ye = ye
        self.ul = ul
        self.ur = ur
        self.ub = ub
        self.ut = ut

        self.xo_grid = xo_grid
        self.yo_grid = yo_grid
        self.uo = uo
        self.xf_grid = xf_grid
        self.yf_grid = yf_grid

        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.lambda_1 = tf.Variable([lambda0[0]], dtype=tf.float32)

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
        # shape: (N_observed * N_observed, 1)/(N_f * N_f, 1) because in net_all: X = tf.concat([x,y],1)
        self.xo_grid_tf = tf.placeholder(tf.float32, shape=[None, self.xo_grid.shape[1]])
        self.yo_grid_tf = tf.placeholder(tf.float32, shape=[None, self.yo_grid.shape[1]])
        self.uo_tf = tf.placeholder(tf.float32, shape=[None, self.uo.shape[1]])
        self.xf_grid_tf = tf.placeholder(tf.float32, shape=[None, self.xf_grid.shape[1]])
        self.yf_grid_tf = tf.placeholder(tf.float32, shape=[None, self.yf_grid.shape[1]])

        self.lr_tf = tf.placeholder(tf.float32)

        # tf Graphs: u, f = net_all(x, y)
        self.ul_pred, _ = self.net_all(self.x0_tf, self.yb_tf)
        self.ur_pred, _ = self.net_all(self.xe_tf, self.yb_tf)
        self.ub_pred, _ = self.net_all(self.xb_tf, self.y0_tf)
        self.ut_pred, _ = self.net_all(self.xb_tf, self.ye_tf)
        self.uo_pred, _ = self.net_all(self.xo_grid_tf, self.yo_grid_tf)
        # used in predict (only call net_all once)
        self.uf_pred, self.f_pred = self.net_all(self.xf_grid_tf, self.yf_grid_tf)

        # lambda_1 * u_xx + u_yy = 0, x in [0, 1] & y in [0, 1] -> 5 * u_xx + u_yy = 0
        # u(0, y) = -5 * y^2            ## left
        # u(1, y) =  1 - 5 * y^2 + 3y   ## right
        # u(x, 0) = x^2                 ## bottom
        # u(x, 1) = x^2 - 5 + 3x        ## top
        # analytical solution: u(x, y) = x^2 - 5 * y^2 + 3xy
        # Loss: boundary + observed data + PDE
        self.loss = tf.reduce_mean(tf.square(self.ul_tf - self.ul_pred)) + \
                    tf.reduce_mean(tf.square(self.ur_tf - self.ur_pred)) + \
                    tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                    tf.reduce_mean(tf.square(self.ut_tf - self.ut_pred)) + \
                    tf.reduce_mean(tf.square(self.uo_tf - self.uo_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))  # f = lambda_1 * u_xx + u_yy = 0
        # tf.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        # return a minimization Op (a graph node that performs computation on tensors) -> updates weights, biases, lambdas
        self.train_op_Adam = tf.train.AdamOptimizer(learning_rate = self.lr_tf).minimize(self.loss)

        # tf session: initiates a tf Graph (defines computations) that processes tensors through operations + allocates resources + holds intermediate values
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
            biases.append(tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32))  # all zeros
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
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lowerbound)/(self.upperbound - self.lowerbound) - 1 # Initializing first input: mapping to [-1, 1]
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
        lambda_1 = self.lambda_1
        u = self.neural_net(X, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        f = lambda_1 * u_xx + u_yy  # f = lambda_1 * u_xx + u_yy = 0
        return u, f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, lr):  # one iteration: uses all training data from tf_dict and updates weights, biases, lambdas
        tf_dict = { self.ul_tf: self.ul, self.x0_tf: self.x0, self.yb_tf: self.yb,
                    self.ur_tf: self.ur, self.xe_tf: self.xe,
                    self.ub_tf: self.ub, self.xb_tf: self.xb, self.y0_tf: self.y0,
                    self.ut_tf: self.ut, self.ye_tf: self.ye,
                    self.uo_tf: self.uo, self.xo_grid_tf: self.xo_grid, self.yo_grid_tf: self.yo_grid,
                    self.xf_grid_tf: self.xf_grid, self.yf_grid_tf: self.yf_grid,
                    self.lr_tf: lr}
        # feeding training examples during training and running the minimization Op of self.loss
        self.sess.run(self.train_op_Adam, tf_dict)
        loss_value, lambda_1 = self.sess.run([self.loss, self.lambda_1], tf_dict)
        return loss_value, lambda_1

    def predict(self, x_grid, y_grid): # tf.concat([x, y], 1)
        tf_dict = {self.xf_grid_tf: x_grid, self.yf_grid_tf: y_grid}
        u, f = self.sess.run([self.uf_pred, self.f_pred], tf_dict)
        # self.uf_pred, self.f_pred = self.net_all(self.xf_grid_tf, self.yf_grid_tf)
        return u, f

def countourPlot(xtest_mesh, ytest_mesh, u_test, u_pred, N_test, figName):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig, width_ratios=[6, 6, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    cset1 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(u_test, (N_test, N_test)), levels=30, cmap='winter')
    plt.gca().set(xlabel='$x$', ylabel='$y$', title='Exact') # $: mathematical font like latex

    ax = fig.add_subplot(gs[0, 1])
    cset2 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(u_pred, (N_test, N_test)), levels=30, cmap='winter')
    plt.gca().set(xlabel='$x$', ylabel='$y$', title='Prediction')

    ax = fig.add_subplot(gs[0, 2])
    fig.colorbar(cset2, cax=ax)

    ax = fig.add_subplot(gs[1, 0:2])
    cset3 = ax.contourf(xtest_mesh, ytest_mesh, np.reshape(np.abs(u_pred-u_test), (N_test, N_test)), levels=30, cmap='autumn')
    plt.gca().set(xlabel='$x$', ylabel='$y$', title='|Prediction - Exact|')
    ax = fig.add_subplot(gs[1, 2])
    fig.colorbar(cset3, cax=ax)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'{dirpath}/{figName}.pdf')
    plt.close(fig)

def axisToGrid(x, y): # [[0] [0.5] [1]] (N, 1)
    x_mesh, y_mesh = np.meshgrid(x, y) # [[0 0.5 1] [0 0.5 1] [0 0.5 1]], [[0 0 0] [0.5 0.5 0.5] [1 1 1]] (N, N)
    x_grid = np.reshape(x_mesh.flatten(), (-1, 1)) # [[0] [0.5] [1] [0] [0.5] [1] [0] [0.5] [1]] # (N * N, 1)
    y_grid = np.reshape(y_mesh.flatten(), (-1, 1)) # [[0] [0] [0] [0.5] [0.5] [0.5] [1] [1] [1]] # (N * N, 1)
    return x_mesh, y_mesh, x_grid, y_grid # net_all: X = tf.concat([x,y],1)

def analyticalU(x, y, lambda_gt):
    # NOTE: lambda defined in terms of f not u, but interchangeable for this problem
    u = x**2 - lambda_gt[0] * y**2 + 3 * x * y
    return u

if __name__ == "__main__":
    # lambda_1 * u_xx + u_yy = 0, x in [0, 1], y in [0, 1] -> 5 * u_xx + u_yy = 0
    # u(0, y) = -5 * y^2            ## left
    # u(1, y) =  1 - 5 * y^2 + 3y   ## right
    # u(x, 0) = x^2                 ## bottom
    # u(x, 1) = x^2 - 5 + 3x        ## top
    # analytical solution: u(x, y) = x^2 - 5(lambda_1) * y^2 + 3xy

    # global settings for all subplots 
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8

    ###########################
    ## PART 1: setting parameters and getting accurate data for evaluation

    # 4-layer deep NN with 50 neurons/layer & hyperbolic tangent act. func.
    layers = [2, 30, 30, 30, 1]

    # Domain bounds
    lowerbound = np.array([0, 0])
    upperbound = np.array([1, 1])

    ###########################
    ## PART 2：setting training and testing data from full analytical solution for uniform grid

    # true and initial value for lambda
    lambda_gt = np.array([2])
    lambda0 = np.array([1.8])
    
    # boundary condition
    N_b = 20
    xb = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_b), (-1, 1)) # [[0] [0.5] [1]]
    yb = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_b), (-1, 1))
    x0 = 0 * yb + lowerbound[0] # left edge
    xe = 0 * yb + upperbound[0] # right edge
    y0 = 0 * xb + lowerbound[1] # bottom edge
    ye = 0 * xb + upperbound[1] # top edge
    ul = -lambda_gt[0] * yb**2 # u(0, y)
    ur = 1 - lambda_gt[0] * yb**2 + 3 * yb # u(1, y)
    ub = xb**2 # u(x, 0)
    ut = xb**2 - lambda_gt[0] + 3 * xb # u(x, 1)

    # observed u based on analytical solution
    N_observed = 3 # along one axis
    xo = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_observed+2), (-1, 1))[1:-1] # (N_observed, 1)
    yo = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_observed+2), (-1, 1))[1:-1] # (N_observed, 1)
    _, _, xo_grid, yo_grid = axisToGrid(xo, yo) # (N_observed * N_observed, 1)
    uo = analyticalU(xo_grid, yo_grid, lambda_gt) # (N_observed * N_observed, 1)

    # collocation points for enforcing f=0 from uniform grid
    # NOTE: want PDE satisfied at positions arbitrarily close to boundary -> include boundary points in collocation points
    N_f = 50 # along one axis
    xf = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_f), (-1, 1)) # (N_f, 1)
    yf = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_f), (-1, 1)) # (N_f, 1)
    _, _, xf_grid, yf_grid = axisToGrid(xf, yf) # (N_f * N_f, 1)

    # testing data
    N_test = 51  # NOTE: different from collocation points
    xtest = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_test), (-1, 1))  # (N_test, 1)
    ytest = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_test), (-1, 1))  # (N_test, 1)
    xtest_mesh, ytest_mesh, xtest_grid, ytest_grid = axisToGrid(xtest, ytest) # (N_test, N_test), (N_test * N_test, 1)
    u_test = analyticalU(xtest_grid, ytest_grid, lambda_gt) # (N_test * N_test, 1)


    ###########################
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(xb, yb, x0, xe, y0, ye, ul, ur, ub, ut, xo_grid, yo_grid, uo, xf_grid, yf_grid, lambda0, layers, lowerbound, upperbound)

    start_time = time.time()
    dirpath = f'./2d/inverse_2d_figures/{start_time}' # where figures are stored
    os.mkdir(dirpath)
    # Note: loss around 10^-3/-4 should be about good
    loss_values, u_preds, f_preds = ([] for i in range(3))
    lambda_1s = [lambda0[0]] # 1d - initial
    N_iter = 50000
    loss_value_step = 50
    pred_step = 5000
    for i in range(N_iter):
        lr = 10**-3 * 2**(-i/10000)
        loss_value, lambda_1= model.train(lr) # from last iteration
        if (i+1) % loss_value_step == 0:  # start with i=9 and end with i=8999 (last iter)
            loss_values.append(loss_value)
            lambda_1s.append(lambda_1)
            print('Iter: %d, Loss: %.3e, Lambda_1: %.5f, Time: %.2f, Learning Rate: %.5f' % (i+1, loss_value, lambda_1, time.time() - start_time, lr))
            # TODO: mse perhaps
        if (i+1) % pred_step == 0:  # start with i=999 and end with i=8999 (last iter)
            u_pred, f_pred = model.predict(xtest_grid, ytest_grid)
            u_preds.append(u_pred)  # (N_test * N_test, 1)
            f_preds.append(f_pred)  # (N_test * N_test, 1)
            ## Plotting 1. u (Exact, Preidction) vs (x,y) and |u_pred-u_test| vs (x,y): contour
            countourPlot(xtest_mesh, ytest_mesh, u_test, u_pred, N_test, f'inverse_2d_contour_iter{i+1}')
    training_time = time.time() - start_time
    # print('Training time: %.4f' % (time))
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)
    u_pred, f_pred = model.predict(xtest_grid, ytest_grid)
    print("Initial lambda_1: %.5f, Final lambda_1: %.5f" % (lambda0[0], lambda_1))
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weights & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network

   ###########################
    ## PART 4: calculating errors
    error_u = np.linalg.norm(u_pred - u_test, 2) / np.linalg.norm(u_test, 2) # scalar
    error_lambda_1 = np.abs(lambda_1 - lambda_gt[0])/lambda_gt[0] * 100  # 1d np array
    print('Error u: %e | Error lambda_1: %.5f%%' % (error_u, error_lambda_1))

    ###########################
    ## PART 5: Plotting

    # 2. loss vs. iteration, lambda_1 vs iteration, MSE between u_pred and u_test vs. iteration
    fig = plt.figure(figsize=(10, 4.8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
    fig.add_subplot(gs[0, 0])
    x_coords = loss_value_step * (np.array(range(len(loss_values))) + 1)
    plt.semilogy(x_coords, loss_values) # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.gca().set(xlabel='Iteration', ylabel='Loss', title='Loss during Training')
    init_tuple = (loss_value_step, loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, textcoords='data', fontsize=6)
    last_tuple = (N_iter, loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=6)
    fig.subplots_adjust(right=0.86)
    # NOTE: Oscillation: actually very small nummerical difference because of small y scale
        # 1. overshoot (fixed -> decaying learning rate)
        # 2. Adam: gradient descent + momentum (sometime parameter change makes the loss go up)
    
    fig.add_subplot(gs[0, 1])
    x_coords = loss_value_step * (np.array(range(len(lambda_1s)))) # has an additional entry than loss_values: initial
    plt.plot(x_coords, lambda_1s)
    plt.gca().set(xlabel="Iteration", ylabel="Lambda_1", title="Lambda_1 during Training")
    init_tuple = (0, lambda_1s[0])
    plt.annotate('(%d, %.3f)' % init_tuple, xy=init_tuple,textcoords='data', fontsize=6)
    last_tuple = (N_iter, lambda_1s[-1])
    plt.annotate('(%d, %.3f)' % last_tuple, xy=last_tuple, textcoords='data', fontsize=6)
    
    x_coords = pred_step * (np.array(range(len(u_preds))) + 1)
    fig.add_subplot(gs[1, 0])
    u_mses = [] # 2d array [[mse1] [mse2] [mse3]]
    for u_pred in u_preds:
        u_mses.append(((u_pred - u_test)**2).mean(axis=0)) # append [mse for that iteration]
    u_mses = np.array(u_mses)
    annots = list(zip(x_coords, u_mses.flatten())) # [(1000, 4.748), (2000, 9.394)]
    plt.semilogy(x_coords, u_mses, '.-')
    for annot in annots[0:1] + annots[-1:]: # first and last
        plt.annotate('(%d, %.3e)' % annot, xy=annot, textcoords='data', fontsize=6)
    plt.gca().set(xlabel='Iteration', ylabel='MSE of u', title='MSE of u during Training')

    fig.subplots_adjust(wspace=0.3, hspace=0.51)
    plt.savefig(f'{dirpath}/inverse_2d_loss.pdf')
    plt.close(fig)

    # TODO: change graph (remove res_means)
    # TODO: train til convergence

    # 3. u vs (x, y): surface
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

    plt.savefig(f'{dirpath}/inverse_2d_surface.pdf')
    plt.close(fig)


    infoDict = {
        'problem':{
            'pde form': 'lambda_1 * u_xx + u_yy = 0)',
            'lambda_1 gt': float(lambda_gt[0]),
            'boundary (x)': [float(lowerbound[0]),float(upperbound[0])], 
            'boundary (y)': [float(lowerbound[1]),float(upperbound[1])], 
            'boundary condition': 'u(0, y) = -5 * y^2, u(1, y) =  1 - 5 * y^2 + 3y, u(x, 0) = x^2, u(x, 1) = x^2 - 5 + 3x',
            'analytical solution': 'u(x, y) = x^2 - 5 * y^2 + 3xy'
        },
        'model':{
            'layers': str(layers),
            'training iteration': N_iter,
            'initial lambda_1': float(lambda0[0]),
            'final lambda_1': float(lambda_1[0]),
            'error_lambda_1 percentage': float(error_lambda_1[0]),
            'error_u': error_u,
            'training_time': training_time
        },
        'training data':{
            'N_b': N_b,
            'xb': str(xb),
            'yb': str(yb),
            'left boundary u': str(ul),
            'right boundary u': str(ur),
            'bottomm boundary u': str(ub),
            'top boundary u': str(ut),
            'N_observed': N_observed,
            'xo': str(xo),
            'yo': str(yo),
            'uo': str(uo),
            'N_f': N_f,
        },
        'testing data':{
            'N_test': N_test
        }
    }
    with open(f'{dirpath}/info.json', 'w') as f:
        json.dump(infoDict, f, indent=4)
    with open(f'{dirpath}/lambda_1s.json', 'w') as f:
        json.dump(lambda_1.tolist(), f)
    with open(f'{dirpath}/loss_values.json', 'w') as f:
        json.dump(loss_values.tolist(), f)