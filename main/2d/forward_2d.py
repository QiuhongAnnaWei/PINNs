# Forward: given model/pde parameters λ -> u(t, x)

import time, sys, os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
# from plotting import newfig, savefig
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from pyDOE import lhs
# from scipy.interpolate import griddata
# import scipy.io

sys.path.insert(0, '../../Utilities/')  # for plotting

# from plotting import newfig, savefig

# np.random.seed(1234)
# tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, xb, yb, x0, xe, y0, ye, boundaryU, boundaryU_, xf_grid, yf_grid, layers, lowerbound, upperbound, mix):
        self.mix = mix

        self.xb = xb
        self.yb = yb
        self.x0 = x0
        self.xe = xe
        self.y0 = y0
        self.ye = ye
        self.ul = boundaryU[0]
        self.ur = boundaryU[1]
        self.ub = boundaryU[2]
        self.ut = boundaryU[3]
        if self.mix:
            self.ul_x = boundaryU_[0]
            self.ur_x = boundaryU_[1]
            self.ub_y = boundaryU_[2]
            self.ut_y = boundaryU_[3]

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
        if self.mix:
            self.ul_x_tf = tf.placeholder(tf.float32, shape=[None, self.ul_x.shape[1]])
            self.ur_x_tf = tf.placeholder(tf.float32, shape=[None, self.ur_x.shape[1]])
            self.ub_y_tf = tf.placeholder(tf.float32, shape=[None, self.ub_y.shape[1]])
            self.ut_y_tf = tf.placeholder(tf.float32, shape=[None, self.ut_y.shape[1]])
        # shape: (N_f * N_f, 1) because in net_all: X = tf.concat([x,y],1)
        self.xf_grid_tf = tf.placeholder(tf.float32, shape=[None, self.xf_grid.shape[1]])
        self.yf_grid_tf = tf.placeholder(tf.float32, shape=[None, self.yf_grid.shape[1]])

        self.lr_tf = tf.placeholder(tf.float32)

        # tf Graphs: u, u_x, u_y, f = net_all(x, y)
        self.ul_pred, self.ul_x_pred, _, _ = self.net_all(self.x0_tf, self.yb_tf)
        self.ur_pred, self.ur_x_pred, _, _ = self.net_all(self.xe_tf, self.yb_tf)
        self.ub_pred, _, self.ub_y_pred, _ = self.net_all(self.xb_tf, self.y0_tf)
        self.ut_pred, _, self.ut_y_pred, _ = self.net_all(self.xb_tf, self.ye_tf)   
        # used in predict (only call net_all once)
        self.uf_pred, _, _, self.f_pred = self.net_all(self.xf_grid_tf, self.yf_grid_tf)

        # Loss: boundary(u, u_x, u_y) + PDE (f = u_xx + u_yy = 0)
        if not self.mix: # purely u for boundary condition
            self.loss = tf.reduce_mean(tf.square(self.ul_tf - self.ul_pred)) + \
                        tf.reduce_mean(tf.square(self.ur_tf - self.ur_pred)) + \
                        tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                        tf.reduce_mean(tf.square(self.ut_tf - self.ut_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred)) 
        else: # mix of u and u_x, u_y for boundary condition
            self.loss = tf.reduce_mean(tf.square(self.ul_x_tf - self.ul_x_pred)) + \
                        tf.reduce_mean(tf.square(self.ur_tf - self.ur_pred)) + \
                        tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                        tf.reduce_mean(tf.square(self.ut_y_tf - self.ut_y_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred))
        # tf.reduce_mean: computes the mean of elements across dimensions of a tensor

        # Optimizers:
        # return a minimization Op (a graph node that performs computation on tensors) -> updates weights and biases
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
        u = self.neural_net(X, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        f = u_xx + u_yy  # f = u_xx + u_yy = 0
        return u, u_x, u_y, f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, lr):  # one iteration: uses all training data from tf_dict and updates weights and biases
        tf_dict = { self.x0_tf: self.x0, self.xe_tf: self.xe, self.xb_tf: self.xb,
                    self.y0_tf: self.y0, self.ye_tf: self.ye, self.yb_tf: self.yb,
                    self.ul_tf: self.ul, self.ur_tf: self.ur, self.ub_tf: self.ub, self.ut_tf: self.ut,
                    self.xf_grid_tf: self.xf_grid, self.yf_grid_tf: self.yf_grid,
                    self.lr_tf: lr}
        if self.mix:
            tf_dict.update({
                self.ul_x_tf: self.ul_x, self.ur_x_tf: self.ur_x,
                self.ub_y_tf: self.ub_y, self.ut_y_tf: self.ut_y
            })
        # feeding training examples during training and running the minimization Op of self.loss
        self.sess.run(self.train_op_Adam, tf_dict)
        loss_value = self.sess.run(self.loss, tf_dict)
        return loss_value

    def predict(self, x_grid, y_grid): # tf.concat([x, y], 1)
        tf_dict = {self.xf_grid_tf: x_grid, self.yf_grid_tf: y_grid}
        u, f = self.sess.run([self.uf_pred, self.f_pred], tf_dict)
        return u, f


def contourPlot(xtest_mesh, ytest_mesh, u_test, u_pred, N_test, i):
    fig = plt.figure(figsize=(6.0, 5.3))
    gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig, width_ratios=[6, 6, 0.6], height_ratios=[1, 1], wspace=0.41, hspace=0.33)
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

    plt.suptitle(f'Snapshot at Iteration = {i+1}')
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.08, top=0.90)
    plt.savefig(f'{dirpath}/forward_2d_contour_iter{i+1}_new.pdf')
    plt.close(fig)
    with open(f'{dirpath}/forward_2d_contour_upred_iter{i+1}.json', 'w') as f:
        json.dump(u_pred.tolist(), f)

def axisToGrid(x, y): # [[0] [0.5] [1]] (N, 1)
    x_mesh, y_mesh = np.meshgrid(x, y) # [[0 0.5 1] [0 0.5 1] [0 0.5 1]], [[0 0 0] [0.5 0.5 0.5] [1 1 1]] (N, N)
    x_grid = np.reshape(x_mesh.flatten(), (-1, 1)) # [[0] [0.5] [1] [0] [0.5] [1] [0] [0.5] [1]] # (N * N, 1)
    y_grid = np.reshape(y_mesh.flatten(), (-1, 1)) # [[0] [0] [0] [0.5] [0.5] [0.5] [1] [1] [1]] # (N * N, 1)
    return x_mesh, y_mesh, x_grid, y_grid # net_all: X = tf.concat([x,y],1)

if __name__ == "__main__":
    # u_xx + u_yy = 0, x in [0, 1], y in [0, 1]
    # u(0, y) = -y^2                    ## left (u)
    # u(1, y) = 1 - y^2 + 3y            ## right (u)
    # u(x, 0) = x^2                     ## bottom (u)
    # u(x, 1) = x^2 - 1 + 3x            ## top (u)
    # u_x(0, y) = 2x + 3y = 3y          ## left (du/dx)
    # u_x(1, y) = 2x + 3y = 2 + 3y      ## right (du/dx)
    # u_y(x, 0) = -2y + 3x = 3x         ## bottom (du/dy)
    # u_y(x, 1) = -2y + 3x = -2 + 3x    ## top (du/dy)
    # analytical solution: u(x, y) = x^2 - y^2 + 3xy
    # NOTE: du/dn (normal direction) for boundary condition:
        # 1) additional information
        # 2) makes sense this way: u=temperature, fixed boundary temperatue, du/dn indicates influx/outflux
    # NOTE: need at least one edge to be u(x, y), otherwise solution have arbitrary constant
    # NOTE: Boundary condition can have order > 1

    ###########################
    ## PART 1: setting parameters and getting accurate data for evaluation

    # 4-layer deep NN with 20 neurons/layer & hyperbolic tangent act. func.
    layers = [2, 50, 50, 50, 1]
    mix = True # mix of boundary conditions (u, u_x, u_y)

    # Domain bounds
    lowerbound = np.array([0, 0])
    upperbound = np.array([1, 1])

    ###########################
    ## PART 2：setting training and testing data from full analytical solution for uniform grid

    # boundary condition
    N_b = 20
    xb = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_b), (-1, 1)) # [[0] [0.5] [1]] (N_b, 1)
    yb = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_b), (-1, 1))
    x0 = 0 * yb + lowerbound[0] # left edge # [[0] [0] [0]]
    xe = 0 * yb + upperbound[0] # right edge
    y0 = 0 * xb + lowerbound[1] # bottom edge
    ye = 0 * xb + upperbound[1] # top edge
    ul = -1 * yb**2             # u(0, y)
    ur = 1 - yb**2 + 3 * yb     # u(1, y)
    ub = xb**2                  # u(x, 0)
    ut = xb**2 - 1 + 3 * xb     # u(x, 1)
    ul_x = 3 * yb               # u_x(0, y)
    ur_x = 2 + 3 * yb           # u_x(1, y)
    ub_y = 3 * xb               # u_y(x, 0)
    ut_y = -2 + 3 * xb          # u_y(x, 1)

    # collocation points for enforcing f=0 from uniform grid
    # NOTE: want PDE satisfied at positions arbitrarily close to boundary -> include boundary points in collocation points
    # NOTE: Generally, want interval of training point < smallest characteristic of solution (fluctuation) (dense enough to capture all landscape within domain)
    # NOTE: To estimate the density: can estimate fluctuation frequency from f (known), geometry (sharp region higher frequency), prior knowledge

    N_f = 30 # along one axis
    xf = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_f), (-1, 1)) # (N_f, 1)
    yf = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_f), (-1, 1)) # (N_f, 1)
    _, _, xf_grid, yf_grid = axisToGrid(xf, yf) # (N_f * N_f, 1)

    # testing data
    N_test = 50  # NOTE: different from collocation points
    xtest = np.reshape(np.linspace(lowerbound[0], upperbound[0], N_test), (-1, 1))  # (N_test, 1)
    ytest = np.reshape(np.linspace(lowerbound[1], upperbound[1], N_test), (-1, 1))  # (N_test, 1)
    xtest_mesh, ytest_mesh, xtest_grid, ytest_grid = axisToGrid(xtest, ytest) # # (N_test, N_test), (N_test * N_test, 1)
    u_test = xtest_grid**2 - ytest_grid**2 + 3 * xtest_grid * ytest_grid  # (N_test * N_test, 1)

    ###########################
    ## PART 3: forming the network, training, predicting
    model = PhysicsInformedNN(xb, yb, x0, xe, y0, ye, [ul, ur, ub, ut], [ul_x, ur_x, ub_y, ut_y], xf_grid, yf_grid, layers, lowerbound, upperbound, mix)

    start_time = time.time()

    # settings for plots 
    dirpath = f'./main/2d/forward_2d_figures/{start_time}' # where figures are stored
    os.mkdir(dirpath)
    ticksize = 8.5
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    plt.rcParams['axes.labelsize'] = 9.5
    plt.rcParams['axes.titlesize'] = 10.5
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['legend.handlelength'] = 0.4
    annotatesize = 9.5
    # Plot 1. Boundary Point, Collocation Point
    fig = plt.figure(figsize=(4.2, 2.9))
    bc, = plt.plot(np.concatenate((x0,xe,xb,xb)), np.concatenate((yb,yb,y0,ye)), 'H', c='#ffa96b', label = 'Boundary Point', clip_on=False)
    cp, = plt.plot(xf_grid, yf_grid, '.', c='#81c9fc',  label = 'Collocation Point', clip_on=False)
    plt.gca().set(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y', title='Training Data')
    plt.figlegend(handles=[bc, cp], loc='center right', bbox_to_anchor=(0.5, 0., 0.5, 0.5), fontsize=ticksize, framealpha=0.9)
    fig.subplots_adjust(left=0.11, right=0.67, bottom=0.13, top=0.92)
    plt.savefig(f'{dirpath}/trainingdata.pdf')
    plt.close(fig)
    dataDict = {
        'boundary points':{
            'N_b': N_b,
            'xb': xb.tolist(),
            'yb': yb.tolist(),
            'x0': x0.tolist(),
            'xe': xe.tolist(),
            'y0': y0.tolist(),
            'ye': ye.tolist(),
            'ul': ul.tolist(),
            'ur': ur.tolist(),
            'ub': ub.tolist(),
            'ut': ut.tolist(),
            'ul_x': ul_x.tolist(),
            'ur_x': ur_x.tolist(),
            'ub_y': ub_y.tolist(),
            'ut_y': ut.tolist(),
        },
        'collocation points':{
            'N_f': N_f,
            'xf_grid': xf_grid.tolist(),
            'yf_grid': yf_grid.tolist(),
        },
        'testing data':{
            "N_test": N_test,
            "xtest_mesh": xtest_mesh.tolist(),
            "ytest_mesh": ytest_mesh.tolist(),
            "u_test": u_test.tolist()
        }
    }
    with open(f'{dirpath}/data.json', 'w') as f:
        json.dump(dataDict, f)

    # Note: loss around 10^-3/-4 should be about good
    loss_values, u_preds, f_preds = ([] for i in range(3))
    N_iter = 10000
    loss_value_step = 10
    pred_step = 100
    contour_step = 1000 # if not pred_step's multiple, graph every least common multiple (pred_step, contour_step)
    for i in range(N_iter):
        lr = 10**-3 * 2**(-i/10000) if i <= 60000 else 10**-3 * 2**(-60000/10000) #  0.00002210/0.00001563 # learning rate decay
        loss_value = model.train(lr) # from last iteration
        if (i+1) % loss_value_step == 0:  # start with i=9 and end with i=8999 (last iter)
            loss_values.append(float(loss_value))
            print('Iter: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.8f' % (i+1, loss_value, time.time() - start_time, lr))
        if (i+1) % pred_step == 0:  # start with i=999 and end with i=8999 (last iter)
            u_pred, f_pred = model.predict(xtest_grid, ytest_grid)
            u_preds.append(u_pred)  # (N_test * N_test, 1)
            f_preds.append(f_pred)  # (N_test * N_test, 1)
            if (i+1) % contour_step == 0: # start with i=2999 and end with i=8999 (last iter)
            ## Plot 2. u (Exact, Preidction) vs (x,y) and |u_pred-u_test| vs (x,y): contour
                contourPlot(xtest_mesh, ytest_mesh, u_test, u_pred, N_test, i)
    training_time = time.time() - start_time
    u_preds = np.array(u_preds)
    f_preds = np.array(f_preds)
    u_pred, f_pred = model.predict(xtest_grid, ytest_grid)
    # NOTE: what is important is the function u_pred resembles, not so much the parameters (weights & biases)
    # NOTE: if no analytical solution, find numerical method/other method to verify -> directly use network

   ###########################
    ## PART 4: calculating errors
    error_u = np.linalg.norm(u_pred - u_test, 2) / np.linalg.norm(u_test, 2) # scalar
    print('Error u: %e' % (error_u))

    ###########################
    ## PART 5: Plotting

    # Plot 3. loss vs. iteration
    fig = plt.figure(figsize=(6.8, 6))
    plt.ticklabel_format(axis='x', style="sci", scilimits=(3,3))
    x_coords = loss_value_step * (np.array(range(len(loss_values))) + 1)
    plt.semilogy(x_coords, loss_values) # linear X axis, logarithmic y axis(log scaling on the y axis)
    plt.gca().set(xlabel='Iteration', ylabel='Loss', title='Loss during Training')
    init_tuple = (loss_value_step, loss_values[0])
    plt.annotate('(%d, %.3e)' % init_tuple, xy=init_tuple, fontsize=annotatesize, ha='left')
    last_tuple = (N_iter, loss_values[-1])
    plt.annotate('(%d, %.3e)' % last_tuple, xy=last_tuple, fontsize=annotatesize, ha='right', va='top')
    plt.plot([init_tuple[0], last_tuple[0]], [init_tuple[1], last_tuple[1]], '.', c='#3B75AF')
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.07, top=0.95) 
    # NOTE: Oscillation: actually very small nummerical difference because of small y scale
        # 1. overshoot (fixed -> decaying learning rate)
        # 2. Adam: gradient descent + momentum (sometime parameter change makes the loss go up)
    plt.savefig(f'{dirpath}/forward_2d_loss.pdf')
    plt.close(fig)
    with open(f'{dirpath}/forward_2d_loss.json', 'w') as f:
        json.dump({"x_coords": x_coords.tolist(), "loss_values": loss_values}, f)

    # Plot 4. MSE between u_pred and u_test vs. iteration
    fig = plt.figure(figsize=(6, 6))
    plt.ticklabel_format(axis='x', style="sci", scilimits=(3,3))
    x_coords = pred_step * (np.array(range(len(u_preds))) + 1)
    u_mses = [((u_pred - u_test)**2).mean(axis=0) for u_pred in u_preds] #[[mse1] [mse2] [mse3]]
    u_mses = np.array(u_mses)
    plt.semilogy(x_coords, u_mses, '.-')
    plt.gca().set(xlabel='Iteration', ylabel='MSE of u', title='MSE of u during Training')
    annots = list(zip(x_coords, u_mses.flatten())) # [(1000, 4.748), (2000, 9.394)]
    plt.annotate('(%d, %.3e)' % annots[0], xy=annots[0], fontsize=annotatesize, ha='left')
    plt.annotate('(%d, %.3e)' % annots[-1], xy=annots[-1], fontsize=annotatesize, ha='right', va='top')
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.07, top=0.95)
    plt.savefig(f'{dirpath}/forward_2d_mse.pdf')
    plt.close(fig)
    with open(f'{dirpath}/forward_2d_mse.json', 'w') as f:
        json.dump({"x_coords": x_coords.tolist(), "u_mses": u_mses.tolist()}, f)

    # Plot 5. plot u vs (x, y): surface
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xtest_mesh, ytest_mesh, np.reshape(u_test, (N_test, N_test)), label='Exact', cmap='winter')  # Data values as 2D arrays: (N_test * N_test, 1)
    plt.gca().set(xlabel='$x$', ylabel='$y$', zlabel='$u$', title='Exact')
    ax.tick_params(labelsize=ticksize)

    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot_surface(xtest_mesh, ytest_mesh, np.reshape(u_pred, (N_test, N_test)), label='Prediction', cmap='winter')  # Data values as 2D arrays: (N_test * N_test, 1)
    plt.gca().set(xlabel='$x$', ylabel='$y$', zlabel='$u$', title='Prediction')
    ax.tick_params(labelsize=ticksize)

    plt.savefig(f'{dirpath}/forward_2d_surface.pdf')
    plt.close(fig)

    ###########################
    ## PART 6: Saving information
    infoDict = {
        'problem':{
            'pde form': 'u_xx + u_yy = 0',
            'boundary (x)': [float(lowerbound[0]),float(upperbound[0])], 
            'boundary (y)': [float(lowerbound[1]),float(upperbound[1])], 
            'boundary conditions mix': mix,
            'boundary condition (u)': 'u(0, y) = -5 * y^2, u(1, y) = 1 - 5 * y^2 + 3y, u(x, 0) = x^2, u(x, 1) = x^2 - 5 + 3x',
            'boundary condition (u_x, u_y)': 'u_x(0, y) = 3y, u_x(1, y) = 2 + 3y, u_y(x, 0) = 3x, u_y(x, 1) = -2 + 3x',
            'analytical solution': 'u(x, y) = x^2 - y^2 + 3xy'
        },
        'model':{
            'layers': str(layers),
            'training iteration': N_iter,
            'loss_value_step':loss_value_step,
            'pred_step': pred_step,
            'contour_step': contour_step,
            'training_time': training_time,
            'error_u': error_u,
        },
        'training data':{
            'N_b': N_b,
            'xb': str(xb),
            'yb': str(yb),
            'N_f': N_f,
            'xf': str(xf),
            'yf': str(yf)
        }
    }
    with open(f'{dirpath}/info.json', 'w') as f:
        json.dump(infoDict, f, indent=4)