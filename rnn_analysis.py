''' Analysis of trained RNN '''

import sys
import random
import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt

def lstm_states(sess, cell, x, dtype=tf.float32):
    ''' Get LSTM states at all time steps '''
    batch_size, num_steps, input_size = x.shape
    c_size, h_size = cell.state_size

    curr_x = tf.placeholder(dtype, [None, input_size], name='curr_x')
    curr_c = tf.placeholder(dtype, [None, c_size], name='curr_c')
    curr_h = tf.placeholder(dtype, [None, h_size], name='curr_h')

    _, new_state = cell(curr_x, [curr_c, curr_h])

    c_state = np.random.uniform(-1, 1, (batch_size, c_size))
    h_state = np.random.uniform(-1, 1, (batch_size, h_size))
    c_states = []
    h_states = []
    for j in range(num_steps):
        feed_dict = {curr_x: x[:, j, :], curr_c: c_state, curr_h: h_state}
        (new_c_state, new_h_state) = sess.run(new_state, feed_dict=feed_dict)
        c_states.append(new_c_state)
        h_states.append(new_h_state)
        c_state = new_c_state
        h_state = new_h_state
    c_states = np.stack(c_states, axis=1)
    h_states = np.stack(h_states, axis=1)

    return c_states, h_states

def component_analysis(states, n_components):
    ''' PCA and ICA of states 
    
    Return the ratios of variance of each component.
    '''
    batch_size, num_steps, state_size = states.shape
    states_ = states.reshape((-1, state_size))

    pca = PCA(n_components=n_components)
    ica = FastICA(n_components=n_components)

    pca.fit(states_)
    ica.fit(states_)

    # Orthogonal the unmixing vectors
    w = ica.components_
    w /= np.linalg.norm(w, axis=1)[:, None]
    ica_comp_var = np.var(states_.dot(w.T), axis=0)
    ica_comp_var = np.sort(ica_comp_var)[::-1]

    return pca.explained_variance_ratio_, ica_comp_var / sum(ica_comp_var)

def plot_component_analysis(pca_var, ica_var):
    ''' Plot the ratios of variances of components '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(pca_var, label='PCA', alpha=.4, linestyle='--') 
    ax[0].plot(ica_var, label='ICA')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xlabel('Component')
    ax[0].set_ylabel('Ratio')
    ax[0].set_title('Ratio of Variance of Component')

    ax[1].plot(np.cumsum(pca_var), label='PCA', alpha=.4, linestyle='--') 
    ax[1].plot(np.cumsum(ica_var), label='ICA')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xlabel('Component')
    ax[1].set_ylabel('Cumulative Ratio')
    ax[1].set_title('Cumulative Ratio of Variance of Component')
    
def build_mlp_graph(n_features, n_hidden, y_type,
                    variable_scope, dtype=tf.float32):
    ''' Build graph for MLP regressor '''
    with tf.variable_scope(variable_scope):
        x = tf.placeholder(dtype, [None, n_features], name='mlp_x')
        y = tf.placeholder(dtype, [None], name='mlp_y')
        
        y_pred_dim = 1 if y_type.lower().startswith('n') else 2

        laplacian = tf.distributions.Laplace(0.0, 1.0)
        w1_init = laplacian.sample([n_features, n_hidden])
        w2_init = laplacian.sample([n_hidden, y_pred_dim])

        w1 = tf.get_variable('w1', dtype=dtype, initializer=w1_init)
        b1 = tf.get_variable('b1', [n_hidden], dtype)
        h1 = tf.tanh(tf.nn.xw_plus_b(x, w1, b1))

        w2 = tf.get_variable('w2', dtype=dtype, initializer=w2_init)
        b2 = tf.get_variable('b2', [y_pred_dim], dtype)

        y_pred = tf.squeeze(tf.nn.xw_plus_b(h1, w2, b2))

        if y_type.lower().startswith('n'):    # numerical
            # loss = tf.losses.mean_squared_error(y, y_pred)
            loss = .5 * (y - y_pred) - tf.minimum(y - y_pred, 0)
            loss = tf.reduce_mean(loss)
        else:                                 # categorical
            loss = tf.losses.softmax_cross_entropy(
                    tf.stack([y, 1 - y], axis=1), y_pred)

    return {'mlp_x': x, 'mlp_y': y, 'y_pred': y_pred, 'loss': loss,
            'w1': w1, 'b1': b1}

def mlp_reg(sess, x, y, y_type, n_hidden, variable_scope, dtype=tf.float32,
            batch_size=50, lr0=3e-3, lr_decay=(50, .99), n_iter=500,
            print_every=5):
    ''' Perform MLP regression '''
    p = x.shape[-1]
    symbols = build_mlp_graph(p, n_hidden, y_type, variable_scope, dtype)

    with tf.variable_scope(variable_scope):
        global_step = tf.Variable(0, name='step', trainable=False)
        learning_rate = tf.train.exponential_decay(lr0, global_step,
                                                   lr_decay[0], lr_decay[1])
        optimizer = (tf.train.MomentumOptimizer(learning_rate, momentum=.5)
                     .minimize(symbols['loss'], global_step=global_step))

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=variable_scope)
    sess.run(tf.variables_initializer(var_list))

    try:
       for i in range(n_iter):
            msg = f'Iter {i}'
            batch = random.sample(range(x.shape[0]), batch_size)

            # Run SGD
            feed_dict = {symbols['mlp_x']: x[batch].reshape((-1, p)),
                         symbols['mlp_y']: y[batch].ravel()}
            _, y_pred, loss = sess.run([optimizer, symbols['y_pred'],
                                        symbols['loss']],
                                       feed_dict=feed_dict)
            msg += f' Train loss {loss:.4f}'

            if y_type.lower().startswith('n'):
                r2 = r2_score(y[batch].ravel(), y_pred)
                msg += f' R2 {r2}'
                correl = stats.spearmanr(y[batch].ravel(), y_pred)
                msg += f' Spearman R {correl}'

            if i % print_every == 0:
                print(msg, file=sys.stderr) 

    except KeyboardInterrupt:
        pass

    return symbols
