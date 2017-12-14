''' Forecast time series '''

import random
import sys 
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def build_lstm_graph(n_features, n_targets, burn_in,
                     num_units, input_keep_prob=1.0, output_keep_prob=1.0,
                     variable_scope='ts', dtype=tf.float32):
    ''' Build the symbolic graph for modeling the time series '''
    # x, y are indexed by batch, time_step and feature
    with tf.variable_scope(variable_scope):
        x = tf.placeholder(dtype, [None, None, n_features], name='x')
        y = tf.placeholder(dtype, [None, None, n_targets], name='y')

        cell = tf.contrib.rnn.LSTMCell(num_units, use_peepholes=True)
        dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob,
                                                     output_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(dropout_cell, x, dtype=dtype)

        w_fcst = tf.get_variable('w_fcst', [n_features + num_units, n_targets])
        b_fcst = tf.get_variable('b_fcst', [n_targets])

        # Use the last n_targets elements in each output vector at
        # each time step to match against y

        # Features for linear forecast 
        features_ = tf.concat([tf.reshape(x, [-1, n_features]),
                               tf.reshape(outputs, [-1, num_units])], axis=1) 
        pred = tf.nn.xw_plus_b(features_, w_fcst, b_fcst)
        pred = tf.reshape(pred, tf.shape(y))

        # Prediction error and loss 
        cost = tf.losses.mean_squared_error(pred[:, burn_in:, :],
                                            y[:, burn_in:, :])

    return {'x': x, 'y': y, 'pred': pred, 'cost': cost,
            'lstm_state': state, 'lstm_outputs': outputs, 
            'lstm_weights': cell.weights,
            'w_fcst': w_fcst, 'b_fcst': b_fcst}, cell

def train_lstm(sess, ts, features_func, targets_func, burn_in,
               batch_size, lr0=1e-5, lr_decay=(50, .99),
               n_iter=500, valid_every=5, print_every=5,
               variable_scope='ts', **kwargs):
    ''' Train LSTM for given features and targets functions '''
    # ts <num samples>-by-<length of every sample>
    # Split ts into train, dev set; we'll only use ts_test once at the end
    ts_train, ts_dev = train_test_split(ts, test_size=.1)

    # Make features, targets for LSTM training
    features = np.apply_along_axis(features_func, axis=1, arr=ts_train)
    targets = np.apply_along_axis(targets_func, axis=1, arr=ts_train)
    dev_features = np.apply_along_axis(features_func, axis=1, arr=ts_dev)
    dev_targets = np.apply_along_axis(targets_func, axis=1, arr=ts_dev)

    if features.ndim == 2:
        features = features[:, :, None]
        dev_features = dev_features[:, :, None]
    if targets.ndim == 2:
        targets = targets[:, :, None]
        dev_targets = dev_targets[:, :, None]
    n_features = features.shape[2]
    n_targets = targets.shape[2]

    # The burn-in period would be excluded from cost calculation
    lstm, cell = build_lstm_graph(n_features, n_targets, burn_in,
                                  variable_scope=variable_scope, **kwargs)

    # Initialise optimiser
    with tf.variable_scope(variable_scope):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr0, global_step,
                                                   lr_decay[0], lr_decay[1])
        optimizer = (tf.train.MomentumOptimizer(learning_rate, momentum=.5)
                     .minimize(lstm['cost'], global_step=global_step))

    # Begin training
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=variable_scope)
    sess.run(tf.variables_initializer(var_list))

    # Run minibatch SGD
    # Break when Ctrl-C is pressed
    try:
        for i in range(n_iter):
            msg = f'Iter {i}'
            # Run SGD
            batch = random.sample(range(ts_train.shape[0]), batch_size) 
            _, cost = sess.run([optimizer, lstm['cost']],
                               feed_dict={lstm['x']: features[batch],
                                          lstm['y']: targets[batch]})
            msg += f' Train loss {np.sqrt(cost):.4f}'

            if i % valid_every == 0:
                dict_ = {lstm['x']: dev_features, lstm['y']: dev_targets}
                dev_cost = sess.run(lstm['cost'], feed_dict=dict_)
                msg += f' Dev loss {np.sqrt(dev_cost):.4f}'

            if i % print_every == 0:
                print(msg, file=sys.stderr)
    except KeyboardInterrupt:
        pass

    return lstm, cell 

def eval_ar(sess, lstm, ts_test, features_func, targets_func, burn_in):
    ''' Evaluate the AR model '''
    # ts_test <num samples>-by-<num variables>
    #            -by-<length of every sample/series>
    TS_WITH_NOISE = 0
    TS_WITH_NO_NOISE = 1

    x = ts_test[:, TS_WITH_NOISE, :].squeeze()
    x_no_noise = ts_test[:, TS_WITH_NO_NOISE, :].squeeze()

    features = np.apply_along_axis(features_func, axis=1, arr=x)
    targets = np.apply_along_axis(targets_func, axis=1, arr=x)
    targets_no_noise = np.apply_along_axis(targets_func, axis=1,
                                           arr=x_no_noise)
    if features.ndim == 2:
        features = features[:, :, None]
    if targets.ndim == 2:
        targets = targets[:, :, None]
        targets_no_noise = targets_no_noise[:, :, None]

    dict_ = {lstm['x']: features, lstm['y']: targets}
    cost, pred = sess.run([lstm['cost'], lstm['pred']], feed_dict=dict_)
    cost_no_noise = mean_squared_error(targets_no_noise[:, burn_in:, 0],
                                       pred[:, burn_in:, 0])

    return np.sqrt(cost), np.sqrt(cost_no_noise), pred
    
if __name__ == '__main__':
    ''' Command line interface
    
    Usage:

    seq 1 50 | xargs -I {} -P 3 python3 tspred.py simulation.npz simulation_test.npz >>arima22_run50.csv
    '''
    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    args = parser.parse_args()
    
    # Read data
    data = np.load(args.train_file)['data']
    data_test = np.load(args.test_file)['data']

    # Train
    simple_features = lambda x: x[:-1]
    moments_features = lambda x: np.column_stack([x[:-1], x[:-1] ** 2])

    sess = tf.Session()
    burn_in = 50
    features_func = simple_features
    res = train_lstm(sess, data[:, 0, :].squeeze() * 10,
                     features_func, lambda x: x[1:],
                     burn_in=burn_in, batch_size=50,
                     lr0=3e-3, lr_decay=(50, .99), n_iter=300,
                     num_units=10)

    # Test
    cost, cost_no_noise, pred = eval_ar(sess, res[0],
                                        data_test * 10,
                                        features_func,
                                        lambda x: x[1:], burn_in)
    pred_error = data_test[:, 1, 1:].squeeze() - pred.squeeze() / 10
    print(' '.join([str(w) for w in pred_error.flat]))
