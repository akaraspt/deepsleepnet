import tensorflow as tf

from deepsleep.nn import *

class DeepFeatureNet(object):

    def __init__(
        self, 
        batch_size, 
        input_dims, 
        n_classes, 
        is_train, 
        reuse_params, 
        use_dropout, 
        name="deepfeaturenet"
    ):
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.use_dropout = use_dropout
        self.name = name

        self.activations = []
        self.layer_idx = 1
        self.monitor_vars = []

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32, 
            shape=[self.batch_size, ],
            name=name + "_targets"
        )

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0):
        input_shape = input_var.get_shape()
        n_batches = input_shape[0].value
        input_dims = input_shape[1].value
        n_in_filters = input_shape[3].value
        name = "l{}_conv".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = conv_1d(name="conv1d", input_var=input_var, filter_shape=[filter_size, 1, n_in_filters, n_filters], stride=stride, bias=None, wd=wd)
            
            # # MONITORING
            # self.monitor_vars.append(("{}_before_bn".format(name), output))

            output = batch_norm_new(name="bn", input_var=output, is_train=self.is_train)

            # # MONITORING
            # self.monitor_vars.append(("{}_after_bn".format(name), output))

            # output = leaky_relu(name="leaky_relu", input_var=output)
            output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def build_model(self, input_var):
        # List to store the output of each CNNs
        output_conns = []

        ######### CNNs with small filter size at the first layer #########

        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=128, n_filters=64, stride=16, wd=1e-3)
        network = self._conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=8, stride=8)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### CNNs with large filter size at the first layer #########

        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=1024, n_filters=64, stride=128)
        network = self._conv1d_layer(input_var=input_var, filter_size=400, n_filters=64, stride=50)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=2, stride=2)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### Aggregate and link two CNNs #########

        # Concat
        name = "l{}_concat".format(self.layer_idx)
        network = tf.concat(axis=1, values=output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:
            
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # Cross-entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.target_var,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            loss = tf.reduce_mean(loss, name="cross_entropy")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # print " "
            # print "Params to compute regularization loss:"
            # for p in tf.compat.v1.get_collection("losses", scope=scope.name + "\/"):
            #     print p.name
            # print " "

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)


class DeepSleepNet(DeepFeatureNet):

    def __init__(
        self, 
        batch_size, 
        input_dims, 
        n_classes, 
        seq_length,
        n_rnn_layers,
        return_last,
        is_train, 
        reuse_params,
        use_dropout_feature, 
        use_dropout_sequence,
        name="deepsleepnet"
    ):
        super(self.__class__, self).__init__(
            batch_size=batch_size, 
            input_dims=input_dims, 
            n_classes=n_classes, 
            is_train=is_train, 
            reuse_params=reuse_params, 
            use_dropout=use_dropout_feature, 
            name=name
        )

        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last

        self.use_dropout_sequence = use_dropout_sequence

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[self.batch_size*self.seq_length, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32, 
            shape=[self.batch_size*self.seq_length, ],
            name=name + "_targets"
        )

    def build_model(self, input_var):
        # Create a network with superclass method
        network = super(self.__class__, self).build_model(
            input_var=self.input_var
        )

        # Residual (or shortcut) connection
        output_conns = []

        # Fully-connected to select some part of the output to add with the output from bi-directional LSTM
        name = "l{}_fc".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output_tmp = fc(name="fc", input_var=network, n_hiddens=1024, bias=None, wd=0)
            output_tmp = batch_norm_new(name="bn", input_var=output_tmp, is_train=self.is_train)
            # output_tmp = leaky_relu(name="leaky_relu", input_var=output_tmp)
            output_tmp = tf.nn.relu(output_tmp, name="relu")
        self.activations.append((name, output_tmp))
        self.layer_idx += 1
        output_conns.append(output_tmp)

        ######################################################################

        # Reshape the input from (batch_size * seq_length, input_dim) to
        # (batch_size, seq_length, input_dim)
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1].value
        seq_input = tf.reshape(network,
                               shape=[-1, self.seq_length, input_dim],
                               name=name)
        assert self.batch_size == seq_input.get_shape()[0].value
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        # Bidirectional LSTM network
        name = "l{}_bi_lstm".format(self.layer_idx)
        hidden_size = 512   # will output 1024 (512 forward, 512 backward)
        with tf.compat.v1.variable_scope(name) as scope:

            def lstm_cell():

                cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_size,                               
                                                use_peepholes=True,
                                                state_is_tuple=True,
                                                reuse=tf.compat.v1.get_variable_scope().reuse) 
                if self.use_dropout_sequence:
                    keep_prob = 0.5 if self.is_train else 1.0
                    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        cell,
                        output_keep_prob=keep_prob
                    )

                return cell

            fw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.n_rnn_layers)], state_is_tuple = True)
            bw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.n_rnn_layers)], state_is_tuple = True)

            # Initial state of RNN
            self.fw_initial_state = fw_cell.zero_state(self.batch_size, tf.float32)
            self.bw_initial_state = bw_cell.zero_state(self.batch_size, tf.float32)

            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unstack(seq_input, axis=1)
            #outputs, fw_state, bw_state = tf.nn.bidirectional_rnn(
            outputs, fw_state, bw_state = tf.compat.v1.nn.static_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if self.return_last:
                network = outputs[-1]
            else:
                network = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size*2],
                                    name=name)
            self.activations.append((name, network))
            self.layer_idx +=1

            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

        # Append output
        output_conns.append(network)

        ######################################################################

        # Add
        name = "l{}_add".format(self.layer_idx)
        network = tf.add_n(output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout_sequence:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:
            
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # Weighted cross-entropy loss for a sequence of logits (per example)
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [self.target_var],
                [tf.ones([self.batch_size * self.seq_length])],
                name="sequence_loss_by_example"
            )
            loss = tf.reduce_sum(loss) / self.batch_size

            # Regularization loss
            regular_loss = tf.add_n(
                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # print " "
            # print "Params to compute regularization loss:"
            # for p in tf.compat.v1.get_collection("losses", scope=scope.name + "\/"):
            #     print p.name
            # print " "

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
