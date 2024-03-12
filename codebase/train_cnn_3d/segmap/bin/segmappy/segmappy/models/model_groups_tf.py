import tensorflow as tf

# define the cnn model
def init_model(input_shape, n_classes):
    tf.compat.v1.disable_eager_execution()
    #tf.compat.v1.reset_default_graph()
    with tf.compat.v1.name_scope("InputScope") as scope:
        cnn_input = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None,) + input_shape + (1,), name="input"
        )

    # base convolutional layers
    y_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, n_classes), name="y_true")

    scales = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 3), name="scales")

    training = tf.compat.v1.placeholder_with_default(
        tf.constant(False, dtype=tf.bool), shape=(), name="training"
    )

    conv1 = tf.compat.v1.layers.conv3d(
        inputs=cnn_input,
        filters=32,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        name="conv1",
    )

    pool1 = tf.compat.v1.layers.max_pooling3d(
        inputs=conv1, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool1"
    )

    conv2 = tf.compat.v1.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        name="conv3",
    )

    pool2 = tf.compat.v1.layers.max_pooling3d(
        inputs=conv2, pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2"
    )

    conv3 = tf.compat.v1.layers.conv3d(
        inputs=pool2,
        filters=64,
        kernel_size=(3, 3, 3),
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        name="conv5",
    )

    # flatten = tf.contrib.layers.flatten(inputs=conv3)
    # flatten = tf.concat([flatten, scales], axis=1, name="flatten")
    # Flatten the convolutional layer
    flatten = tf.compat.v1.layers.flatten(inputs=conv3)
    # Concatenate the flattened tensor and scales tensor along axis 1
    flatten = tf.concat([flatten, scales], axis=1, name="flatten")

    # classification network
    dense1 = tf.compat.v1.layers.dense(
        inputs=flatten,
        units=512,
        activation=tf.nn.relu,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        use_bias=True,
        name="dense1",
    )

    bn_dense1 = tf.compat.v1.layers.batch_normalization(
        dense1, training=training, name="bn_dense1"
    )

    dropout_dense1 = tf.compat.v1.layers.dropout(
        bn_dense1, rate=0.5, training=training, name="dropout_dense1"
    )

    descriptor = tf.compat.v1.layers.dense(
        inputs=dropout_dense1,
        units=64,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation=tf.nn.relu,
        use_bias=True,
        name="descriptor",
    )

    bn_descriptor = tf.compat.v1.layers.batch_normalization(
        descriptor, training=training, name="bn_descriptor"
    )

    with tf.compat.v1.name_scope("OutputScope") as scope:
        tf.add(bn_descriptor, 0, name="descriptor_bn_read")
        tf.add(descriptor, 0, name="descriptor_read")

    dropout_descriptor = tf.compat.v1.layers.dropout(
        bn_descriptor, rate=0.35, training=training, name="dropout_descriptor"
    )

    y_pred = tf.compat.v1.layers.dense(
        inputs=dropout_descriptor,
        units=n_classes,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation=None,
        use_bias=True,
        name="classes",
    )   

    print("Shape of y_pred:", y_pred.get_shape())
    print("Shape of y_true:", y_true.get_shape())
    print("y_pred:", y_pred)
    print("y_true:", y_true)
    assert y_pred.get_shape().as_list() == y_true.get_shape().as_list(), "Logits and labels must have the same shape"
    tf.debugging.check_numerics(y_pred, "y_pred contains NaNs or Infs")
    tf.debugging.check_numerics(y_true, "y_true contains NaNs or Infs")


    loss_c = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true),
        name="loss_c",
    )

    # reconstruction network
    dec_dense1 = tf.compat.v1.layers.dense(
        inputs=descriptor,
        units=8192,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation=tf.nn.relu,
        use_bias=True,
        name="dec_dense1",
    )

    reshape = tf.reshape(dec_dense1, (tf.shape(cnn_input)[0], 8, 8, 4, 32))

    dec_conv1 = tf.compat.v1.layers.conv3d_transpose(
        inputs=reshape,
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation=tf.nn.relu,
        name="dec_conv1",
    )

    dec_conv2 = tf.compat.v1.layers.conv3d_transpose(
        inputs=dec_conv1,
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation=tf.nn.relu,
        name="dec_conv2",
    )

    dec_reshape = tf.compat.v1.layers.conv3d_transpose(
        inputs=dec_conv2,
        filters=1,
        kernel_size=(3, 3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        activation=tf.nn.sigmoid,
        name="dec_reshape",
    )

    reconstruction = dec_reshape
    with tf.compat.v1.name_scope("ReconstructionScopeAE") as scope:
        tf.add(reconstruction, 0, name="ae_reconstruction_read")

    FN_TO_FP_WEIGHT = 0.9
    loss_r = -tf.reduce_mean(
        FN_TO_FP_WEIGHT * cnn_input * tf.math.log(reconstruction + 1e-10)
        + (1 - FN_TO_FP_WEIGHT) * (1 - cnn_input) * tf.math.log(1 - reconstruction + 1e-10)
    )
    tf.identity(loss_r, "loss_r")

    # training
    LOSS_R_WEIGHT = 200
    LOSS_C_WEIGHT = 1
    loss = tf.add(LOSS_C_WEIGHT * loss_c, LOSS_R_WEIGHT * loss_r, name="loss")

    global_step = tf.Variable(0, trainable=False, name="global_step")

    #print('print in model: global step::', global_step)
    update_step = tf.compat.v1.assign(
        global_step, tf.add(global_step, tf.constant(1)), name="update_step"
    )
    #print('print in model: up step::', global_step)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.002)
    # original learning rate: 0.0001

    # add batch normalization updates to the training operation
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, name="train_op")

    # statistics
    y_prob = tf.nn.softmax(y_pred, name="y_prob")

    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    roc_auc = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name="roc_auc")

    with tf.compat.v1.name_scope("summary"):
        tf.compat.v1.summary.scalar("loss", loss, collections=["summary_batch"])
        tf.compat.v1.summary.scalar("loss_c", loss_c, collections=["summary_batch"])
        tf.compat.v1.summary.scalar("loss_r", loss_r, collections=["summary_batch"])
        tf.compat.v1.summary.scalar("accuracy", accuracy, collections=["summary_batch"])
        tf.compat.v1.summary.scalar("global_step", global_step, collections=["summary_batch"])
        tf.compat.v1.summary.scalar("roc_auc", roc_auc, collections=["summary_epoch"])
