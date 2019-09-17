#' Trains a feedforward DL model on the MNIST dataset.


# Prerequisites -----------------------------------------------------------

library(keras)


# Data Prep ---------------------------------------------------------------

# Import MNIST training data
mnist <- dslabs::read_mnist()
mnist_x <- mnist$train$images
mnist_y <- mnist$train$labels

# rename features
colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))

# standardize features
mnist_x <- mnist_x / 255

# one-hot encode response
mnist_y <- to_categorical(mnist_y, 10)

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  # nodes
  flag_numeric("nodes1", 256),
  flag_numeric("nodes2", 128),
  flag_numeric("nodes3", 64),
  # dropout
  flag_numeric("dropout1", 0.4),
  flag_numeric("dropout2", 0.3),
  flag_numeric("dropout3", 0.2),
  # learning paramaters
  flag_string("optimizer", "rmsprop"),
  flag_numeric("lr_annealing", 0.1)
)

# Define Model --------------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes1, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$nodes2, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$nodes3, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    metrics = c('accuracy'),
    optimizer = FLAGS$optimizer
  ) %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = FLAGS$lr_annealing)
    ),
    verbose = FALSE
  )
