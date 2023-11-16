#'
#' https://towardsdatascience.com/how-to-implement-deep-learning-in-r-using-keras-and-tensorflow-82d135ae4889
#'
suppressPackageStartupMessages(suppressWarnings({
  library(argparse)
  parser <- ArgumentParser()
  parser$add_argument("--job_home", default = "~/dev/xinh3ng/DSResearch/ML/gist")
  args <- parser$parse_args()

  setwd(args$job_home)  # set working dir
  Sys.setenv(TZ = "America/Los_Angeles")  # remove "unknown timezone warning"
}))
suppressPackageStartupMessages(suppressWarnings({
  library(dplyr)
  library(futile.logger)  # Set up the logger
  flog.layout(layout.format("~t ~l ~n.~f: ~m"))
  flog.threshold(futile.logger::INFO)  # DEBUG, INFO

  library(keras)
}))

cifar <- dataset_cifar10()

# convert a vector class to binary class matrix
# converting the target variable to once hot encoded vectors using #keras inbuilt function to_categorical()
train_y <- to_categorical(cifar$train$y, num_classes = 10)
train_x <- cifar$train$x / 255

test_y <- to_categorical(cifar$test$y, num_classes=10)
test_x <- cifar$test$x / 255
flog.info(paste("No of training samples:", dim(train_x)[[1]],", No of test samples:", dim(test_x)[[1]]))

########################
# Define CNN architecture
########################
model <- keras_model_sequential()
model %>%
  # define a 2-D convolution layer
  keras::layer_conv_2d(filter = 32, kernel_size = c(3, 3), padding = "same", input_shape = c(32,32,3)) %>%
  layer_activation("relu") %>%

  # another 2-D convolution layer
  layer_conv_2d(filter = 32 ,kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  # a Pooling layer reduces the dimensions of the features map and reduces the computational complexity of the model
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  # dropout layer to avoid overfitting
  layer_dropout(0.25) %>%

  layer_conv_2d(filter = 32 , kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%

  #Flatten the input
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  # Output layer-10 classes-10 units
  layer_dense(10) %>%

  # Apply softmax nonlinear activation function to the output layer to calculate cross-entropy
  layer_activation("softmax") # for computing Probabilities of classes-"logit(log probabilities)

# Model Optimizer
# lr: learning rate , decay: learning rate decay over each update
opt <- optimizer_adam(lr= 0.0001, decay = 1e-6 )
model %>%
  compile(loss = "categorical_crossentropy",
          optimizer = opt, metrics = "accuracy")
print(summary(model))  # Summary of the Model and its Architecture

###############################
# Model Training
##############################
data_augmentation <- FALSE
if(!data_augmentation) {
  model %>% fit(train_x, train_y, batch_size = 32,
                epochs = 80, validation_data = list(test_x, test_y),
                shuffle = TRUE)
} else {
  # Generating images
  gen_images <- keras::image_data_generator(
    featurewise_center = TRUE, featurewise_std_normalization = TRUE,
    rotation_range = 20, width_shift_range = 0.30, height_shift_range = 0.30,
    horizontal_flip = TRUE  )

  # Fit image data generator internal statistics to some sample data
  gen_images %>% fit_image_data_generator(train_x)
  # Generates batches of augmented/normalized data from image data and
  # labels to visually see the generated images by the Model
  model %>% fit_generator(
    flow_images_from_data(train_x, train_y, gen_images,
                          batch_size = 32, save_to_dir = "/tmp/cnn_cifar_images/"),
    steps_per_epoch = as.integer(50000/32), epochs = 80,
    validation_data = list(test_x, test_y))
}


# Stop everything
flog.info("ALL DONE\n")
