import rpyc
import time
import numpy as np
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

CNN_LAYERS_COUNT=3
STEPS=10
MASTER_PORT=18861

def cnn_model_fn(
  features, labels, mode,
  filters_counts=[32, 32, 32], 
  filter_sizes=[[5, 5], [5, 5], [5, 5]],
):
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Additively add convolutional layers
  cur_layer = input_layer
  for i in range(0, CNN_LAYERS_COUNT):
    cur_layer = tf.layers.conv2d(
      inputs=cur_layer,
      filters=filters_counts[i],
      kernel_size=filter_sizes[i],
      padding="same",
      activation=tf.nn.relu
    )

  pool2_flat = tf.reshape(
    cur_layer, 
    [-1, 28 * 28 * filters_counts[CNN_LAYERS_COUNT - 1]]
  )

  dense = tf.layers.dense(
    inputs=pool2_flat,
    units=1024,
    activation=tf.nn.relu
  )

  dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.4,
    training=(mode == tf.estimator.ModeKeys.TRAIN)
  )

  logits = tf.layers.dense(
    inputs=dropout,
    units=10
  )

  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. Used for PREDICT and by the `logging_hook`
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
  # Calculate loss (for TRAIN and EVAL mode)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure training op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
      labels=labels,
      predictions=predictions["classes"]
    )
  }
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class Worker():
    def start(self):
      # Infinitely, receive training options from master, train a ConvNet, then inform master of results
      results_prev = None
      options_prev = None

      print("Connecting to master on port {}...".format(MASTER_PORT))
      connection = rpyc.connect("localhost", MASTER_PORT, config={"allow_all_attrs": True})
      master = connection.root

      while True:
        if options_prev and results_prev:
          print("Sending results {} for options {} to master...".format(results_prev, options_prev))
          master.put_training_results(options_prev, results_prev)

        options_prev = master.get_training_options()
        print("Received options from master: {}".format(options_prev))
        
        results_prev = self.train(options_prev)
        print("Finished training - results {} for options {}".format(results_prev, options_prev))

      connection.close()
          
    def train(self, options):
      # Load training and eval data
      mnist = tf.contrib.learn.datasets.load_dataset("mnist")
      train_data = mnist.train.images
      train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
      eval_data = mnist.test.images
      eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

      def model_fn(features, labels, mode):
        return cnn_model_fn(
          features, labels, mode, 
          filters_counts=options["filters_counts"], 
          filter_sizes=options["filter_sizes"]
        )

      # Create the estimator
      mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=options["model_dir"]
      )

      # Set up logging for predictions
      tensors_to_log = {
        "probabilities": "softmax_tensor"
      }
      logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=100
      )

      # Train the model
      train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
      )
      mnist_classifier.train(
        input_fn=train_input_fn,
        steps=STEPS,
        hooks=[logging_hook]
      )

      # Evaluate the model and print results
      eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
      )
      eval_results = mnist_classifier.evaluate(
        input_fn=eval_input_fn
      )

      return eval_results

if __name__ == "__main__":
  worker = Worker()
  worker.start()
  