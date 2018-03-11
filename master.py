import rpyc
import random
import numpy as np
import time
from rpyc.utils.server import ThreadedServer

RESULTS_FILE_PATH="./results.txt"
PORT=18861
MAX_FILTER_COUNT=32
MAX_FILTER_SIZE=10

class Master(rpyc.Service):
    def on_connect(self):
      print("A worker connected!")
  
    def on_disconnect(self):
        print("A worker disconnected!")

    def exposed_get_training_options(self):
      timestamp = time.time()
      options = {
        "model_dir": "./models/model{}".format(timestamp),
        "filters_counts": [
          random.randint(1, MAX_FILTER_COUNT), 
          random.randint(1, MAX_FILTER_COUNT), 
          random.randint(1, MAX_FILTER_COUNT)
        ], 
        "filter_sizes": [
          np.repeat(random.randint(1, MAX_FILTER_SIZE), 2).tolist(),
          np.repeat(random.randint(1, MAX_FILTER_SIZE), 2).tolist(),
          np.repeat(random.randint(1, MAX_FILTER_SIZE), 2).tolist()
        ]
      }
      print("Offering options {}...".format(options))
      return options

    def exposed_put_training_results(self, options, results):
      # Writes training results to file
      print("Received results {} for options {}...".format(results, options))
      f = open(RESULTS_FILE_PATH, "a")
      line = str({
        "options": options,
        "results": results
      })
      f.write("{}\n".format(line))
      f.flush()
      f.close()

if __name__ == "__main__":
  t = ThreadedServer(Master, port=PORT, protocol_config={"allow_all_attrs" : True})
  print("Running the master process on port {}...".format(PORT))
  t.start()