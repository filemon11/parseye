import predictor
import sys

args = sys.argv

dt = predictor.DualTransformer.create_aj_vanilla_from_pretrained(args[4], args[5], args[2], shift = args[3])
dt.train("../readingtimes/training_data.csv", "../readingtimes/trial_data.csv", "../readingtimes/test_data.csv", args[1], lr = 0.01, momentum = 0, dampening = 0,
         weight_decay = 0, wandb = False, checkpoint = False, clip = 5.0, epochs = 100)
