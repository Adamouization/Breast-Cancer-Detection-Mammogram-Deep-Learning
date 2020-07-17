"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
MINI_MIAS_IMG_SIZE = {
    "HEIGHT": 1024,
    "WIDTH": 1024
}
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512
}
RESNET_IMG_SIZE = VGG_IMG_SIZE
INCEPTION_IMG_SIZE = {
    "HEIGHT": 299,
    "WIDTH": 299
}
XCEPTION_IMG_SIZE = INCEPTION_IMG_SIZE


# Variables set by command line arguments/flags
dataset = "mini-MIAS"       # The dataset to use.
mammogram_type = "all"      # The type of mammogram (Calc or Mass).
model = "VGG"               # The model to use.
run_mode = "training"       # The type of running mode, either training or testing.
batch_size = 2              # Batch size.
max_epoch_frozen = 100      # Max number of epochs when original CNN layers are frozen.
max_epoch_unfrozen = 50     # Max number of epochs when original CNN layers are unfrozen.
is_grid_search = False      # Run the grid search algorithm to determine the optimal hyper-parameters for the model.
verbose_mode = False        # Boolean used to print additional logs for debugging purposes.
