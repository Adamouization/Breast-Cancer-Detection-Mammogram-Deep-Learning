"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512
}
RESNET_IMG_SIZE = VGG_IMG_SIZE
VGG_IMG_SIZE_LARGE = {
    "HEIGHT": 2048,
    "WIDTH": 2048
}
INCEPTION_IMG_SIZE = {
    "HEIGHT": 299,
    "WIDTH": 299
}
XCEPTION_IMG_SIZE = INCEPTION_IMG_SIZE


# Variables set by command line arguments/flags
dataset = "mini-MIAS"       # The dataset to use.
model = "VGG"               # The model to use.
run_mode = "training"       # The type of running mode, either training or testing.
image_size = "small"        # Image resizing for VGG19 model.
batch_size = 2              # Batch size.
max_epoch_frozen = 100      # Max number of epochs when original CNN layers are frozen.
max_epoch_unfrozen = 50     # Max number of epochs when original CNN layers are unfrozen.
verbose_mode = False        # Boolean used to print additional logs for debugging purposes.
