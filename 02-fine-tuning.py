
from pylab import *
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import os
import tempfile


#####################################################
# Params
#####################################################
GPU_ENABLED = False
FIGURE_COUNT = 0
caffe_root = '/home/parallels/dev/caffe-master/'

if GPU_ENABLED:
    caffe.set_device(0)
    caffe.set_mode_gpu()

#####################################################
# Aux Functions
#####################################################
# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

#####################################################
# Prepare the Data
#####################################################
 #Download just a small subset of the data for this exercise.
# (2000 of 80K images, 5 of 20 labels.)
# To download the entire dataset, set `full_dataset = True`.
full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 2000
    NUM_STYLE_LABELS = 5

# This downloads the ilsvrc auxiliary data (mean file, etc),
# and a subset of 2000 images for the style recognition task.
os.chdir(caffe_root)  # run scripts from caffe root
os.system('data/ilsvrc12/get_ilsvrc_aux.sh')
os.system('scripts/download_model_binary.py ' + caffe_root + 'models/bvlc_reference_caffenet')
os.system('python examples/finetune_flickr_style/assemble_data.py \
    --workers=-1  --seed=1701 \
    --images=$NUM_STYLE_IMAGES  --label=$NUM_STYLE_LABELS')
# back to examples
os.chdir('examples')