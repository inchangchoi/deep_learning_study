
import caffe
import numpy as np
import os
import matplotlib.pyplot as plt
# import cv2

GPU_ENABLED = False
FIGURE_COUNT = 0

#matplotlib inline
# plt.interactive(True)

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


caffe_root = '/home/parallels/dev/caffe-master/'
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    # !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
    os.system(caffe_root + 'scripts/download_model_binary.py ../models/bvlc_reference_caffenet')


#####################################################
# Model Definition and Load Weights
#####################################################
# Load net and setup input preprocessing
caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

#####################################################
# Forward Propagation
#####################################################
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.figure(FIGURE_COUNT)
FIGURE_COUNT = FIGURE_COUNT + 1
plt.imshow(image)
# plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


#####################################################
# Check Classification Result
#####################################################
# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system(caffe_root + 'data/ilsvrc12/get_ilsvrc_aux.sh')

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])


#####################################################
# GPU mode
#####################################################
if GPU_ENABLED:
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    net.forward()  # run once before timing to set up memory
    # for each layer, show the output shape
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    for layer_name, param in net.params.iteritems():
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

#####################################################
# Define Visualization
#####################################################
def vis_square(data):
    global  FIGURE_COUNT
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure(FIGURE_COUNT)
    FIGURE_COUNT = FIGURE_COUNT + 1
    plt.imshow(data);
    plt.axis('off')

#####################################################
# Show conv1 layer
#####################################################
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
#####################################################
# Show conv1 blob (response)
#####################################################
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)
#####################################################
# Show pool5 blob (response)
#####################################################
feat = net.blobs['pool5'].data[0]
vis_square(feat)
#####################################################
# Show FC 6
#####################################################
feat = net.blobs['fc6'].data[0]
plt.figure(FIGURE_COUNT)
FIGURE_COUNT = FIGURE_COUNT + 1
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
#####################################################
# Show Final Probability
#####################################################
feat = net.blobs['prob'].data[0]
plt.figure(num = FIGURE_COUNT, figsize=(15, 3))
FIGURE_COUNT = FIGURE_COUNT + 1
plt.plot(feat.flat)

#####################################################
# Try my image
#####################################################
# download an image
# my_image_url = "http://media.caranddriver.com/images/13q4/543506/2014-hyundai-sonata-photos-and-info-news-car-and-driver-photo-545351-s-original.jpg"  # paste your URL here
my_image_url = 'http://assets.worldwildlife.org/photos/1620/images/story_full_width/bengal-tiger-why-matter_7341043.jpg?1345548942'
# for example:
# my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
os.system('wget -O ./image.jpg ' + my_image_url)

# transform it and copy it into the net
image = caffe.io.load_image('./image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]

plt.figure(FIGURE_COUNT)
FIGURE_COUNT = FIGURE_COUNT + 1
plt.imshow(image)


print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])

plt.show()