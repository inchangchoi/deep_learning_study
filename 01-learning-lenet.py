
from pylab import *
import caffe
from caffe import layers as L, params as P
import os

#####################################################
# Params
#####################################################
GPU_ENABLED = False
FIGURE_COUNT = 0
caffe_root = '/home/parallels/dev/caffe-master/'

#####################################################
# Prepare Data
#####################################################
os.chdir(caffe_root)
# Download data
os.system('data/mnist/get_mnist.sh')
# Prepare data
os.system('examples/mnist/create_mnist.sh')
# back to examples
os.chdir('examples')


#####################################################
# Creating the Net
#####################################################
def lenet(lmdb, batch_size):
    n = caffe.NetSpec()
    [n.data, n.label] = L.Data(batch_size = batch_size, backend = P.Data.LMDB, source = lmdb,\
                                                transform_param=dict(scale = 1/255.0), ntop = 2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))


#####################################################
# Loading and Checking the Solver
#####################################################
if GPU_ENABLED:
    caffe.set_device(0)
    caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')

print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

print solver.net.forward()  # train net
print solver.test_nets[0].forward()  # test net (there can be more than one)

# we use a little trick to tile the first eight images
figure(FIGURE_COUNT)
FIGURE_COUNT = FIGURE_COUNT + 1
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print 'train labels:', solver.net.blobs['label'].data[:8]
figure(FIGURE_COUNT)
FIGURE_COUNT = FIGURE_COUNT + 1
imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]

#####################################################
# Stepping the Solver
#####################################################
solver.step(1)
figure(FIGURE_COUNT)
FIGURE_COUNT = FIGURE_COUNT + 1
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4 * 5, 5 * 5), cmap='gray');
axis('off')

#####################################################
# Writing Custom Training Loop
#####################################################
# % % time
niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4


#####################################################
# Draw plot
#####################################################
FIGURE_COUNT = FIGURE_COUNT + 1
f, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))

for i in range(8):
    figure(num = FIGURE_COUNT, figsize=(2, 2))
    FIGURE_COUNT = FIGURE_COUNT + 1
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(num = FIGURE_COUNT, figsize=(10, 2))
    FIGURE_COUNT = FIGURE_COUNT + 1
    imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')

show()