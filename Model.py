import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

# 1 读取训练数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 2 建立模型
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 3 损失函数   交叉熵
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 4 优化训练
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 5 开启会话
with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())

    # 训练
    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
    print(accuracy)

    # 保存模型
    model_path = "model/"
    model_name = "model_" + (str(accuracy))[:4] + "%"
    tf.train.Saver().save(sess, model_path + model_name)

def pred(image):
    img = tf.read_file(image)
    im_3 = tf.image.decode_jpeg(img, channels=3)
    im_resize = tf.image.resize_images(im_3, [28, 28])
    im_gry = tf.squeeze(tf.image.rgb_to_grayscale(im_resize), 2)
    im_reshape = tf.reshape(im_gry, [1, 784])

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, "model/model_91.0%")
        xx = sess.run(im_reshape)
        result = sess.run(tf.argmax(y, 1), feed_dict={x: xx})
        for num in result:
            return num

f = open('mnist_test_s.txt','w')
file = "mnist/mnist_test_s"
img_list = os.listdir("mnist/mnist_test_s")
for root, dirs, files in os.walk(file):
    for name in files:
        res = pred(os.path.join(root, name))
        print(os.path.join(name)+' '+str(res))
        f.write(os.path.join(name)+' '+str(res)+'\n')
f.close()
