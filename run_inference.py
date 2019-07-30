#%%
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
#%%
pb_path = './mobilenet_aesthetic'
graph_def = tf.GraphDef()
#%%
op = 'dense_1/Softmax:0'
output_layer = 'Placeholder_136:0'
input_node = 'input_1:0'
#%%

#%%
def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def resize_to_224_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (224, 224), interpolation = cv2.INTER_LINEAR)

file_name = 'bad_q1'
filename = './'+file_name+'.jpg'
# Load from a file
imageFile = filename
image = Image.open(imageFile)

# Convert to OpenCV format
image = convert_to_opencv(image)

# Resize that square down to 256x256
augmented_image = resize_to_224_square(image)
#%%
# print(augmented_image.shape)
#%%
with tf.Session(graph=tf.Graph()) as sess:
    '''
    You can provide 'tags' when saving a model,
    in my case I provided, 'serve' tag 
    '''

    tf.saved_model.loader.load(sess, ['serve'], pb_path)
    graph = tf.get_default_graph()

    # print your graph's ops, if needed
    # print(graph.get_operations())

    try:
        prob_tensor = sess.graph.get_tensor_by_name(op)
        predictions, = sess.run(prob_tensor,feed_dict={input_node: [augmented_image]})    
    except KeyError:
        print ("Couldn't find classification output layer: " + output_layer + ".")
        print ("Verify this a model exported from an Object Detection project.")
        exit(-1)


#%%
# print(predictions)
def get_mean_score(score):
    buckets = np.arange(1, 11)
    mu = (buckets * score).sum()
    return mu


def get_std_score(scores):
    si = np.arange(1, 11)
    mean = get_mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

output1 = predictions
mean1 = get_mean_score(output1)
std1 = get_std_score(output1)
print(mean1,std1,'\n','for',file_name+' asthetic')

