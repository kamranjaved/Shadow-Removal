import numpy as np
import os
from skimage import io
import html
import tensorflow as tf

def save_examples(img, img_dir, name, num=None, label=np.array([None])):
    # img pixel value: (-1,1)

    if num == None:
        num = len(img)

    img = (img[0:num]+1)*127.5
    img = img.astype('uint8')

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    if img.shape[3] == 1:
        img = np.resize(img,(num,img.shape[1],img.shape[2]))
    
    if len(label) < num:
        label = np.array([None])

    if label.all() != None:
        for i in range(num):
            io.imsave(img_dir+'/'+str(name[i])+'_'+str(label[i])+'.jpg',img[i])
    else:
        for i in range(num):
            io.imsave(img_dir+'/'+str(name[i])+'.jpg',img[i])
    
    return


def write_html(path_base, label=None, input_labels=np.array([None]), output_labels=np.array([None]), target_labels=np.array([None])):

    inputs = os.listdir(path_base+'/inputs')
    inputs.sort()
    outputs = os.listdir(path_base+'/outputs')
    outputs.sort()
    targets = os.listdir(path_base+'/targets')
    targets.sort()

    html = open(path_base+'/result.html', 'w')
    html.write('<html><body><table>')
    html.write('<tr><th>NAME</th><th>INPUT</th><th>OUTPUT</th><th>TARGET</th></tr>')

    for i in range(len(inputs)):
        html.write('<tr>')
        html.write('<td><center>%s</center></td>' % inputs[i])
        html.write("<td><img src='%s'></td>" % ("inputs/"+inputs[i]))
        html.write("<td><img src='%s'></td>" % ("outputs/"+outputs[i]))
        html.write("<td><img src='%s'></td>" % ("targets/"+targets[i]))
        html.write('</tr>')
        
        # label
        if label != None:
            html.write('<tr>')
            html.write('<td><center>%s</center></td>' % label)
            if input_labels.all() != None:
                html.write("<td><center>%s</center></td>" % str(input_labels[i]))
            else:
                html.write("<td> </td>")
            if output_labels.all() != None:
                html.write("<td><center>%s</center></td>" % str(output_labels[i]))
            else:
                html.write("<td> </td>")
            if target_labels.all() != None:
                html.write("<td><center>%s</center></td>" % str(target_labels[i]))
            else:
                html.write("<td> </td>")
            html.write('</tr>')

    html.write('</table></body></html>')
    html.close()

    return


def write_txt(txt_dir, result, names, epoch):
    txtfile = open(txt_dir+'/log.txt', 'w')
    print("epoch: %d" % epoch, file=txtfile)
    print("generator_loss: %f" % result['g_loss'], file=txtfile)
    print("discriminator_loss: %f" % result['d_loss'], file=txtfile)
    print("similarity_loss: %f" % result['s_loss'], file=txtfile)

    print("kernel norms", file=txtfile)
    for i in range(len(names)):
        print(names[i], file=txtfile)
        print(result['kernel_norm'][i], file=txtfile)

    txtfile.close()

    return


def visualize_results(log_dir, result, label=False):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    name_list=[]
    for i in range(len(result['name'])):
        name_list.append(os.path.splitext(result['name'][i].decode('utf-8'))[0])

    save_examples(result['inp'], log_dir+'/inputs', name_list)
    if label:
        save_examples(result['gen'], log_dir+'/outputs', name_list, label=result['gen_score'])
        save_examples(result['target'], log_dir+'/targets', name_list, label=result['real_score'])
    else:
        save_examples(result['gen'], log_dir+'/outputs', name_list)
        save_examples(result['target'], log_dir+'/targets', name_list)


    return


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    if img1.get_shape()[3] == 3:
        img1 = tf.image.rgb_to_grayscale(img1)
    if img2.get_shape()[3] == 3:
        img2 = tf.image.rgb_to_grayscale(img2)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def loss_normalize(loss):
    loss_value = tf.Variable(1.0)
    
    loss_normalized = loss / loss_value.assign(loss)

    return loss_normalized

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

