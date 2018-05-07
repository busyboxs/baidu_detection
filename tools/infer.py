import argparse
import gzip
import os

import net.alexnet as alexnet
import net.inception_resnet_v2 as inception_resnet_v2
import numpy as np
import paddle.v2 as paddle
import net.resnet as resnet
import net.vgg as vgg

from net import googlenet, inception_v4, xception

DATA_DIM = 3 * 224 * 224  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
CLASS_DIM = 102


def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_list',
        help='The path of data list file, which consists of one image path per line'
    )
    parser.add_argument(
        'model',
        help='The model for image classification',
        choices=[
            'alexnet', 'vgg13', 'vgg16', 'vgg19', 'resnet', 'googlenet',
            'inception-resnet-v2', 'inception_v4', 'xception'
        ])
    parser.add_argument(
        'params_path', help='The file which stores the parameters')
    args = parser.parse_args()

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))

    if args.model == 'alexnet':
        out = alexnet.alexnet(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg13':
        out = vgg.vgg13(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg16':
        out = vgg.vgg16(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg19':
        out = vgg.vgg19(image, class_dim=CLASS_DIM)
    elif args.model == 'resnet':
        out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM)
    elif args.model == 'googlenet':
        out, _, _ = googlenet.googlenet(image, class_dim=CLASS_DIM)
    elif args.model == 'inception-resnet-v2':
        assert DATA_DIM == 3 * 331 * 331 or DATA_DIM == 3 * 299 * 299
        out = inception_resnet_v2.inception_resnet_v2(
            image, class_dim=CLASS_DIM, dropout_rate=0.5, data_dim=DATA_DIM)
    elif args.model == 'inception_v4':
        out = inception_v4.inception_v4(image, class_dim=CLASS_DIM)
    elif args.model == 'xception':
        out = xception.xception(image, class_dim=CLASS_DIM)

    # load parameters
    with gzip.open(args.params_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    test_dir = 'data/test'

    file_list = [os.path.join(test_dir, line.strip()) for line in open(args.data_list)]
    test_data = [(paddle.image.load_and_transform(image_file, 256, 224, False)
                  .flatten().astype('float32'), ) for image_file in file_list]
    save_file = open('data/result.csv', 'w')
    for i in range(100):
        test_data_tmp = test_data[10*i:10*(i+1)]
        probs = paddle.infer(
            output_layer=out, parameters=parameters, input=test_data_tmp)
        lab = np.argsort(-probs)
        for file_name, result in zip(file_list, lab):
            save_file.write('{} {}\n'.format(os.path.split(file_name)[1], result[0]))
            print "Label of %s is: %d" % (file_name, result[0])

    save_file.close()


if __name__ == '__main__':
    main()
