from tensor import Tensor
from model import Conv2D, AvgPool2D, MaxPool2D, Flatten, Linear, ReLU, Sigmoid


def load_data(file_path):
    images = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        vec = list(map(float, line.strip().split(',')))
        image_tensor = Tensor(vec[:-1]).reshape(8,8) * (1/16)
        image = image_tensor.data
        label = int(vec[-1])
        images.append(image)
        labels.append(label)
    return images, labels

def load_config(file_path):
    layers = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if not line.strip():
            continue
        layer, *params = line.strip().split()
        params_dict = {k: int(v) for k, v in (p.split('=') for p in params)}

        if layer == "Conv2D":
            in_channels = params_dict.get('in_channels', 0)
            out_channels = params_dict.get('out_channels', 0)
            if in_channels == 0 or out_channels == 0:
                raise ValueError('The config is invalid')
            kernel_size = params_dict.get('kernel_size', 3)
            stride = params_dict.get('stride', 1)
            padding = params_dict.get('padding', 0)
            dilation = params_dict.get('dilation', 1)
            weights = params_dict.get('weights', None)
            use_bias = params_dict.get('use_bias', True)
            bias = params_dict.get('bias', None)
            layers.append(Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, weights, use_bias, bias))
        
        elif layer == "AvgPool2D":
            kernel_size = params_dict.get('kernel_size', 2)
            stride = params_dict.get('stride', 2)
            layers.append(AvgPool2D(kernel_size, stride))

        elif layer == "MaxPool2D":
            kernel_size = params_dict.get('kernel_size', 2)
            stride = params_dict.get('stride', 2)
            layers.append(MaxPool2D(kernel_size, stride))
        
        elif layer == "Flatten":
            layers.append(Flatten())
        
        elif layer == "Linear":
            input_size = params_dict.get('input_size', 0)
            output_size = params_dict.get('output_size', 0)
            if input_size == 0 or output_size == 0:
                raise ValueError('The config is invalid')
            weights = params_dict.get('weights', None)
            layers.append(Linear(input_size, output_size, weights))
        
        elif layer == 'ReLU':
            layers.append(ReLU())
        
        elif layer == 'Sigmoid':
            layers.append(Sigmoid())
    return layers


def arg_max(vec):
    max_idx = 0
    for i in range(len(vec)):
        if vec[i] > vec[max_idx]: max_idx = i
    return max_idx


if __name__ == "__main__":
    print(load_config('config.txt'))