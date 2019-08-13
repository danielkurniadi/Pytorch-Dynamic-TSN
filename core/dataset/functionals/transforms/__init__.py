from PIL import Image
from torchvision import transforms as torch_transforms
from .group_transforms import (
    __group_resize,
    __group_center_crop,
    __group_random_crop,
    __group_random_horizontal_flip,
    __group_normalize,
    __stack,
)


def get_transform(opts, grayscale=False, interpolation=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(torch_transforms.Grayscale(1))

    if 'resize' in opts.preprocess:
        size = (opts.input_size, opts.input_size)
        transform_list.append(torch_transforms.Lambda(lambda imgs: __group_resize(imgs, size, interpolation)))
    elif 'scale' in opts.preprocess:
        transform_list.append(torch_transforms.Lambda(lambda imgs: __group_scale(imgs, opts.input_size, interpolation)))

    if 'random_crop' in opts.preprocess:
        transform_list.append(torch_transforms.Lambda(lambda imgs: __group_random_crop(imgs, opts.crop_size)))
    elif 'center_crop' in opts.preprocess:
        transform_list.append(torch_transforms.Lambda(lambda imgs: __group_center_crop(imgs, opts.crop_size)))

    if not opts.no_flip:
        invert = (opts.modality == 'Flow')
        transform_list.append(torch_transforms.Lambda(lambda imgs: __group_random_horizontal_flip(imgs, invert=invert)))

    transform_list.append(torch_transforms.Lambda(lambda imgs: __stack(imgs)))

    if convert:
        transform_list += [torch_transforms.ToTensor()]
    
    input_mean = opts.input_mean    # assumed list with length equals to dataset original channel
    input_std = opts.input_std      # assumed list with length equals to dataset original channel

    transform_list.append(torch_transforms.Lambda(
        lambda tensor: __group_normalize(tensor, input_mean, input_std)))

    return torch_transforms.Compose(transform_list)

