from torchvision import transforms

def transforms_cpu_train():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def transforms_cpu_val(image_size=(252, 252)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

def transforms_gpu_train(
    image_size=(252, 252),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    hflip_p=0.5,
    vflip_p=0.5,
    rotation_deg=15,
    color_jitter=(0.2, 0.2, 0.2, 0.05),
    affine_scale=(0.85, 1.15),
    affine_translate=(0.1, 0.1),
    crop_scale=(0.8, 1.0),
):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomVerticalFlip(p=vflip_p),
        transforms.RandomAffine(
            degrees=rotation_deg,
            translate=affine_translate,
            scale=affine_scale,
        ),
        transforms.ColorJitter(
            brightness=color_jitter[0],
            contrast=color_jitter[1],
            saturation=color_jitter[2],
            hue=color_jitter[3],
        ),
        transforms.Normalize(mean, std),
    ])

def transforms_gpu_val(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Normalize(mean, std),
    ])
