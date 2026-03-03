from torchvision import transforms


def transforms_train(
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
    train = transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.15), int(image_size[1] * 1.15))),
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomVerticalFlip(p=vflip_p),
        transforms.RandomRotation(rotation_deg),
        transforms.RandomAffine(
            degrees=0,
            translate=affine_translate,
            scale=affine_scale,
        ),
        transforms.ColorJitter(
            brightness=color_jitter[0],
            contrast=color_jitter[1],
            saturation=color_jitter[2],
            hue=color_jitter[3],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train

def transforms_val(image_size=(252, 252),mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)):
    val = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return val
