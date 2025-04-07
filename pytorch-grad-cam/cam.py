import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from models.resnet20_baseline import ResNet20 as ResNet20_baseline
from models.resnet20 import ResNet20
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST
import time

model_path = "/home/daniel/DLRT-Net-main/cifar10/results/resnet20"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam'
                        ],
                        help='CAM method')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    parser.add_argument('--model-path', type=str, default='resnet20_cifar10_baseline_0.0__best_weights.pt', 
                        help='Path to the model weights')
    parser.add_argument('--low-rank', action='store_true',
                        help='Use low rank approximation for the CAM computation')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM
    }

    if args.device=='hpu':
        import habana_frameworks.torch.core as htcore

    # model = models.resnet50(pretrained=True).to(torch.device(args.device)).eval()
    # Load model from `DLRT-Net-main/cifar10/results/resnet20/resnet20_cifar10_baseline_0.0__best_weights.pt`
    
    if args.model_path and args.low_rank:
        state_dict = torch.load(os.path.join(model_path, args.model_path))
        model = ResNet20(device=torch.device(args.device))
    elif args.model_path and not args.low_rank:
        state_dict = torch.load(os.path.join(model_path, args.model_path))
        model = ResNet20_baseline(device=torch.device(args.device))
    else:
        state_dict = torch.load(os.path.join(model_path, "resnet20_cifar10_baseline_0.0__best_weights.pt"))
        model = ResNet20_baseline()

    model.load_state_dict(state_dict)
    model = model.to(torch.device(args.device)).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    target_layers = [model.layer3]

    if os.path.isdir(args.image_path):
        # If the image path is a directory, load all images in the directory
        image_paths = [os.path.join(args.image_path, img) for img in os.listdir(args.image_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        rgb_imgs = []
        for img_path in image_paths:
            print(f'Processing image: {os.path.basename(img_path)}')
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            # Resize while maintaining aspect ratio
            h, w = rgb_img.shape[:2]
            scale = args.image_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Crop the center to (args.image_size, args.image_size)
            h, w = rgb_img.shape[:2]
            start_h, start_w = (h - args.image_size) // 2, (w - args.image_size) // 2
            rgb_img = rgb_img[start_h:start_h + args.image_size, start_w:start_w + args.image_size]
            rgb_imgs.append(rgb_img)

        input_tensor = torch.stack([
            preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(args.device)
            for img in rgb_imgs
        ]).squeeze()
    else:
        # If the image path is a single file, process it as before
        rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        h, w = rgb_img.shape[:2]
        scale = args.image_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Crop the center to (args.image_size, args.image_size)
        h, w = rgb_img.shape[:2]
        start_h, start_w = (h - args.image_size) // 2, (w - args.image_size) // 2
        rgb_img = rgb_img[start_h:start_h + args.image_size, start_w:start_w + args.image_size]
        rgb_imgs = [rgb_img]
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(args.device)

    print(f'Input tensor shape: {input_tensor.shape}')
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(3)]
    # targets = [ClassifierOutputReST(281)]
    targets = None
    # Targets: Airplane(0), Cat&Dog(3, 5), Dog(5), Automobile(1), Cat(3)
    # targets = [
    #     [ClassifierOutputTarget(0)],
    #     [ClassifierOutputTarget(3), ClassifierOutputTarget(5)],
    #     [ClassifierOutputTarget(5)],
    #     [ClassifierOutputTarget(1)],
    #     [ClassifierOutputTarget(3)]
    # ]
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    times = []
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       low_rank=args.low_rank) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        cam.time_list = times

        # grayscale_cam = grayscale_cam[0, :]
        cam_images = []
        for i in range(input_tensor.shape[0]):
            # start = time.time()
            grayscale_cam = cam(input_tensor=input_tensor[i:i+1],
                            targets=targets[i] if targets is not None else None,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)
            
            cam_image = show_cam_on_image(rgb_imgs[i], grayscale_cam[0, :], use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cam_images.append(cam_image)
            # end = time.time()
            # times.append(end - start)
        # Stack cam images on new first axis
        cam_images = np.stack(cam_images, axis=0)

    # gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
    # print(f"Input tensor shape: {input_tensor.shape}")
    # gb = gb_model(input_tensor, target_category=None)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    print(f"Output shape: {cam_images.shape}")
    print(f"Times: {times}")
    # print(f"Output shape: {gb.shape}")
    # print(f"Output shape: {cam_gb.shape}")
    os.makedirs(args.output_dir, exist_ok=True)

    # gb_output_path = os.path.join(args.output_dir, f'{args.method}_{image_name}_gb.jpg')
    # cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_{image_name}_cam_gb.jpg')
    times = np.array(times)
    times_path = os.path.join(args.output_dir, f'{args.method}_times.npy')
    if os.path.isdir(args.image_path):
        image_paths = [os.path.join(args.image_path, img) for img in os.listdir(args.image_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        for i, cam_img in enumerate(cam_images):
            iname = os.path.basename(image_paths[i]).split('.')[0]
            cv2.imwrite(os.path.join(args.output_dir, f'{args.method}_{iname}_cam.jpg'), cam_img)
        # np.save(times_path, times)
    else:
        image_name = os.path.basename(args.image_path).split('.')[0]
        cam_output_path = os.path.join(args.output_dir, f'{args.method}_{image_name}_cam.jpg')
        cv2.imwrite(cam_output_path, cam_images[0])
        # np.save(times_path, times)
    
        

    # cv2.imwrite(cam_gb_output_path, cam_gb)
    # cv2.imwrite(gb_output_path, gb)
