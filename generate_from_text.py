import argparse
import torch
import torchvision
from diffusers import DDPMScheduler
from model import UNET, BasicUNet
from utils import sample_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='model', help='Saved model filename (in ./models/)')
    p.add_argument('--model_type', type=str, default='UNET', choices=['UNET', 'Basic'])
    p.add_argument('--label', type=str, required=True, help='Text label to condition on (e.g. "7" or "seven")')
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--img_size', type=int, default=28)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--out', type=str, default='generated.png')
    return p.parse_args()


def load_model(model, model_name, device, timesteps):
    """Helper function to load model weights with proper error handling"""
    possible_paths = [
        f"models/{model_name}.pth",
        f"models/checkpoints/{model_name}.pth",
        f"models/{model_name}_EMA_BS_32_MaxT_{timesteps}.pth",
        f"models/EMA/{model_name}_EMA_BS_32.pth"
    ]
    
    for path in possible_paths:
        try:
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Full checkpoint with training state
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model state from checkpoint: {path}")
                return True
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights from: {path}")
                return True
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading from {path}: {str(e)}")
            continue
    
    print("\nWarning: Could not load model weights from any of:")
    for path in possible_paths:
        print(f"- {path}")
    print("Proceeding with random weights.\n")
    return False


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_ch = 1 if args.img_size <= 28 else 3
    out_ch = in_ch
    model = BasicUNet(in_channels=in_ch, out_channels=out_ch, num_classes=args.num_classes).to(device) if args.model_type == 'Basic' else \
            UNET(in_channels=in_ch, out_channels=out_ch, num_classes=args.num_classes).to(device)

    # Load model weights using the helper function
    load_model(model, args.model_name, device, args.timesteps)

    scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_start=0.0001, beta_end=0.02)
    scheduler.set_timesteps(args.timesteps)

    model.eval()
    with torch.no_grad():
        samples = sample_images(model, scheduler, img_size=args.img_size, device=device, n=args.num_samples, Test=(in_ch==1), labels=args.label, num_classes=args.num_classes)

    # sample_images may return (samples, intermediates)
    if isinstance(samples, (list, tuple)):
        samples = samples[0]

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu()
    else:
        raise TypeError('sample_images returned unexpected type')

    torchvision.utils.save_image(samples, args.out, nrow=int(max(1, min(8, args.num_samples//2))), normalize=True)
    print(f"Saved generated samples to {args.out}")


if __name__ == '__main__':
    main()
