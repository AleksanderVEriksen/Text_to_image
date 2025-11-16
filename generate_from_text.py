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
    p.add_argument('--batch_size', type=int, default=32, help='Batch size used during training (for loading correct model weights)')
    return p.parse_args()

def parse_label_string(label_str, num_classes, num_samples, device):
    """Convert a string like '7', 'seven', '1,3,5', 'two four', 'all' into a LongTensor."""
    word_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    s = label_str.lower().strip()
    if s == 'all':
        vals = list(range(num_classes))
    else:
        # split by comma or whitespace
        parts = [p for seg in s.split(',') for p in seg.strip().split() if p]
        vals = []
        for p in parts:
            if p in word_map:
                vals.append(word_map[p])
            else:
                try:
                    v = int(p)
                    vals.append(v)
                except ValueError:
                    raise ValueError(f"Cannot parse label token '{p}'")
    if not vals:
        raise ValueError("Parsed no labels from input string.")
    # clamp to valid class range
    vals = [v for v in vals if 0 <= v < num_classes]
    if not vals:
        raise ValueError("No valid class indices after filtering.")
    # If single value, repeat to match num_samples
    if len(vals) == 1 and num_samples > 1:
        vals = vals * num_samples
    # If fewer than num_samples, tile
    if len(vals) < num_samples:
        reps = (num_samples + len(vals) - 1) // len(vals)
        vals = (vals * reps)[:num_samples]
    return torch.tensor(vals, dtype=torch.long, device=device)


def load_model(model, batch_size, model_name, device):
    """Helper function to load model weights with proper error handling"""
    possible_paths = [
        f"models/{batch_size}/{model_name}.pth",
        f"models/checkpoints/{batch_size}/{model_name}.pth",
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
    print(f"Using model type: {args.model_type} with in/out channels: {in_ch}/{out_ch}")
    model = BasicUNet(in_channels=in_ch, out_channels=out_ch, num_classes=args.num_classes).to(device) if args.model_type == 'Basic' else \
            UNET(in_channels=in_ch, out_channels=out_ch, num_classes=args.num_classes).to(device)

    # Load model weights using the helper function
    load_model(model, args.batch_size, args.model_name, device)

    scheduler = DDPMScheduler(
    num_train_timesteps=args.timesteps,
    beta_schedule="scaled_linear",
    beta_start=0.0001,
    beta_end=0.02,
    clip_sample=True
)
    scheduler.set_timesteps(args.timesteps)

    model.eval()
    with torch.no_grad():
        labels = parse_label_string(args.label, args.num_classes, args.num_samples, device)
        samples, timesteps_tensor = sample_images(model, scheduler, img_size=args.img_size, device=device, n=args.num_samples, Test=(in_ch==1), labels=labels, num_classes=args.num_classes)

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu()
    else:
        raise TypeError('sample_images returned unexpected type')

    torchvision.utils.save_image(samples, args.out, nrow=int(max(1, min(8, args.num_samples//2))), normalize=True)
    print(f"Saved generated samples to {args.out}")


if __name__ == '__main__':
    main()
