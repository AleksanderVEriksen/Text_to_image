from datasets import load_dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import torch

MNIST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)
])

def load_urls(urls):
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    req = ['jpg', 'json']
    return dataset.remove_columns([col for col in dataset.column_names if col not in req])

def get_dataset(train:bool=True, test:bool=False, val: bool=False):
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
    num_shards = 46  # Number of webdataset tar files
    urls = [base_url.format(i=i) for i in range(num_shards)]
    
    train_end = int(num_shards*0.8)
    val_end = int(num_shards*0.9)

    train_urls = urls[:train_end]
    val_urls = urls[train_end:val_end]
    test_urls = urls[val_end:]

    if train:
        return load_urls(train_urls)
    if test:
        return load_urls(test_urls)
    if val:
        return load_urls(val_urls)
    else:
        return {
        "train": load_urls(train_urls),
        "val": load_urls(val_urls),
        "test": load_urls(test_urls)
    }

def get_mnist_dataset(train:bool=True, augment:bool=False):
    
    if augment:
        transform_list = [transforms.ToTensor()]
        transform_list = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
        ] + transform_list
        aug_transform = transforms.Compose(transform_list)
        dataset = MNIST(root='./data', train=train, download=True, transform=aug_transform)
    else:
        dataset = MNIST(root='./data', train=train, download=True, transform=MNIST_TRANSFORM)
    return dataset

def estimate_dataset_stats(dataloader, max_batches=20, device='cpu'):
    cnt = 0
    mean = 0
    M2 = 0
    pixels = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device).float()
            b = imgs.size(0)
            pixels += b * imgs[0].numel()
            batch_mean = imgs.mean()
            batch_var = imgs.var()
            delta = batch_mean - mean
            cnt += 1
            mean += delta / cnt
            M2 += batch_var
            if cnt >= max_batches:
                break
    return {"approx_mean": float(mean), "approx_var": float(M2 / cnt)}

# Example of iterating through the dataset
def show_sample_images(dataset, num_images=3):
    for i, image in enumerate(dataset):
        if i >= num_images:
            break
        print(f"\n--- Image Record nr: {i} ---\n")
        print(image)  # single image in row with associated columns
        print(type(image))
        print("\n---", image.keys() , "---\n")  # keys in the image dictionary
        if isinstance(image, dict) and 'jpg' in image:
            image['jpg'].show()
        else:
            print("No image found in this record.")
    # Note: The dataset is streamed, so it does not load all data into memory at once.
