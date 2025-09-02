from datasets import load_dataset
from PIL import Image


def get_dataset():
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
    num_shards = 46  # Number of webdataset tar files
    urls = [base_url.format(i=i) for i in range(num_shards)]
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    return dataset

dataset = get_dataset()

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
