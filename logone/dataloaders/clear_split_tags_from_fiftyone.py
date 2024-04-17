import fiftyone as fo 
import argparse
from tqdm import tqdm


def clear_split_tags_from_fiftyone(dataset):
    # Load the dataset if it exists
    try: 
        dataset = fo.load_dataset(dataset)
    except Exception as e: 
        print(f"Error loading dataset: {e}")
        return

    # Clear any tag that ends with "train" "test" or "val" and 
    # Doesn't start wth "default_split", separated by "/" 
    tags_cleared = 0

    for sample in tqdm(dataset):
        for tag in sample.tags:
            if tag.endswith("train") or tag.endswith("test") or tag.endswith("val"):
                if not tag.startswith("default_split/"):
                    sample.tags.remove(tag)
                    sample.save()
                    tags_cleared += 1
    
    print(f"Cleared {tags_cleared} tags from dataset {dataset.name}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Clears split tags from a fiftyone dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the fiftyone dataset to clear split tags from")
    args = parser.parse_args()

    clear_split_tags_from_fiftyone(args.dataset)