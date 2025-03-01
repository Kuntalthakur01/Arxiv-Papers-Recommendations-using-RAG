import json
import os

from torch.utils.data import Dataset


class ArxivPaperDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.paper_folders = [
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        ]

    def __len__(self) -> int:
        return len(self.paper_folders)

    def __getitem__(self, index):
        paper_folder = self.paper_folders[index]
        metadata = {}

        with open(os.path.join(paper_folder, "metadata.json"), "r") as f:
            metadata = json.load(f)

        sections = {}
        with open(os.path.join(paper_folder, metadata["text_file"])) as f:
            sections = json.load(f)

        images = []
        for image_metadata in metadata["images"]:
            image_path = os.path.join(
                paper_folder, "images", f'{image_metadata["image_id"]}.png'
            )
            if os.path.exists(image_path):
                images.append(
                    {
                        "id": image_metadata["image_id"],
                        "path": image_path,
                    }
                )
        # print(images, metadata["equations"])
        return {
            "paper_path": paper_folder,
            "sections": sections,
            "images": images,
            "equations": metadata["equations"],
        }
