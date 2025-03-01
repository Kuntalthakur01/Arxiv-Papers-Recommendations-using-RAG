from typing import List, Optional

import torch
from llama_index.core import Document, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import ImageDocument, NodeWithScore
from PIL import Image
from transformers import Blip2Model, Blip2Processor


# WARNING: Implementation Not Complete. Future Work. You can use this as reference
# The BLIP model is 10gb and does not fit in our VRAM. Can use this as a starting point
# to implement BLIPReranker on your own if you have enough resources.
class BLIPReranker(BaseNodePostprocessor):
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2Model.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.bfloat16
        )
        model.to(device)  # type: ignore

        super().__init__(device=device, processor=processor, model=model)

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:

        texts = []
        images = []
        for node in nodes:
            if type(node) == ImageDocument:
                images.append(Image.open(node.image_path))
            else:
                texts.append(node.text)

        inputs = self.processor(images=images, text=texts, return_tensors="pt").to(
            self.device, torch.bfloat16
        )  # type:ignore

        outputs = self.model(**inputs)
        print(outputs)

        return nodes
