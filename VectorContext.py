# 1. Package Import
import os
import torch
import timm
from PIL import Image
from pymilvus import MilvusClient, DataType

# 2. Updated Encoder for Vision Transformer
class OptimizedEncoder:
    def __init__(self):
        # Swapping ResNet-50 for ViT-Base for better geometric/logo recognition
        self.model_name = "vit_base_patch16_224" 
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model.eval()
        config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**config)
    def get_normalized_vector(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
        # Current output shape: [1, 197, 768] (Batch, Patches+1, Dim)
            out = self.model(tensor)
        # If the model returns the full sequence, take only the CLS token (index 0)
        if out.ndim == 3:
            vec = out[0, 0, :] # Extract the first token's embedding
        else:
            vec = out.squeeze(0) # For models that already pool the result
        norm_vec = torch.nn.functional.normalize(vec, p=2, dim=0)
        return norm_vec.tolist()
encoder = OptimizedEncoder()

# 3. Milvus Optimization Strategy (Updated Dimension)
client = MilvusClient("optimized_search.db")
COLLECTION = "image_vault"
if not client.has_collection(COLLECTION):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE", 
        index_type="IVF_SQ8",  
        params={"nlist": 1024}
    )
    client.create_collection(
        collection_name=COLLECTION,
        # ViT-Base outputs 768 dimensions (ResNet was 2048)
        dimension=768, 
        index_params=index_params
    )

# 4. Batch Insertion
def add_images_from_directory():
    data = []
    # Ensure directory exists before running
    if not os.path.exists("Images"):
        return "Images directory not found."    
    for id, filename in enumerate(os.listdir("Images"), start=1):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            p = os.path.join("Images", filename)
            data.append({
                "id": id,
                "vector": encoder.get_normalized_vector(p), 
                "path": p 
            })
    if data:
        message = client.insert(collection_name=COLLECTION, data=data)
        return message
    return "No images found to insert."

# 5. Milvus Drop Command
def drop_collection():
    client.drop_collection(collection_name=COLLECTION)
    return COLLECTION
    
# 6. Milvus Read Query
def get_all_entities():
    results = client.query(
        collection_name=COLLECTION,
        filter="",
        output_fields=["path"],
        limit=100
    )
    return results

# 7. Search By Image
def search_by_image(query_image_path):
    query_vector = encoder.get_normalized_vector(query_image_path)
    results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],
        limit=1,
        output_fields=["path"],
        search_params={"metric_type": "COSINE"}
    )
    if results and results[0]:
        match = results[0][0]
        image_path = match['entity']['path']
        score = match['distance'] 
        return (image_path, score)
    return None