#1. Package Import
import os
import torch
import timm
from PIL import Image
from pymilvus import MilvusClient, DataType

#2. Efficient Embedding Extraction
class OptimizedEncoder:
    def __init__(self):
        # Using a mid-sized model for balance between speed and accuracy
        self.model = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.model.eval()
        config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**config)
    def get_normalized_vector(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = self.model(tensor).squeeze(0)
            # Manual normalization ensures maximum compatibility with IP/Cosine
            norm_vec = torch.nn.functional.normalize(vec, p=2, dim=0)
        return norm_vec.tolist()
encoder = OptimizedEncoder()

#3. Milvus Optimization Strategy
client = MilvusClient("optimized_search.db")
COLLECTION = "image_vault"
if not client.has_collection(COLLECTION):
    # IVF_SQ8 is the 'Sweet Spot' for most production image searches
    # It compresses float32 to uint8, saving massive RAM with ~2% recall drop
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE", 
        index_type="IVF_SQ8",  
        params={"nlist": 1024} # Increase nlist for larger datasets (>1M)
    )
    client.create_collection(
        collection_name=COLLECTION,
        dimension=2048, # ResNet50 dimension
        index_params=index_params
    )

#4. Batch Insertion (Optimized for Throughput)
def add_images_from_directory():
    data = []
    id = 0
    for filename in os.listdir("Images"):
        p = "Images/" + filename
        id = id + 1
        data.append({"vector": encoder.get_normalized_vector(p), "path": p, "id": id})
    message = client.insert(collection_name=COLLECTION, data=data)
    return message

#5. Milvus Drop Command
def drop_collection():
    client.drop_collection(collection_name=COLLECTION)
    return COLLECTION
    
#6. Milvus Read Query
def get_all_entities():
    results = client.query(
    collection_name=COLLECTION,
    filter="",                # Empty filter retrieves all items
    output_fields=["path"],
    limit=100             # Good practice to limit results for large databases
    )
    return results

#7. Search By Image
def search_by_image(query_image_path):
    # Convert query image to vector
    query_vector = encoder.get_normalized_vector(query_image_path)
    
    # Search Milvus using COSINE similarity
    results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],
        limit=1,
        output_fields=["path"],
        search_params={"metric_type": "COSINE"} # Double check metric
    )
    
    if results and results[0]:
        match = results[0][0]
        image_path = match['entity']['path']
        score = match['distance'] # In Cosine, 1.0 is identical
        
        return (image_path, score)
    else:
        return None



