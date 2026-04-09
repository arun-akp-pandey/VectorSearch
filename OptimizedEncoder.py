import os
import torch
import timm
from PIL import Image
from pymilvus import MilvusClient, DataType

# --- 1. Efficient Embedding Extraction ---
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

# --- 2. Milvus Optimization Strategy ---
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

# --- 3. Batch Insertion (Optimized for Throughput) ---
#def add_images(paths):
#   data = [{"vector": encoder.get_normalized_vector(p), "path": p} for p in paths]
#    client.insert(collection_name=COLLECTION, data=data)

def add_images_from_directory():
    paths = []
    id = 0
    for filename in os.listdir("Images"):
        p = "Images/" + filename
        data = []
        id = id + 1
        data.append({"vector": encoder.get_normalized_vector(p), "path": p, "id": id})
        client.insert(collection_name=COLLECTION, data=data)
    

# add_images_from_directory()

def get_all_entities():
    # Replace 'my_collection' with your collection name
    results = client.query(
    collection_name=COLLECTION,
    filter="",            # Empty filter retrieves all items
    output_fields=["path"],  # "*" returns all scalar fields (but not the vector)
    limit=100             # Good practice to limit results for large databases
    )

    for item in results:
        print(item)

get_all_entities()

def drop_collection():
    client.drop_collection(collection_name=COLLECTION)


# drop_collection()