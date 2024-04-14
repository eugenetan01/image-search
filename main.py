from dotenv import load_dotenv
import os
from glob import glob
from math import ceil
import os
from pathlib import Path
from random import choices
import re

import cv2
import matplotlib.pyplot as plt
from PIL import Image

# I'm using MongoDB as my vector database:
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, DuplicateKeyError
from pymongo.operations import SearchIndexModel

from sentence_transformers import (
    SentenceTransformer,
)  # The transformer used to execute the clip model.
from tqdm.notebook import tqdm

# Specify the path to your alternative file
dotenv_path = os.path.join(os.path.dirname(__file__), ".secrets")

# Load the environment variables from your custom file
load_dotenv(dotenv_path)
print(os.getenv("mongo_uri"))

DATABASE_NAME = "image_search_demo"
IMAGE_COLLECTION_NAME = "images"

# Change this to 1000 to load a suitable number of images into MongoDB:
NUMBER_OF_IMAGES_TO_LOAD = 1000

# Set this as an environment variable to avoid accidentally sharing your cluster credentials:
MONGODB_URI = os.environ["mongo_uri"]

client = MongoClient(MONGODB_URI)
db = client.get_database(DATABASE_NAME)
model = SentenceTransformer("clip-ViT-L-14")
# collection = db.get_collection(IMAGE_COLLECTION_NAME)
# Ensure the collection exists, because otherwise you can't add a search index to it.


def setup():
    try:
        db.create_collection(IMAGE_COLLECTION_NAME)
    except CollectionInvalid:
        # This is raised when the collection already exists.
        print("Images collection already exists")

    # Add a search index (if it doesn't already exist):
    collection = db.get_collection(IMAGE_COLLECTION_NAME)
    if len(list(collection.list_search_indexes(name="default"))) == 0:
        print("Creating search index...")
        collection.create_search_index(
            SearchIndexModel(
                {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "dimensions": 768,
                                "similarity": "cosine",
                                "type": "knnVector",
                            }
                        },
                    }
                },
                name="default",
            )
        )
        print("Done.")
    else:
        print("Vector search index already exists")
    return collection


def load_images(image_count, collection):
    """
    Load `image_count` images into the database, creating an embedding for each using the sentence transformer above.

    This can take some time to run if image_count is large.

    The image's pixel data is not loaded into MongoDB, just the image's path and vector embedding.
    """
    image_paths = choices(glob("images/**/*.JPEG", recursive=True), k=image_count)
    for path in tqdm(image_paths):
        emb = model.encode(Image.open(path))
        try:
            collection.insert_one(
                {
                    "_id": re.sub("images/", "", path),
                    "embedding": emb.tolist(),
                }
            )
        except DuplicateKeyError:
            pass


def display_images(docs, cols=3, show_paths=False):
    """
    Helper function to display some images in a grid.
    """
    for doc in docs:
        doc["image_path"] = "images/" + doc["_id"]

    rows = ceil(len(docs) / cols)

    f, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 8), tight_layout=True)
    for i, doc in enumerate(docs):
        image_path = doc["image_path"]
        score = doc["score"]
        image = cv2.imread(image_path)[:, :, ::-1]
        axis = axarr[i // cols, i % cols]
        axis.imshow(image)
        axis.axis("off")
        if show_paths:
            axis.set_title(image_path.rsplit("/", 1)[1])
        else:
            axis.set_title(f"Score: {score:.4f}")
    plt.show()


def image_search(search_phrase):
    """
    Use MongoDB Vector Search to search for a matching image.

    The `search_phrase` is first converted to a vector embedding using
    the `model` loaded earlier in the Jupyter notebook. The vector is then used
    to search MongoDB for matching images.
    """
    emb = model.encode(search_phrase)
    cursor = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": "default",
                    "path": "embedding",
                    "queryVector": emb.tolist(),
                    "numCandidates": 100,
                    "limit": 9,
                }
            },
            {"$project": {"_id": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]
    )

    return list(cursor)


collection = setup()
# load_images(NUMBER_OF_IMAGES_TO_LOAD, collection)
display_images(image_search("tiger"))
