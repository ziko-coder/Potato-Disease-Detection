from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Early Blight", "Late Blight", "Healthy"]
BUCKET_NAME = "potato-example"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

    
def process_image(file_data, img_size=(250, 250)):
    try:
        # Read the file data and convert it to a PIL Image object
        image = Image.open(BytesIO(file_data))
        # Resize the image to the desired dimensions
        image = image.resize(img_size)
        # Convert the image to RGB if it's not already in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert the PIL Image to a NumPy array and return it
        return np.array(image)
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model.h5",
            "/tmp/potatoes.h5",
        )
        model = tf.keras.models.load_model("/tmp/potatoes.h5")
    try:
        # Read the file data and process the image
        image = process_image(file.read())

        if image is None:
            raise ValueError("Failed to process the image.")

        # Add any additional image preprocessing if required
        # ...

        # Prepare the image for prediction
        img_batched = np.expand_dims(image, 0)

        # Rest of the code for model prediction
        prediction = model.predict(img_batched)
        pred_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        return {
            'class': pred_class,
            'confidence': float(confidence)
        }

    except Exception as e:
        return {"error": str(e)}