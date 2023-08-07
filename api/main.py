from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from PIL import Image


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), "potatoes.keras")
model = tf.keras.models.load_model(model_path)


IMG_SIZE = 250
class_names = ["Early Blight", "Late Blight", "Healthy"]


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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file data and process the image
        image = process_image(await file.read())

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


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port='8000')
