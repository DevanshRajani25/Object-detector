import warnings
warnings.filterwarnings("ignore")
import gradio as gr
from PIL import Image, ImageDraw, ImageFont  # For object detection & Processing
from transformers import pipeline  # Use a pipeline as a high-level helper

model_path = r"C:/Users/Devansh/.cache/huggingface/hub/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"

object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

def draw_bounding_boxes(image, detections, font_path=None, font_size=20):

    # Make a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Default font if path not provided
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']

        # Draw the bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        # Draw the label and score
        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"

        # Draw text with background rectangle for visibility
        if font_path:  # Use the custom font with increased size
            text_size = draw.textbbox((xmin, ymin), text, font=font)
        else:
            # Calculate text size using the default font
            text_size = draw.textbbox((xmin, ymin), text)

        draw.rectangle([(text_size[0], text_size[1]), (text_size[2], text_size[3])], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image

# Function for detecting objects
def detect_object(image):
    raw_image = image
    output = object_detector(raw_image)
    processed_image = draw_bounding_boxes(raw_image, output)
    return processed_image

# Used Gradio for UI
demo = gr.Interface(fn=detect_object,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[gr.Image(label="Processed Image", type="pil")],
                    title="Object Detector",
                    description="Useful for detecting objects from given Input Image.")
demo.launch()