from flask import Flask, request, jsonify
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# import torch
from PIL import Image
import genai
import textwrap
from IPython.display import Markdown

app = Flask(__name__)

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
Gemmodel = genai.GenerativeModel('gemini-pro')

genai.configure(api_key='KEY')  # Replace with your actual API key

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# def predict_step(image_path):
#     images = []
#     i_image = Image.open(image_path)
#     if i_image.mode != "RGB":
#         i_image = i_image.convert(mode="RGB")
#     images.append(i_image)

#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]
#     return preds

def generate_output(image_path, mood_category):
    generated_text = predict_step(image_path)
    prompt = f"write captions for instagram with emoji and hashtag. The photo is described as {generated_text[0]} if my mood is described as {mood_category}"
    response = Gemmodel.generate_content(prompt)
    return to_markdown(response.text)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    data = request.get_json()
    image_path = data['image_path']
    mood_category = data['mood_category']
    result = generate_output(image_path, mood_category)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
