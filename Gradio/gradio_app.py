#! /usr/bin/env python

import gradio as gr

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import torch

from PIL import Image

def model_inference(image):
    model = VisionEncoderDecoderModel.from_pretrained("model/trocr-base-printed_afriberta")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    pixel_values = processor(image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    return output

interface = gr.Interface(fn=model_inference, 
                         inputs=gr.inputs.Image(type='pil'), 
                         outputs='text',
                         title='LowResource TransformerOCR Demo')
interface.launch()