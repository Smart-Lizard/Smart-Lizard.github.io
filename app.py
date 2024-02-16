__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.components.Image(height=192, width=192)
label = gr.components.Label()
examples = ['dog.jpg', 'cat.jpg', 'both.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)