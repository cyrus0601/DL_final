import gradio as gr
import random
import time
from transformers import pipeline
import requests
API_URL = "https://api-inference.huggingface.co/models/textattack/distilbert-base-uncased-CoLA"
headers = {"Authorization": "Bearer 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"}

def query1(payload):
    start = time.time()
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json(), time.time()-start

output = query1({
	"inputs": "I like you. I love you",
})

def demo_func_distil_label(input):
    output, duration = query1({
	    "inputs": str(input),
    })
    print(f'output = {output}')
    score = 0
    for item in output[0]:
         if item['score'] > score:
              result = item['label']
              score = item['score']

    return result, duration
    


API_URL_BERT = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
def query2(payload):
    start = time.time()
    response = requests.post(API_URL_BERT, headers=headers, json=payload)
    return response.json(), time.time()-start
	
def demo_func_bert_label(input):
    output, duration = query2({
	    "inputs": str(input),
    })
    print(f'output = {output}')
    score = 0
    for item in output[0]:
         if item['score'] > score:
              result = item['label']
              score = item['score']

    return result, duration
    

model_name = "distilbert-base-uncased"
classifier = pipeline('sentiment-analysis', model=model_name)


with gr.Blocks() as distilbot:
    input_text = gr.Textbox()
    with gr.Row():
        with gr.Column(scale=1):
            distil_output_label = gr.Label()
            distil_time_text = gr.Textbox()
        with gr.Column(scale=1):
            bert_output_label = gr.Label()
            bert_time_text = gr.Textbox()

    input_text.submit(demo_func_distil_label, 
                      inputs = input_text, 
                      outputs = [distil_output_label, distil_time_text])
    print(input_text)
    
    input_text.submit(demo_func_bert_label, 
                      inputs = input_text, 
                      outputs = [bert_output_label, bert_time_text])
    

if __name__ == "__main__":
    distilbot.launch()
