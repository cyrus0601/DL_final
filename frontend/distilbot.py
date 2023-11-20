import gradio as gr
import random
import time

def demo_func_distil_label(input):
    start_distil = time.time()
    if input == 'good':
        return 'POSITIVE', time.time()-start_distil
    elif input == 'bad':
        return 'NEGATIVE', time.time()-start_distil
    else:
        return random.choice(['POSITIVE', 'NEGATIVE']), time.time()-start_distil
    
def demo_func_bert_label(input):
    start_bert = time.time()
    if input == 'good':
        return 'POSITIVE', time.time()-start_bert
    elif input == 'bad':
        return 'NEGATIVE', time.time()-start_bert
    else:
        return random.choice(['POSITIVE', 'NEGATIVE']), time.time()-start_bert

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
    
    input_text.submit(demo_func_bert_label, 
                      inputs = input_text, 
                      outputs = [bert_output_label, bert_time_text])
    

if __name__ == "__main__":
    distilbot.launch()