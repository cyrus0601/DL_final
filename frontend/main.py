import gradio as gr
from distilbot import distilbot

demo = gr.TabbedInterface([distilbot], ["distilbot"],
                          title = "An application of distilbert - sentiment analysis")

if __name__=="__main__":
    demo.launch(server_name = "0.0.0.0", server_port = 8086)