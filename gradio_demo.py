#  _*_ coding: utf-8 _*_

# @Date        : 2025/6/13 12:30
# @File        : gradio_demo
# @Author      : TanJingjing
# @Email       : caroline_jing@163.com
# @Description : gradio基本使用

import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
