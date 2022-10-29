import tkinter as tk
import os
import torch
from PIL import ImageTk
from tkinter import filedialog
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from network import SimpleNet


def cnn():
    test_transformations = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)])
    image = Image.open(filepath).convert('1')
    image = test_transformations(image)
    image = Variable(image.unsqueeze(0))

    model = SimpleNet()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    output = model(image)
    _, prediction = torch.max(output.data, 1)
    result = prediction[0].item()
    return result


def upload_file():
    global filepath, panel
    f_types = [('Png Files', '*.png')]
    file = filedialog.askopenfile(filetypes=f_types)
    filepath = os.path.abspath(file.name)
    img = Image.open(file.name)
    width, height = img.size
    width_new = int(width * 2)
    height_new = int(height * 2)
    img_resized = img.resize((width_new, height_new))
    img = ImageTk.PhotoImage(img_resized)
    my_image_lbl.image = img
    my_image_lbl.config(image=img)


def printCNNMessage():
    possible_fronts = ["No Front", "Cold Front", "Warm Front", "Mixed Front"]
    result = possible_fronts[cnn()]
    message.text = result
    message.config(text=result)


root = tk.Tk()
my_image_lbl = tk.Label(root)

filepath = ""
model_name = "./cnn_23.model"
root.geometry("600x400")
root.title('Synoptic Maps')
l1 = tk.Label(root, text='Upload synpotic map')

l1.pack()

b1 = tk.Button(root, text='Upload Photo',
               width=20, command=lambda: upload_file())
b1.pack()
run = tk.Button(root, text='Find fronts',
                width=20, command=lambda: printCNNMessage())
run.pack()
my_image_lbl.pack()
message = tk.Label(root, text="", width=60)
message.pack()
root.mainloop()
