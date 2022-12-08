import tkinter as tk
import os
import torch
from PIL import ImageTk
from tkinter import filedialog
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from network import SimpleNet


def cnn(image):
    test_transformations = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)])
    image = image.convert('1')
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
    global filepath, panel, name
    f_types = [('Png Files', '*.png')]
    file = filedialog.askopenfile(filetypes=f_types)
    filepath = os.path.abspath(file.name)
    name = file.name
    img = Image.open(file.name)
    width, height = img.size
    width_new = int(width * 2)
    height_new = int(height * 2)
    img_resized = img.resize((width_new, height_new))
    img = ImageTk.PhotoImage(img_resized)
    w.create_image(0, 0, image=img, anchor="nw")
    w.img = img
    w.pack()


def printCNNMessage(image):
    possible_fronts = ["No Front", "Cold Front", "Warm Front", "Mixed Front"]
    result = possible_fronts[cnn(image)]
    message.text = result
    message.config(text=result)


def printErrorMessage():
    error = "Please choose a point closer to the center of the map!"
    message.text = error
    message.config(text=error)

def cropPicture(x,y):
    border = 100
    if border < x < (1000 - border) and border < y < (1000 - border):
        img = Image.open(name)
        formattedImage = img.crop((x - border, y - border, x + border, y + border))
        img = ImageTk.PhotoImage(img)
        w.create_image(0, 0, image=img, anchor="nw")
        w.img = img
        w.pack()
        printCNNMessage(formattedImage)
    else:
        printErrorMessage()

def callback(e):
    x = e.x
    y = e.y
    cropPicture(x, y)

root = tk.Tk()
w = tk.Canvas(root, width=1000, height=1000)
name = ""
w.bind('<Button-1>', callback)
img = Image
filepath = ""
model_name = "./cnn49.model"
root.geometry("600x400")
root.title('Synoptic Maps')
desc = tk.Label(root, text="", width=60)
text = "Please choose a point to be the center of the area you want to analyse"
desc.text = text
desc.config(text = text)
desc.pack()
l1 = tk.Label(root, text='Upload synpotic map')

l1.pack()

b1 = tk.Button(root, text='Upload Photo',
               width=20, command=lambda: upload_file())
b1.pack()
run = tk.Button(root, text='Find fronts',
                width=20, command=lambda: printCNNMessage())
run.pack()
message = tk.Label(root, text="", width=60)
message.pack()
root.mainloop()
