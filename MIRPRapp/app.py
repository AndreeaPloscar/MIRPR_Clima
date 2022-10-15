import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import random as rand


def upload_file():
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    width, height = img.size
    width_new = int(width / 2)
    height_new = int(height / 2)
    img_resized = img.resize((width_new, height_new))
    img = ImageTk.PhotoImage(img_resized)
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.pack()


def algo():
    possible_fronts = ["Cold Front", "Warm Front", "Occluded Front"]
    result = possible_fronts[rand.randint(0, 2)]
    message = tk.Label(root, text=result, width=60)
    message.pack()


root = tk.Tk()

root.geometry("600x400")
root.title('Synoptic Maps')
l1 = tk.Label(root, text='Upload synpotic map')

l1.pack()

b1 = tk.Button(root, text='Upload Photo',
               width=20, command=lambda: upload_file())
b1.pack()
run = tk.Button(root, text='Find fronts',
                width=20, command=lambda: algo())
run.pack()

root.mainloop()
