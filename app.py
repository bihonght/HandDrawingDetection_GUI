import tkinter as tk
from tkinter import Canvas, Button
from PIL import ImageGrab
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def activate_event(event):
    global lastx, lasty
    canvas.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y), width=8, fill='black',
                       capstyle=tk.ROUND, smooth=tk.TRUE)
    lastx, lasty = event.x, event.y

def recognize_digit():
    canvas.update_idletasks()  # Ensure canvas is updated
    time.sleep(0.1)  # Just in case, give it a little time to render
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    # Ensure the coordinates are correct
    filename = "output.png"
    ImageGrab.grab(bbox=(x+5, y+5, x1-5, y1-5)).save(filename)
    # ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('gray', th) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours,_ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("Number of Contours found = " + str(len(contours))) 
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('Contours', image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(x, y, w, h)
        # Create a rectangle around the digit
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)    
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0

        # Predict the digit
        pred = model.predict(img)[0]
        final_pred = np.argmax(pred)
        text = f'{final_pred} {np.max(pred)*100:.2f}%'
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

model = load_model('model.h5')
root = tk.Tk()
root.title('Digit Predictor')
lastx, lasty = None, None

canvas = Canvas(root, width=640, height=480, bg='white')
canvas.grid(row=0, column=0, pady=2, sticky='w', columnspan=2)
canvas.bind('<Button-1>', activate_event)

btn_save = Button(root, text="Predict", command=recognize_digit)
btn_save.grid(row=1, column=0, pady=1)
btn_clear = Button(root, text="Clear Canvas", command=lambda: canvas.delete('all'))
btn_clear.grid(row=1, column=1, pady=1)

root.mainloop()
