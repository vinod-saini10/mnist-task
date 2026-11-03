
# ==========================================
# PHASE 3: TKINTER GUI FOR DIGIT RECOGNITION
# ==========================================

import tkinter as tk
import torch
import torch.nn as nn
import numpy as np

from PIL import Image, ImageDraw

# 1. DEFINE SAME CNN ARCHITECTURE

class DigitCNN(nn.Module):

    def __init__(self):
        super(DigitCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 2. LOAD TRAINED MODEL

device = torch.device("cpu")

model = DigitCNN().to(device)

model.load_state_dict(
    torch.load("digit_cnn.pth", map_location=device)
)

model.eval()

print("Trained CNN Model Loaded Successfully!")


# 3. CREATE TKINTER WINDOW

root = tk.Tk()

root.title("Handwritten Digit Recognition")
root.geometry("400x480")

title = tk.Label(
    root,
    text="Draw a Digit (0-9)",
    font=("Arial", 18, "bold")
)

title.pack(pady=10)


# 4. CREATE DRAWING CANVAS

canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    cursor="cross"
)

canvas.pack(pady=10)


# 5. CREATE INVISIBLE PIL IMAGE BUFFER

image = Image.new("L", (280, 280), color=0)

draw = ImageDraw.Draw(image)

last_x = None
last_y = None


# 6. DRAW USING MOUSE

def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y

    # Draw a small dot
    canvas.create_oval(
        event.x - 10,
        event.y - 10,
        event.x + 10,
        event.y + 10,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (event.x - 10, event.y - 10,
         event.x + 10, event.y + 10),
        fill=255
    )


def draw_digit(event):

    global last_x, last_y

    if last_x is not None and last_y is not None:

        # Draw on visible canvas
        canvas.create_line(
            last_x, last_y,
            event.x, event.y,
            fill="white",
            width=20,
            capstyle=tk.ROUND,
            smooth=True
        )

        # Draw on invisible PIL image
        draw.line(
            (last_x, last_y, event.x, event.y),
            fill=255,
            width=20
        )

    last_x = event.x
    last_y = event.y


def stop_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_draw)


# 7. PREDICT DIGIT

def predict_digit():

    # Get drawing boundaries
    bbox = image.getbbox()

    if bbox is None:

        result_label.config(
            text="Please draw a digit first!"
        )

        return

    # Crop the drawn digit
    cropped = image.crop(bbox)

    # Resize while keeping the digit's proportions
    width, height = cropped.size

    ratio = min(20 / width, 20 / height)

    new_width = max(1, int(width * ratio))
    new_height = max(1, int(height * ratio))

    resized = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create a black 28x28 image
    final_image = Image.new("L", (28, 28), color=0)

    # Place digit in the center
    x_position = (28 - new_width) // 2
    y_position = (28 - new_height) // 2

    final_image.paste(
        resized,
        (x_position, y_position)
    )

    # Convert image into NumPy array
    img_array = np.array(
        final_image,
        dtype=np.float32
    ) / 255.0

    # Apply same normalization as training
    img_array = (img_array - 0.1307) / 0.3081

    # Convert into PyTorch tensor
    img_tensor = torch.tensor(
        img_array,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    # 8. PREDICT USING CNN

    with torch.no_grad():

        output = model(img_tensor.to(device))

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities, dim=1
        )

    digit = prediction.item()
    confidence = confidence.item() * 100

    # 9. DISPLAY RESULT

    result_label.config(
        text=f"Predicted Digit: {digit}\n"
             f"Confidence: {confidence:.2f}%"
    )


# 10. CLEAR CANVAS

def clear_canvas():

    global image, draw, last_x, last_y

    canvas.delete("all")

    image = Image.new("L", (280, 280), color=0)

    draw = ImageDraw.Draw(image)

    last_x = None
    last_y = None

    result_label.config(text="Draw a digit and click Predict")


# 11. CREATE BUTTONS

button_frame = tk.Frame(root)

button_frame.pack(pady=10)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    bg="green",
    fg="white",
    font=("Arial", 12),
    width=10
)

predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    bg="red",
    fg="white",
    font=("Arial", 12),
    width=10
)

clear_button.grid(row=0, column=1, padx=10)


# 12. DISPLAY PREDICTION RESULT

result_label = tk.Label(
    root,
    text="Draw a digit and click Predict",
    font=("Arial", 14, "bold")
)

result_label.pack(pady=10)


# 13. START APPLICATION

root.mainloop()
