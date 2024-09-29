import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import Label, Button
from PIL import Image, ImageTk
from diffusers import StableDiffusionUpscalePipeline
import torch
import threading
 
 
# Initialize the main window
root = tk.Tk()
root.title("Image Upscaler - The Pycodes")
root.geometry("600x400")
 
 
# Create a scrollable frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)
 
 
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
 
 
scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
 
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
 
 
# Create a frame inside the canvas for the content
content_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=content_frame, anchor="nw")
 
 
# Global variables to store the selected and upscaled images
selected_image_path = None
upscaled_img = None
 
 
# Set default device to CPU
device = "cpu"
 
 
# Load the Stable Diffusion Upscaler model for both CPU and GPU
pipe_cpu = StableDiffusionUpscalePipeline.from_pretrained(
   "stabilityai/stable-diffusion-x4-upscaler",
   torch_dtype=torch.float32  # Use float32 for CPU
).to("cpu")
 
 
pipe_gpu = StableDiffusionUpscalePipeline.from_pretrained(
   "stabilityai/stable-diffusion-x4-upscaler",
   torch_dtype=torch.float16  # Use float16 for GPU
).to("cuda") if torch.cuda.is_available() else pipe_cpu
 
 
pipe_cpu.enable_attention_slicing(None)  # Optimize for CPU performance
 
 
# Function to select device (CPU or GPU)
def select_device(device_choice):
   global device
   device = device_choice
 
 
# Function to open a file dialog and select an image
def select_image():
   global selected_image_path, img_label, upscaled_img_label
 
 
   file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
 
 
   if file_path:
       selected_image_path = file_path
       original_img = Image.open(selected_image_path)
       original_img = original_img.resize((200, 200))  # Resize the image for display
 
 
       imgtk = ImageTk.PhotoImage(original_img)
       img_label.config(image=imgtk)
       img_label.image = imgtk
 
 
 
 
# Function to upscale the selected image using Stable Diffusion Upscaler
def upscale_image_with_diffusion(image_path):
   global pipe_cpu, pipe_gpu, device
 
 
   # Step 1: Open the original image
   image = Image.open(image_path)
 
 
   # Check if CPU or GPU is selected
   if device == "cpu":
       # Resize image for CPU to 128x128 for faster performance
       image = image.resize((128, 128))
       upscaled_image = pipe_cpu(prompt="high-quality photo", image=image, num_inference_steps=25).images[0]
   else:
       # For GPU, use the full-resolution image (512x512 or higher)
       image = image.resize((512, 512))
       upscaled_image = pipe_gpu(prompt="very high-quality, highly detailed, accurate photo",
                                 image=image, num_inference_steps=50).images[0]
 
 
   return upscaled_image
 
 
 
 
# Function to display the upscaled image (runs in the main thread)
def display_upscaled_image():
   global selected_image_path, upscaled_img
 
 
   if selected_image_path:
       # Disable buttons while upscaling
       upscale_button.config(state=tk.DISABLED)
       save_button.config(state=tk.DISABLED)
 
 
       # Run the upscaling process in a separate thread
       threading.Thread(target=upscale_in_background).start()
 
 
# Function to upscale the image in a separate thread
def upscale_in_background():
   global selected_image_path, upscaled_img
 
 
   try:
       # Use the Stable Diffusion model to upscale the image
       upscaled_img = upscale_image_with_diffusion(selected_image_path)
       upscaled_img = upscaled_img.resize((200, 200))  # Resize the upscaled image for display
 
 
       # Update the UI with the upscaled image (back to the main thread)
       imgtk = ImageTk.PhotoImage(upscaled_img)
       upscaled_img_label.config(image=imgtk)
       upscaled_img_label.image = imgtk
   finally:
       # Re-enable buttons after upscaling is done
       upscale_button.config(state=tk.NORMAL)
       save_button.config(state=tk.NORMAL)
 
 
 
 
# Function to save the upscaled image
def save_image():
   global upscaled_img
 
 
   if upscaled_img:
       save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                           ("All files", "*.*")])
       if save_path:
           upscaled_img.save(save_path)
           messagebox.showinfo("Image Saved", f"Image has been saved successfully at {save_path}")
   else:
       messagebox.showerror("Error", "No upscaled image to save!")
 
 
 
 
# Create the layout: Select Image button, Upscale Image button, Save Image button, and labels for images
select_button = Button(content_frame, text="Select Image", command=select_image)
select_button.pack(pady=10)
 
 
# Dropdown to select CPU or GPU
device_choice_label = Label(content_frame, text="Choose Device:")
device_choice_label.pack(pady=5)
 
 
device_choice = ttk.Combobox(content_frame, values=["cpu", "gpu"], state="readonly")
device_choice.set("cpu")  # Set default value to CPU
device_choice.pack(pady=5)
device_choice.bind("<<ComboboxSelected>>", lambda event: select_device(device_choice.get()))
 
 
img_label = Label(content_frame)  # To display the selected image
img_label.pack(pady=10)
 
 
upscale_button = Button(content_frame, text="Upscale Image", command=display_upscaled_image)
upscale_button.pack(pady=10)
 
 
upscaled_img_label = Label(content_frame)  # To display the upscaled image
upscaled_img_label.pack(pady=10)
 
 
save_button = Button(content_frame, text="Save Image", command=save_image)  # Button to save the upscaled image
save_button.pack(pady=10)
 
 
# Start the Tkinter event loop
root.mainloop()
