import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [torch.nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                      torch.nn.InstanceNorm2d(in_features),
                      torch.nn.ReLU(inplace=True),
                      torch.nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                      torch.nn.InstanceNorm2d(in_features)]
        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=4):
        super(Generator, self).__init__()
        # Encoder
        model = [torch.nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
                 torch.nn.InstanceNorm2d(64),
                 torch.nn.ReLU(inplace=True),
                 torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                 torch.nn.InstanceNorm2d(128),
                 torch.nn.ReLU(inplace=True),
                 torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                 torch.nn.InstanceNorm2d(256),
                 torch.nn.ReLU(inplace=True)]

        # Transformer (Residual Blocks)
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(256)]

        # Decoder
        model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  torch.nn.InstanceNorm2d(128),
                  torch.nn.ReLU(inplace=True),
                  torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  torch.nn.InstanceNorm2d(64),
                  torch.nn.ReLU(inplace=True),
                  torch.nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
                  torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Utility functions for loading models and processing images
def load_model(model_path, device):
    model = Generator(3, 3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    # Remove "module." prefix if necessary
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def process_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(img_tensor).squeeze().cpu()
    output_image = (output_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # De-normalize
    output_image_pil = Image.fromarray((output_image * 255).astype(np.uint8))  # Convert to PIL Image
    return output_image_pil

# Streamlit App Layout
st.title('CycleGAN Image Converter')
st.write('Upload a photo and choose to convert it to a sketch or vice versa.')

# User selects conversion type
conversion_type = st.radio('Choose conversion type:', ['Photo to Sketch', 'Sketch to Photo'])

# User uploads an image
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Set up device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if uploaded_image is not None:
    # Resize the uploaded image to 128x128 and display
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((128, 128))  # Resize to 128x128
    st.image(image, caption='Uploaded Image (128x128)', width=128)

    # Load the appropriate model based on the selected conversion type
    if conversion_type == 'Photo to Sketch':
        st.write("Converting photo to sketch...")
        model_path = 'photo_to_sketch.pth'
    else:
        st.write("Converting sketch to photo...")
        model_path = 'sketch_to_photo (3).pth'

    # Load model and process the image
    model = load_model(model_path, device)
    output_image = process_image(image, model, device)

    st.image(output_image, caption='Converted Image (128x128)', width=128)
