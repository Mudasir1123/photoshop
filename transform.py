import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os
import time

# Create input and output directories if they don't exist
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Streamlit app title
st.title("🖼️ Image Processing with Streamlit")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save original image in "input/" folder
    image_path = os.path.join("input", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Select operation
    st.subheader("🎨 Choose an Operation")

    operation = st.radio(
        "Select an operation:",
        ("None", "Blur", "Brighten", "Darken", "Increase Contrast", "Decrease Contrast", "Edge Detection"),
    )

    output_image = image  # Default to original image

    # Generate a unique filename based on timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Format: YYYYMMDD-HHMMSS
    output_filename = f"processed_{timestamp}.png"
    output_path = os.path.join("output", output_filename)

    if operation == "Blur":
        kernel_size = st.slider("🔄 Kernel Size", min_value=1, max_value=10, value=3)
        output_image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

    elif operation == "Brighten":
        brightness_factor = st.slider("☀️ Brightness Factor", min_value=1.0, max_value=3.0, value=1.5)
        enhancer = ImageEnhance.Brightness(image)
        output_image = enhancer.enhance(brightness_factor)

    elif operation == "Darken":
        darken_factor = st.slider("🌙 Darken Factor", min_value=0.1, max_value=1.0, value=0.5)
        enhancer = ImageEnhance.Brightness(image)
        output_image = enhancer.enhance(darken_factor)

    elif operation == "Increase Contrast":
        contrast_factor = st.slider("🎭 Increase Contrast Factor", min_value=1.0, max_value=3.0, value=1.5)
        enhancer = ImageEnhance.Contrast(image)
        output_image = enhancer.enhance(contrast_factor)

    elif operation == "Decrease Contrast":
        decr_contrast_factor = st.slider("🌫 Decrease Contrast Factor", min_value=0.1, max_value=1.0, value=0.5)
        enhancer = ImageEnhance.Contrast(image)
        output_image = enhancer.enhance(decr_contrast_factor)

    elif operation == "Edge Detection":
        st.write("⚡ Processing Edge Detection...")

        img_array = np.array(image.convert("L"))  # Convert to grayscale

        @st.cache_data
        def apply_edge_detection(img_np):
            sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            sobel_y = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

            def apply_kernel(img_np, kernel):
                output_array = np.zeros_like(img_np)
                pad = kernel.shape[0] // 2
                padded_img = np.pad(img_np, pad, mode="edge")

                for i in range(img_np.shape[0]):
                    for j in range(img_np.shape[1]):
                        region = padded_img[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                        output_array[i, j] = np.sum(region * kernel)

                output_array = np.clip(output_array, 0, 255)
                return output_array

            edge_x = apply_kernel(img_np, sobel_x)
            edge_y = apply_kernel(img_np, sobel_y)
            final_edge = np.sqrt(edge_x**2 + edge_y**2)
            final_edge = np.clip(final_edge, 0, 255).astype(np.uint8)

            return Image.fromarray(final_edge)

        output_image = apply_edge_detection(img_array)

    # Save processed image to output folder with unique name
    output_image.save(output_path)

    # Display output image
    st.subheader("🖼 Processed Image")
    st.image(output_image, use_container_width=True)

    # Download button
    with open(output_path, "rb") as file:
        st.download_button(
            label="📥 Download Processed Image",
            data=file,
            file_name=output_filename,
            mime="image/png",
        )

# Footer with your name
st.markdown("---")
st.markdown("💻 **Developed by Muhammad Mudasir** 🎨")
