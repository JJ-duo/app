import streamlit as st
import os

dataset_dir = "mAP/images/"
gt_txt_dir = "mAP/ground-truth"

# Input for the name of the test dataset
option_output_path = st.text_input(
    'Name of test dataset',
    placeholder="DO NOT USE SPACE SYMBOL",
    key="output",
)

# Check if the user has provided a name for the test dataset
if option_output_path:
    uploaded_gt = st.file_uploader("Choose the images", accept_multiple_files=True)
    if uploaded_gt:
        # Check if the directory exists, if not create it
        full_path = os.path.join(dataset_dir, option_output_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        # Save each file with its original name
        for img_file in uploaded_gt:
            file_path = os.path.join(full_path, img_file.name)
            with open(file_path, "wb") as f:
                f.write(img_file.getvalue())

        # Show success message after upload is finished
        st.success("Upload completed successfully.")

    uploaded_txt = st.file_uploader("Upload the GT labels", accept_multiple_files=True)
    if uploaded_txt:
        # Check if the directory exists, if not create it
        full_path = os.path.join(gt_txt_dir, option_output_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        # Save each file with its original name
        for img_file in uploaded_txt:
            file_path = os.path.join(full_path, img_file.name)
            with open(file_path, "wb") as f:
                f.write(img_file.getvalue())

        # Show success message after upload is finished
        st.success("Upload completed successfully.")

    st.write("The GT label format should be as follows:")
    st.write("Class_Name x1 y1 x2 y2 (e.g., stain 100 100 100 100)")
    #st.image("20240223094300.jpg")
else:
    st.warning("Please input the 'Name of test dataset' first, Then press 'Enter'")
