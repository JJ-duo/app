import streamlit as st
import os

# Function to display images given a directory and a list of file names
def display_images(directory, files):
    col1, col2, col3 = st.columns(3)  # Create three columns layout
    for i, file in enumerate(files):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
            
        image_path = os.path.join(directory, file)
        try:
            col.image(image_path, caption=file)
        except FileNotFoundError:
            st.warning(f"Image '{file}' not found.")


def get_subfolders_at_depth(directory, target_depth):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        depth = root.count(os.sep) - directory.count(os.sep)
        if depth == target_depth:
            subfolders.extend([os.path.join(root, d) for d in dirs])
    return subfolders


# Define the directory where the datasets are stored
dataset_dir = "mAP/output/"
subfolders_at_depth = get_subfolders_at_depth(dataset_dir, 1)

# Get the list of folders (results)
results = subfolders_at_depth

############################
# Allow the user to select three result folders
option_result_1 = st.selectbox("Select the first result folder", ["Select a folder"] + results)

# Display images for the first result folder
if option_result_1 != "Select a folder":
    directory_1 = os.path.join(option_result_1)
    files_1 = ["detection-results-info.png", "ground-truth-info.png", "precision-recall-info.png"]
    display_images(directory_1, files_1)

#############################
option_result_2 = st.selectbox("Select the second result folder", ["Select a folder"] + results)

# Display images for the second result folder
if option_result_2 != "Select a folder":
    directory_2 = os.path.join(option_result_2)
    files_2 = ["detection-results-info.png", "ground-truth-info.png", "precision-recall-info.png"]
    display_images(directory_2, files_2)

#############################
option_result_3 = st.selectbox("Select the third result folder", ["Select a folder"] + results)

# Display images for the third result folder
if option_result_3 != "Select a folder":
    directory_3 = os.path.join(option_result_3)
    files_3 = ["detection-results-info.png", "ground-truth-info.png", "precision-recall-info.png"]
    display_images(directory_3, files_3)

