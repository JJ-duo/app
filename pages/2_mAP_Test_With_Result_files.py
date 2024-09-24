import streamlit as st
import subprocess
import pandas as pd
import os
import shutil

# Specify the target directory
weights_dir = "yolov5/weights/"
dataset_dir = "mAP/images/"
det_result_dir = "mAP/detection-results/"

# Filter only ".py" files
script_data = [file for file in os.listdir("mAP") if file.endswith(".py")]

dataset = os.listdir(dataset_dir)

# user_list = ["Hunter","other","Zhangwenyuan"]
# option_user = st.selectbox(
#    "Select a user",
#    (user_list),
#    index=None,
#    placeholder="",
#    key= "user",
# )

# 用户列表文件路径
user_list_file = "user_list.txt"

# 加载用户列表
def load_user_list(filename):
    try:
        with open(filename, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return []

# 保存用户列表
def save_user_list(user_list, filename):
    with open(filename, "w") as f:
        for user in user_list:
            f.write(user + '\n')

# 初始化用户列表
user_list = load_user_list(user_list_file)

# Streamlit 应用逻辑
option_user = st.selectbox(
    "Select a user or type 'other' to add a new one:",
    user_list + ["other"],
    index=None,
    placeholder="",
    key= "user",
)

if option_user == "other":
    new_user = st.text_input(
        "Enter a new username:"
    )
    if new_user and new_user not in user_list:
        user_list.append(new_user)
        save_user_list(user_list, user_list_file)  # 更新文件
        st.success(f"'{new_user}' has been added to the list.")
    else:
        st.error("Username must be unique and not empty.")
# else:
#     st.write(f"You selected: {option_user}")
# st.write("Current user list:", user_list)

option_dataset = st.selectbox(
   "Select a dataset",
   (dataset),
   index=None,
   placeholder="Or uploaded GT first, then refresh this page to select your dataset",
   key= "dataset",
)

option_output_path = st.text_input(
    'Name of output folder',
    placeholder="DO NOT USE SPACE SYMBOL",
    key= "output",
    )

option = st.selectbox(
    'Keep the results:',
    ('temporary', 'permanent'),
    key= "keep_type",
    )

# option_source = st.selectbox(
#     'The result from:',
#     ('pc', 'chip'),
#     key= "input_source",
#     )

option_pyscript = st.selectbox(
    "Select a python script",
    script_data,
    key="python_script")

uploaded_detection_result = st.file_uploader("Upload results folder :red[(Named the output folder first)]", accept_multiple_files=True)
if uploaded_detection_result:
    # Check if the directory exists, if not create it
    # full_path = os.path.join(det_result_dir, option, option_source, option_user , option_output_path)
    full_path = os.path.join(det_result_dir, option, option_user , option_output_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Save each file with its original name
    for img_file in uploaded_detection_result:
        file_path = os.path.join(full_path, img_file.name)
        with open(file_path, "wb") as f:
            f.write(img_file.getvalue())

if uploaded_detection_result: 
    # Show success message after upload is finished
    st.success("Upload completed successfully.")

st.write("The results' label format should as follow")
st.write("[Class_Name Conf. x1 y1 x2 y2] (eg. stain 0.88 100 100 100 100)")

result = st.button("Start mAP Test")
# st.write(result)

if result:
    #st.text("debug")
    shell_command_mAP = f'''
        python mAP/{option_pyscript} \
        --ground-truth {option_dataset} \
        --detection-results {option}/{option_user}/{option_output_path} \
        --images-folder {option_dataset} \
        --output-path {option}/{option_user}/{option_output_path} \
        
        '''
    # Run the mAP shell command
    try:
        # st.text("debug")
        # Use subprocess.run() to run the command
        result = subprocess.run(shell_command_mAP, shell=True, check=True, text=True)

        # Access the output and return code if needed
        output = result.stdout
        return_code = result.returncode

        
        # # 读取的文件内容显示在前端界面上
        # with open('../output.txt', 'r') as file:
        #     fileoutput = file.read()
        # file.close()
        # st.text("debug")
        # st.text(fileoutput)
        
        
        # 打印输出和返回码
        print(f"Command output:\n{output}")
        print(f"Return code: {return_code}")
        st.write(":sunglasses: mAP finished")
        
    except subprocess.CalledProcessError as e:
        # Handle errors if the command fails
        print(f"Error running the command: {e}")

    # st.write(":sunglasses: mAP finished") 