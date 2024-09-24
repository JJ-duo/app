import streamlit as st
import pandas as pd
import os


# 函数：用于管理文件列表，假如有一个目录包含多个图像文件，可以将这些文件名作为列表传递给这个函数，它会返回一个以文件名为索引的 DataFrame。
# DataFrame 可以用于后续的文件管理和显示操作，例如分页显示图像或选择特定的文件进行处理。

# Function to initialize DataFrame
def initialize_df(files):
    df = pd.DataFrame({'file': files})
    df.set_index('file', inplace=True)
    return df

# Select a result's folder
root_dir = "mAP/output"

# Getting type 
saving_type = os.listdir(root_dir)
option_type = st.selectbox("Select a saving type",saving_type,key="saving_type")


# Getting user
user_path = os.path.join(root_dir,option_type)
user = os.listdir(user_path)
option_user = st.selectbox("Select a user", user, key="user")

# Getting final data
final_path = os.path.join(root_dir,option_type, option_user)
final_data = os.listdir(final_path)
option_final = st.selectbox("Select a result to show", final_data, key="final_data")

# Display P&R images

if st.button("Show P&R"):
    
    directory = os.path.join(final_path, option_final)
    # 构建包含所选结果数据的目录路径
    files = ["detection-results-info.png", "ground-truth-info.png", "precision-recall-info.png"]
    # 定义一个列表，包含要显示的三个图像文件的名称
    # Check if all three files exist in the directory
    all_exist = all(os.path.exists(os.path.join(directory, file)) for file in files)
    # 使用 all 函数和列表推导式检查所有三个文件是否都存在于指定目录中
    # os.path.exists 函数用于检查单个文件是否存在

    if all_exist:
        col1, col2, col3 = st.columns(3)  # Use 3 columns
    else:
        col1, col2 = st.columns(2)  # Use 2 columns

    for i, file in enumerate(files):
        # If not all files exist and the current file does not exist, skip it
        if not all_exist and not os.path.exists(os.path.join(directory, file)):
            continue
        if all_exist:
            if i == 0:
                col = col1
            elif i == 1:
                col = col2
            else:
                col = col3
        else:
            col = col1 if i == 0 else col2
        with col:
            st.image(os.path.join(directory, file), caption=file)
                
st.write(":green[Green boxes] means detected results are matched the actual object's boxes")
st.write(":red[Red boxes] means error detected results")
st.markdown(":violet[Pink boxes] means the model missed the results")

# Select images directory
directory = os.path.join(final_path, option_final, 'images') if option_final else "mAP/output/temporary/hunter/exp_test"
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Initialize or load DataFrame
if 'df' not in st.session_state:
    df = initialize_df(files)
    st.session_state.df = df
else:
    df = st.session_state.df 

# Controls
controls = st.columns(3)
# 使用 st.columns 函数创建三个列，用于放置不同的控件
with controls[0]:
    batch_size = st.select_slider("Batch size:", range(10, 110, 10))
    # 在第一个列中创建一个滑动选择器，允许用户选择每批次显示的图像数量
with controls[1]:
    row_size = st.select_slider("Row size:", range(1, 4), value=3)
    # 在第二个列中创建一个滑动选择器，允许用户选择每行显示的图像数量。
num_batches = -(-len(files) // batch_size)  # Ceiling division
# 计算需要多少批次来显示所有图像，使用向上取整的除法。
with controls[2]:
    page = st.selectbox("Page", range(1, num_batches + 1))
    # 在第三个列中创建一个下拉选择框，允许用户选择要查看的批次（即页面）

# Pagination
#batch = files[(page - 1) * batch_size: page * batch_size]
#batch = files[page * batch_size - batch_size:page * batch_size]
# 根据用户选择的页面和每批次的大小，计算当前批次应显示的图像列表
if page is None:
    # 处理 page 为 None 的情况，例如设置默认值或者显示错误信息
    page = 1 
batch = files[(page - 1) * batch_size: page * batch_size]

# Display images in grid layout
st.write('<style>div.row-widget.stHorizontal {flex-wrap: wrap;}</style>', unsafe_allow_html=True)
# 插入自定义 CSS 样式，使得 Streamlit 的行控件可以包裹内容

grid = st.columns(row_size)
# 根据用户选择的每行显示的图像数量，创建相应数量的列
for i, image in enumerate(batch):
    # 遍历当前批次的图像列表
    with grid[i % row_size]:
        #  确定图像应该显示在哪一列
        st.image(os.path.join(directory, image), caption=image)
        # 在每个列中显示图像和其文件名作为标题