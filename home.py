import streamlit as st

st.set_page_config(
    page_title="mAP Tools",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
       
    }
)


tab1, tab2 = st.tabs(["数据查看SOP", "平台操作SOP"])

with tab1:
   st.header("数据查看SOP")
   st.write("1. 详细的模型测试结果查看。点击左侧边栏的 :rainbow[mAP Detail Board] 选择不同的模型结果开始查看")
   st.write("2. 模型结果对比。点击左侧边栏的 :rainbow[mAP Compare] 可选择至多3个模型结果进行比较")
with tab2:
   st.header("平台操作SOP")
   st.write("1. 上传测试集。点击左侧边栏的 :rainbow[Upload mAP Test Dataset] 注意上传数据的标签格式")
   st.write("2. PC端模型测试。点击左侧边栏的 :rainbow[mAP Test With PC Results] 注意里面的各项选择")
   st.write("3. Chip端模型测试。点击左侧边栏的 :rainbow[mAP Test With Chip Results] 注意先上传板端推理结果（TXT文件）")
  
   


