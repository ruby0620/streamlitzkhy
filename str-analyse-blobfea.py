import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import copy
try:
    from PIL import Image
except ImportError:
    Image = None

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="缺陷数据分析", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏导航索引
st.sidebar.title("📑 功能导航")
st.sidebar.markdown("---")

# 主要功能模块
st.sidebar.subheader("🔍 主要功能")
st.sidebar.markdown("""
- [📁 过漏检分析](#过漏检分析)
- [🖼️ 图像查看](#图像查看)
- [✂️ 区域过滤](#区域过滤)
""")

st.sidebar.markdown("---")

# 过漏检分析子功能
st.sidebar.subheader("📁 过漏检分析功能")
st.sidebar.markdown("""
**文件夹对比：**
- [🌐 多文件夹晶圆图](#多文件夹晶圆图)
- [🔗 缺陷坐标匹配](#缺陷坐标匹配分析)

**KLA匹配分析：**
- [📊 过漏检统计](#过漏检统计)
- [📏 DSIZE尺寸分析](#DSIZE尺寸分析)
- [📐 过检尺寸分布](#过检尺寸分布)
- [🔢 按尺寸区间统计](#按尺寸区间统计)
- [📊 MaxOrg比值分析](#MaxOrg比值分析)
- [🔍 MaxOrg65532统计](#MaxOrg65532统计)
- [📈 DW1O通道比值](#DW1O通道比值)
- [💡 BGMean值分布](#BGMean值分布)
- [📋 BGMean汇总表](#BGMean汇总表)
- [� TotalSNR尺寸分布](#TotalSNR尺寸分布)
- [�🔄 共有率分析](#共有率分析)
""")

st.sidebar.markdown("---")

# 系统信息
st.sidebar.subheader("ℹ️ 系统信息")
st.sidebar.info("""
**缺陷数据分析系统 v3.0**

支持功能：
- 过漏检综合分析
- TIFF图像查看
- 区域过滤
""")

# 主标题
st.title("缺陷数据分析系统")

# 创建标签页
tab1, tab2, tab3, tab4 = st.tabs(["📁 过漏检分析", "🖼️ 图像查看", "✂️ 区域过滤", "⚙️ 规则编辑器"])

with tab2:
    st.markdown('<a name="图像查看"></a>', unsafe_allow_html=True)
    st.header("🖼️ TIFF图像查看器")
    
    # 文件夹选择
    folder_path = st.text_input("请输入包含TIFF图像的文件夹路径", 
                               placeholder="例如: D:/images/tiff_folder")
    
    if folder_path:
        try:
            if Image is None:
                st.error("请安装PIL库: pip install Pillow")
                st.stop()
            
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                st.error("文件夹路径不存在，请检查路径是否正确")
            else:
                # 搜索TIFF文件
                tiff_patterns = [
                    os.path.join(folder_path, "*-DN1O.tiff"),
                    os.path.join(folder_path, "*-DN1O.TIFF"),
                    os.path.join(folder_path, "*-DW1O.tiff"), 
                    os.path.join(folder_path, "*-DW1O.TIFF"),
                    os.path.join(folder_path, "*-DW2O.tiff"),
                    os.path.join(folder_path, "*-DW2O.TIFF")
                ]
                
                all_files = []
                for pattern in tiff_patterns:
                    all_files.extend(glob.glob(pattern))
                
                if not all_files:
                    st.warning("在指定文件夹中未找到符合格式的TIFF文件")
                    st.info("文件格式应为: ID-DN1O.tiff, ID-DW1O.tiff, ID-DW2O.tiff")
                else:
                    st.success(f"找到 {len(all_files)} 个TIFF文件")
                    
                    # 解析文件并按ID分组
                    @st.cache_data
                    def parse_tiff_files(file_list):
                        file_groups = {}
                        
                        for file_path in file_list:
                            filename = os.path.basename(file_path)
                            
                            # 解析文件名格式：ID-通道.tiff
                            if '-DN1O.' in filename.upper():
                                channel = 'DN1O'
                                file_id = filename.split('-DN1O.')[0]
                            elif '-DW1O.' in filename.upper():
                                channel = 'DW1O'
                                file_id = filename.split('-DW1O.')[0]
                            elif '-DW2O.' in filename.upper():
                                channel = 'DW2O'
                                file_id = filename.split('-DW2O.')[0]
                            else:
                                continue
                            
                            if file_id not in file_groups:
                                file_groups[file_id] = {}
                            
                            file_groups[file_id][channel] = file_path
                        
                        return file_groups
                    
                    file_groups = parse_tiff_files(all_files)
                    
                    # 显示文件统计
                    st.subheader("文件统计")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("总ID数量", len(file_groups))
                    with col2:
                        dn1o_count = sum(1 for group in file_groups.values() if 'DN1O' in group)
                        st.metric("DN1O图像", dn1o_count)
                    with col3:
                        dw1o_count = sum(1 for group in file_groups.values() if 'DW1O' in group)
                        st.metric("DW1O图像", dw1o_count)
                    with col4:
                        dw2o_count = sum(1 for group in file_groups.values() if 'DW2O' in group)
                        st.metric("DW2O图像", dw2o_count)
                    
                    # 按ID排序（数字排序）
                    sorted_ids = sorted(file_groups.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
                    
                    # 初始化session state
                    if 'current_id_index' not in st.session_state:
                        st.session_state.current_id_index = 0
                    if 'folder_path' not in st.session_state:
                        st.session_state.folder_path = folder_path
                    
                    # 如果文件夹路径改变，重置索引
                    if st.session_state.folder_path != folder_path:
                        st.session_state.current_id_index = 0
                        st.session_state.folder_path = folder_path
                    
                    # 确保索引在有效范围内
                    if st.session_state.current_id_index >= len(sorted_ids):
                        st.session_state.current_id_index = len(sorted_ids) - 1 if sorted_ids else 0
                    elif st.session_state.current_id_index < 0:
                        st.session_state.current_id_index = 0
                    
                    # ID选择和导航
                    st.subheader("批量浏览控制")
                    
                    # 导航按钮 - 使用表单来确保按钮点击被正确处理
                    with st.form("navigation_form", clear_on_submit=False):
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
                        
                        with col1:
                            first_btn = st.form_submit_button("⏮️ 第一个")
                        
                        with col2:
                            prev_disabled = st.session_state.current_id_index <= 0
                            prev_btn = st.form_submit_button("⬅️ 上一个", disabled=prev_disabled)
                        
                        with col3:
                            next_disabled = st.session_state.current_id_index >= len(sorted_ids) - 1
                            next_btn = st.form_submit_button("➡️ 下一个", disabled=next_disabled)
                        
                        with col4:
                            last_btn = st.form_submit_button("⏭️ 最后一个")
                        
                        with col5:
                            # 进度条和当前状态
                            if sorted_ids:
                                progress = (st.session_state.current_id_index + 1) / len(sorted_ids)
                                st.progress(progress)
                                current_id = sorted_ids[st.session_state.current_id_index]
                                st.write(f"**{st.session_state.current_id_index + 1} / {len(sorted_ids)}** (ID: {current_id})")
                    
                    # 处理按钮点击
                    if first_btn:
                        st.session_state.current_id_index = 0
                        st.rerun()
                    elif prev_btn and not prev_disabled:
                        st.session_state.current_id_index = max(0, st.session_state.current_id_index - 1)
                        st.rerun()
                    elif next_btn and not next_disabled:
                        st.session_state.current_id_index = min(len(sorted_ids) - 1, st.session_state.current_id_index + 1)
                        st.rerun()
                    elif last_btn:
                        st.session_state.current_id_index = len(sorted_ids) - 1
                        st.rerun()
                    
                    
                    # ID选择下拉框
                    # st.subheader("选择特定ID")
                    # col1, col2 = st.columns([3, 1])
                    # with col1:
                    #     # 创建ID选择框
                    #     selected_index = st.selectbox(
                    #         "跳转到特定ID", 
                    #         range(len(sorted_ids)),
                    #         index=st.session_state.current_id_index,
                    #         format_func=lambda x: f"ID: {sorted_ids[x]}",
                    #         key="id_selectbox",
                    #         help="直接选择要查看的ID"
                    #     )
                        
                        # 如果选择框的值改变了，更新session state
                        # if selected_index != st.session_state.current_id_index:
                        #     st.session_state.current_id_index = selected_index
                        #     st.rerun()
                    
                    with col2:
                        auto_enhance = st.checkbox("自动增强显示", value=True, help="自动调整16位图像的显示对比度")
                    
                    # 当前选择的ID
                    selected_id = sorted_ids[st.session_state.current_id_index] if sorted_ids else None
                    
                    # 高级显示选项
                    with st.expander("高级显示选项"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            percentile_low = st.slider("对比度下限百分位", 0.0, 5.0, 1.0, 0.1, help="用于对比度拉伸的下限百分位")
                        with col2:
                            percentile_high = st.slider("对比度上限百分位", 95.0, 100.0, 100.0, 0.1, help="用于对比度拉伸的上限百分位")
                        with col3:
                            gamma_value = st.slider("伽马校正", 0.1, 3.0, 1.1, 0.1, help="调整图像亮度和对比度")
                    
                    if selected_id and selected_id in file_groups:
                        st.subheader(f"ID: {selected_id} 的图像")
                        
                        # 获取该ID的所有通道图像
                        channels = ['DN1O', 'DW1O', 'DW2O']
                        available_channels = [ch for ch in channels if ch in file_groups[selected_id]]
                        
                        if available_channels:
                            # 显示图像信息
                            with st.expander("图像信息"):
                                for channel in available_channels:
                                    file_path = file_groups[selected_id][channel]
                                    st.write(f"**{channel}**: {os.path.basename(file_path)}")
                            
                            # 读取和显示图像
                            def load_and_process_16bit_tiff(file_path, enhance=True, p_low=1.0, p_high=99.0, gamma=1.0):
                                """加载16位TIFF图像并处理为可显示的格式（类似ImageJ）"""
                                try:
                                    # 使用PIL读取16位TIFF
                                    img = Image.open(file_path)
                                    
                                    # 转换为numpy数组
                                    img_array = np.array(img, dtype=np.float64)  # 使用float64保持精度
                                    
                                    if enhance:
                                        # 使用用户指定的百分位数进行对比度调整
                                        p_low_val, p_high_val = np.percentile(img_array, (p_low, p_high))
                                        
                                        # 如果上下限相同，使用min和max
                                        if p_high_val - p_low_val == 0:
                                            p_low_val, p_high_val = img_array.min(), img_array.max()
                                        
                                        if p_high_val - p_low_val > 0:
                                            # 对比度拉伸
                                            img_normalized = (img_array - p_low_val) / (p_high_val - p_low_val)
                                            img_normalized = np.clip(img_normalized, 0, 1)
                                        else:
                                            img_normalized = img_array / img_array.max() if img_array.max() > 0 else img_array
                                        
                                        # 应用用户指定的伽马校正
                                        if gamma != 1.0:
                                            img_normalized = np.power(img_normalized, 1.0/gamma)  # 注意伽马的倒数
                                        
                                        # 转换为8位
                                        img_enhanced = (img_normalized * 255).astype(np.uint8)
                                    else:
                                        # 简单的线性缩放，保持16位到8位的线性关系
                                        max_val = img_array.max()
                                        if max_val > 0:
                                            if max_val <= 255:
                                                img_enhanced = img_array.astype(np.uint8)
                                            else:
                                                # 16位到8位的线性映射
                                                img_enhanced = (img_array / 65535.0 * 255).astype(np.uint8)
                                        else:
                                            img_enhanced = img_array.astype(np.uint8)
                                    
                                    # 返回处理后的图像、原始数组（用于统计）、形状和统计信息
                                    return img_enhanced, img_array.shape, (img_array.min(), img_array.max(), img_array.mean()), p_low_val if enhance else None, p_high_val if enhance else None
                                
                                except Exception as e:
                                    st.error(f"读取图像失败: {str(e)}")
                                    return None, None, None, None, None
                            
                            # 创建列布局显示图像
                            if len(available_channels) == 1:
                                # 单张图像
                                channel = available_channels[0]
                                file_path = file_groups[selected_id][channel]
                                
                                result = load_and_process_16bit_tiff(file_path, auto_enhance, percentile_low, percentile_high, gamma_value)
                                img_processed, img_shape, img_stats, p_low_val, p_high_val = result
                                
                                if img_processed is not None:
                                    st.write(f"**{channel} 通道**")
                                    caption = f"{channel} - 形状: {img_shape}"
                                    if auto_enhance and p_low_val is not None:
                                        caption += f", 显示范围: {p_low_val:.0f}-{p_high_val:.0f}"
                                    caption += f", 原始范围: {img_stats[0]:.0f}-{img_stats[1]:.0f}, 平均值: {img_stats[2]:.1f}"
                                    st.image(img_processed, caption=caption)
                            
                            elif len(available_channels) == 2:
                                # 两张图像
                                col1, col2 = st.columns(2)
                                
                                for i, channel in enumerate(available_channels):
                                    file_path = file_groups[selected_id][channel]
                                    result = load_and_process_16bit_tiff(file_path, auto_enhance, percentile_low, percentile_high, gamma_value)
                                    img_processed, img_shape, img_stats, p_low_val, p_high_val = result
                                    
                                    if img_processed is not None:
                                        with [col1, col2][i]:
                                            st.write(f"**{channel} 通道**")
                                            st.image(img_processed, caption=f"{channel}")
                                            with st.expander(f"{channel} 详细信息"):
                                                st.write(f"图像尺寸: {img_shape}")
                                                st.write(f"原始值范围: {img_stats[0]:.0f} - {img_stats[1]:.0f}")
                                                st.write(f"平均值: {img_stats[2]:.1f}")
                                                if auto_enhance and p_low_val is not None:
                                                    st.write(f"显示范围: {p_low_val:.0f} - {p_high_val:.0f}")
                                                    st.write(f"对比度拉伸: {percentile_low}% - {percentile_high}%")
                            
                            else:
                                # 三张图像（标准情况）
                                col1, col2, col3 = st.columns(3)
                                columns = [col1, col2, col3]
                                
                                for i, channel in enumerate(available_channels):
                                    file_path = file_groups[selected_id][channel]
                                    result = load_and_process_16bit_tiff(file_path, auto_enhance, percentile_low, percentile_high, gamma_value)
                                    img_processed, img_shape, img_stats, p_low_val, p_high_val = result
                                    
                                    if img_processed is not None:
                                        with columns[i]:
                                            st.write(f"**{channel} 通道**")
                                            st.image(img_processed, caption=f"{channel}")
                                            with st.expander(f"{channel} 详细信息"):
                                                st.write(f"图像尺寸: {img_shape}")
                                                st.write(f"原始值范围: {img_stats[0]:.0f} - {img_stats[1]:.0f}")
                                                st.write(f"平均值: {img_stats[2]:.1f}")
                                                if auto_enhance and p_low_val is not None:
                                                    st.write(f"显示范围: {p_low_val:.0f} - {p_high_val:.0f}")
                                                    st.write(f"伽马校正: {gamma_value}")
                                                    st.write(f"对比度百分位: {percentile_low}% - {percentile_high}%")
                            

                            
                            # 使用提示
                            st.caption("💡 提示：使用上方的导航按钮或快速跳转功能切换图像")
                        
                        else:
                            st.warning(f"ID {selected_id} 没有找到任何通道的图像")
                    
                    # 显示所有ID的概览
                    with st.expander("所有ID概览"):
                        st.write("可用的ID和对应的通道:")
                        
                        overview_data = []
                        for file_id in sorted_ids:
                            channels_available = list(file_groups[file_id].keys())
                            overview_data.append({
                                'ID': file_id,
                                'DN1O': '✓' if 'DN1O' in channels_available else '✗',
                                'DW1O': '✓' if 'DW1O' in channels_available else '✗',
                                'DW2O': '✓' if 'DW2O' in channels_available else '✗',
                                '文件数': len(channels_available)
                            })
                        
                        overview_df = pd.DataFrame(overview_data)
                        st.dataframe(overview_df, use_container_width=True)

    # 快速跳转功能
                    st.subheader("快速跳转")
                    # 先获取跳转步长
                    jump_step = st.number_input("跳转步长", min_value=1, max_value=10, value=5, key="jump_step_input")
                    
                    # 使用按钮进行跳转（不用表单，直接用按钮）
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"当前显示: ID {sorted_ids[st.session_state.current_id_index] if sorted_ids else 'None'}")
                    
                    with col2:
                        if st.button(f"⏪ 后退 {jump_step} 个", key="jump_back_btn"):
                            new_index = max(0, st.session_state.current_id_index - jump_step)
                            st.session_state.current_id_index = new_index
                            st.rerun()
                    
                    with col3:
                        if st.button(f"⏩ 前进 {jump_step} 个", key="jump_forward_btn"):
                            new_index = min(len(sorted_ids) - 1, st.session_state.current_id_index + jump_step)
                            st.session_state.current_id_index = new_index
                            st.rerun()
                    
        
        except ImportError as e:
            st.error("缺少必要的库，请安装: pip install Pillow")
        except Exception as e:
            st.error(f"处理文件夹时出错: {str(e)}")
    
    else:
        st.info("请输入文件夹路径开始查看TIFF图像")
        st.markdown("""
        ### TIFF图像查看器使用说明：
        
        1. **文件格式要求**：
           - 文件命名格式：`ID-通道.tiff` 或 `ID-通道.TIFF`
           - 支持的通道：DN1O, DW1O, DW2O
           - 例如：`1-DN1O.tiff`, `1-DW1O.tiff`, `1-DW2O.tiff`
        
        2. **功能特点**：
           - 自动检测文件夹中所有符合格式的TIFF文件
           - 按ID分组显示，一行显示三个通道
           - 支持16位TIFF图像的正确显示
           - 自动增强对比度，优化显示效果
           - 批量浏览功能，快速切换不同ID
        
        3. **16位图像处理**：
           - 自动进行对比度拉伸（2%-98%分位数）
           - 显示原始数值范围和统计信息
           - 可选择开启/关闭自动增强
        
        4. **浏览功能**：
           - ID选择下拉框
           - 上一个/下一个ID快速切换
           - 所有ID概览表格
           - 详细的图像信息显示
        """)

# 第五个标签页 - 多文件夹缺陷对比
with tab1:
    st.markdown('<a name="多工况对比"></a>', unsafe_allow_html=True)
    st.header("📁 过漏检分析")
    
    st.markdown("""
    ### 功能说明
    选择一个包含多个子文件夹的主文件夹，程序会读取每个子文件夹中的缺陷数据文件。
    
    **文件要求：**
    - 主文件夹包含多个子文件夹（如：P1, P2, P3...）
    - 普通子文件夹：读取 `BlobFeatures.csv` 文件（第4、5列为X、Y坐标）
    - 包含"kla"的子文件夹：读取 `jianchu.csv` 文件（XREL、YREL列为坐标）
    - 可选择是否过滤nDefectType为1000和10001的数据（仅对BlobFeatures.csv有效）
    
    **显示方式：**
    - 以坐标(150000, 150000)为中心绘制晶圆图
    - 不同子文件夹的数据用不同颜色显示
    - 支持交互式查看和统计分析
    """)
    
    # 文件夹选择
    st.subheader("选择主文件夹")
    
    # 输入文件夹路径
    folder_path = st.text_input("输入主文件夹路径", placeholder=r"例如: D:\data\wafer_folders")
    
    # 或者使用文件上传作为备选方案
    st.write("或者")
    use_upload = st.checkbox("通过上传CSV文件（如果无法选择文件夹）")
    
    if use_upload:
        uploaded_files = st.file_uploader(
            "上传多个BlobFeatures.csv文件",
            type=['csv'],
            accept_multiple_files=True,
            key="multi_folder_uploader"
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            try:
                # 绘图参数
                st.subheader("绘图参数")
                col1, col2 = st.columns(2)
                with col1:
                    center_x = st.number_input("中心X坐标", value=150000.0, key="mf_center_x")
                    center_y = st.number_input("中心Y坐标", value=150000.0, key="mf_center_y")
                with col2:
                    plot_range = st.number_input("绘图范围（半径）", value=150000.0, min_value=1000.0, key="mf_range")
                    point_size = st.slider("点的大小", min_value=3, max_value=15, value=6, key="mf_size")
                
                # 网格显示选项
                show_grid_upload = st.checkbox("显示背景网格", value=True, key="show_grid_upload", help="控制图表中是否显示背景网格线")
                
                # 数据过滤选项
                st.subheader("数据过滤选项")
                filter_special_types_upload = st.checkbox(
                    "过滤特殊类型缺陷（nDefectType=1000/10001）", 
                    value=True,
                    key="filter_special_types_upload",
                    help="仅对CASI数据有效"
                )
                
                if st.button("生成对比图", type="primary", key="mf_plot_btn"):
                    # 准备颜色列表
                    colors = [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AED6F1',
                        '#A9DFBF', '#F9E79F', '#D7BDE2', '#A2D9CE', '#FAD7A0'
                    ]
                    
                    # 创建图形
                    fig = go.Figure()
                    
                    # 用于存储所有数据
                    all_data = []
                    
                    # 读取每个文件
                    for idx, uploaded_file in enumerate(uploaded_files):
                        # 使用文件名作为标签，如果文件名包含文件夹信息
                        folder_name = f"P{idx + 1}"
                        if hasattr(uploaded_file, 'name'):
                            # 尝试从文件名提取文件夹信息
                            file_name = uploaded_file.name
                            folder_name = file_name.replace('BlobFeatures.csv', '').replace('.csv', '').strip('_-') or f"P{idx + 1}"
                        
                        # 读取CSV
                        df = pd.read_csv(uploaded_file)
                        
                        # 根据用户选择过滤nDefectType为1000和10001的数据
                        if filter_special_types_upload and 'nDefectType' in df.columns:
                            df = df[(df['nDefectType'] != 1000) & (df['nDefectType'] != 10001)]
                        
                        # 获取第4列和第5列（索引3和4）
                        if df.shape[1] >= 5:
                            x_col = df.columns[3]
                            y_col = df.columns[4]
                            
                            x_data = pd.to_numeric(df[x_col], errors='coerce')
                            y_data = pd.to_numeric(df[y_col], errors='coerce')
                            
                            # 获取DW1O_BGMean列（如果存在）
                            dw1o_bgmean = None
                            if 'DW1O_BGMean' in df.columns:
                                dw1o_bgmean = pd.to_numeric(df['DW1O_BGMean'], errors='coerce')
                            
                            # 过滤有效数据
                            valid_mask = pd.notna(x_data) & pd.notna(y_data)
                            x_valid = x_data[valid_mask]
                            y_valid = y_data[valid_mask]
                            
                            if len(x_valid) > 0:
                                # 计算DW1O_BGMean统计值（排除0值）
                                bgmean_stats = {}
                                if dw1o_bgmean is not None:
                                    dw1o_valid = dw1o_bgmean[valid_mask].dropna()
                                    # 排除值为0的数据
                                    dw1o_valid = dw1o_valid[dw1o_valid != 0]
                                    if len(dw1o_valid) > 0:
                                        bgmean_stats = {
                                            'min': dw1o_valid.min(),
                                            'max': dw1o_valid.max(),
                                            'mean': dw1o_valid.mean()
                                        }
                                
                                # 存储数据用于统计
                                all_data.append({
                                    'folder': folder_name,
                                    'count': len(x_valid),
                                    'x_data': x_valid,
                                    'y_data': y_valid,
                                    'bgmean_stats': bgmean_stats
                                })
                                
                                # 创建悬浮文本
                                hover_text = [
                                    f"<b>{folder_name}</b><br>X: {x:.2f}<br>Y: {y:.2f}"
                                    for x, y in zip(x_valid, y_valid)
                                ]
                                
                                # 添加散点图
                                color = colors[idx % len(colors)]
                                fig.add_trace(go.Scatter(
                                    x=x_valid,
                                    y=y_valid,
                                    mode='markers',
                                    name=f'{folder_name} ({len(x_valid)})',
                                    marker=dict(
                                        size=point_size,
                                        color=color,
                                        line=dict(width=0.5, color='white'),
                                        opacity=0.7
                                    ),
                                    hovertext=hover_text,
                                    hoverinfo='text'
                                ))
                    
                    # 添加晶圆边界圆
                    theta = np.linspace(0, 2*np.pi, 100)
                    circle_x = center_x + plot_range * np.cos(theta)
                    circle_y = center_y + plot_range * np.sin(theta)
                    
                    fig.add_trace(go.Scatter(
                        x=circle_x,
                        y=circle_y,
                        mode='lines',
                        name='晶圆边界',
                        line=dict(color='gray', width=2, dash='dash'),
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    
                    # 设置布局
                    fig.update_layout(
                        title=dict(
                            text='多文件夹缺陷对比图',
                            x=0.5,
                            xanchor='center',
                            font=dict(size=20)
                        ),
                        xaxis=dict(
                            title='X坐标',
                            range=[center_x - plot_range - 10000, center_x + plot_range + 10000],
                            scaleanchor="y",
                            scaleratio=1,
                            showgrid=show_grid_upload,
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(
                            title='Y坐标',
                            range=[center_y - plot_range - 10000, center_y + plot_range + 10000],
                            showgrid=show_grid_upload,
                            gridcolor='lightgray'
                        ),
                        plot_bgcolor='white',
                        hovermode='closest',
                        width=900,
                        height=900,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='gray',
                            borderwidth=1
                        )
                    )
                    
                    # 显示图形
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示统计信息
                    st.subheader("统计信息")
                    
                    # 创建统计表格
                    if all_data:
                        stats_data = []
                        total_defects = sum(d['count'] for d in all_data)
                        
                        for data in all_data:
                            row = {
                                '文件夹': data['folder'],
                                '缺陷数量': data['count'],
                                '占比': f"{(data['count'] / total_defects * 100):.2f}%"
                            }
                            
                            # 添加DW1O_BGMean统计
                            if data['bgmean_stats']:
                                row['BGMean最小值'] = f"{data['bgmean_stats']['min']:.2f}"
                                row['BGMean最大值'] = f"{data['bgmean_stats']['max']:.2f}"
                                row['BGMean均值'] = f"{data['bgmean_stats']['mean']:.2f}"
                            else:
                                row['BGMean最小值'] = 'N/A'
                                row['BGMean最大值'] = 'N/A'
                                row['BGMean均值'] = 'N/A'
                            
                            stats_data.append(row)
                        
                        stats_df = pd.DataFrame(stats_data)
                        
                        # 显示指标卡片
                        cols = st.columns(min(len(all_data) + 1, 5))
                        for i, data in enumerate(all_data):
                            with cols[i % 5]:
                                st.metric(data['folder'], data['count'])
                        
                        if len(all_data) % 5 == 4 or len(all_data) < 5:
                            with cols[len(all_data) % 5 if len(all_data) < 5 else 4]:
                                st.metric("总计", total_defects)
                        else:
                            st.metric("总计", total_defects)
                        
                        # 显示详细统计表
                        st.write("### 详细统计")
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # 柱状图对比
                        st.write("### 缺陷数量对比")
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=[d['folder'] for d in all_data],
                                y=[d['count'] for d in all_data],
                                marker_color=[colors[i % len(colors)] for i in range(len(all_data))],
                                text=[d['count'] for d in all_data],
                                textposition='auto',
                            )
                        ])
                        fig_bar.update_layout(
                            title='各文件夹缺陷数量对比',
                            xaxis_title='文件夹',
                            yaxis_title='缺陷数量',
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # 单独显示每个文件夹 - 每行5个
                        st.write("---")
                        st.subheader("各文件夹单独显示")
                        
                        # 每行显示5个文件夹
                        num_per_row = 5
                        for row_start in range(0, len(all_data), num_per_row):
                            cols = st.columns(num_per_row)
                            row_end = min(row_start + num_per_row, len(all_data))
                            
                            for col_idx, data_idx in enumerate(range(row_start, row_end)):
                                data = all_data[data_idx]
                                
                                with cols[col_idx]:
                                    with st.expander(f"📊 {data['folder']}", expanded=False):
                                        st.write(f"**缺陷数量: {data['count']}**")
                                        
                                        # 创建单独的图形
                                        fig_single = go.Figure()
                                        
                                        # 创建悬浮文本
                                        hover_text = [
                                            f"<b>{data['folder']}</b><br>X: {x:.2f}<br>Y: {y:.2f}"
                                            for x, y in zip(data['x_data'], data['y_data'])
                                        ]
                                        
                                        # 添加散点图
                                        color = colors[data_idx % len(colors)]
                                        fig_single.add_trace(go.Scatter(
                                            x=data['x_data'],
                                            y=data['y_data'],
                                            mode='markers',
                                            name=f"{data['folder']}",
                                            marker=dict(
                                                size=point_size + 2,  # 单独显示时点稍微大一点
                                                color=color,
                                                line=dict(width=0.5, color='white'),
                                                opacity=0.8
                                            ),
                                            hovertext=hover_text,
                                            hoverinfo='text'
                                        ))
                                        
                                        # 添加晶圆边界圆
                                        theta = np.linspace(0, 2*np.pi, 100)
                                        circle_x = center_x + plot_range * np.cos(theta)
                                        circle_y = center_y + plot_range * np.sin(theta)
                                        
                                        fig_single.add_trace(go.Scatter(
                                            x=circle_x,
                                            y=circle_y,
                                            mode='lines',
                                            name='晶圆边界',
                                            line=dict(color='gray', width=2, dash='dash'),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                        
                                        # 设置布局
                                        fig_single.update_layout(
                                            title=dict(
                                                text=f'{data["folder"]}',
                                                x=0.5,
                                                xanchor='center',
                                                font=dict(size=14)
                                            ),
                                            xaxis=dict(
                                                title='',
                                                range=[center_x - plot_range - 10000, center_x + plot_range + 10000],
                                                scaleanchor="y",
                                                scaleratio=1,
                                                showgrid=show_grid_upload,
                                                gridcolor='lightgray',
                                                showticklabels=False
                                            ),
                                            yaxis=dict(
                                                title='',
                                                range=[center_y - plot_range - 10000, center_y + plot_range + 10000],
                                                showgrid=show_grid_upload,
                                                gridcolor='lightgray',
                                                showticklabels=False
                                            ),
                                            plot_bgcolor='white',
                                            hovermode='closest',
                                            height=300,
                                            showlegend=False,
                                            margin=dict(l=20, r=20, t=40, b=20)
                                        )
                                        
                                        # 显示图形
                                        st.plotly_chart(fig_single, use_container_width=True)
                                        
                                        # 显示该文件夹的统计信息
                                        st.metric("缺陷数", data['count'])
                                        st.caption(f"X: {data['x_data'].min():.0f}~{data['x_data'].max():.0f}")
                                        st.caption(f"Y: {data['y_data'].min():.0f}~{data['y_data'].max():.0f}")
                                        
                                        # 显示BGMean统计
                                        if data['bgmean_stats']:
                                            st.caption(f"BGMean: {data['bgmean_stats']['min']:.2f}~{data['bgmean_stats']['max']:.2f}")
                                            st.caption(f"BGMean均值: {data['bgmean_stats']['mean']:.2f}")
                    
            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")
                st.exception(e)
    
    elif folder_path and os.path.exists(folder_path):
        try:
            # 获取所有子文件夹（不再排除kla文件夹）
            subfolders = [f for f in os.listdir(folder_path) 
                         if os.path.isdir(os.path.join(folder_path, f))]
            
            if not subfolders:
                st.warning("未找到子文件夹")
            else:
                st.success(f"找到 {len(subfolders)} 个子文件夹")
                
                # 显示找到的文件夹
                with st.expander("子文件夹列表"):
                    st.write(subfolders)
                
                # 绘图参数
                st.subheader("绘图参数")
                col1, col2 = st.columns(2)
                with col1:
                    center_x = st.number_input("中心X坐标", value=150000.0, key="folder_center_x")
                    center_y = st.number_input("中心Y坐标", value=150000.0, key="folder_center_y")
                with col2:
                    plot_range = st.number_input("绘图范围（半径）", value=150000.0, min_value=1000.0, key="folder_range")
                    point_size = st.slider("点的大小", min_value=3, max_value=15, value=6, key="folder_size")
                
                # 网格显示选项
                show_grid_folder = st.checkbox("显示背景网格", value=True, key="show_grid_folder", help="控制图表中是否显示背景网格线")
                
                # 数据过滤选项
                st.subheader("数据过滤选项")
                filter_special_types = st.checkbox(
                    "过滤特殊类型缺陷（nDefectType=1000/10001）", 
                    value=True,
                    key="filter_special_types_folder",
                    help="仅对CASI数据（BlobFeatures.csv）有效，KLA数据不受影响"
                )
                
                if st.button("生成对比图", type="primary", key="folder_plot_btn"):
                    # 准备颜色列表
                    colors = [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AED6F1',
                        '#A9DFBF', '#F9E79F', '#D7BDE2', '#A2D9CE', '#FAD7A0'
                    ]
                    
                    # 创建图形
                    fig = go.Figure()
                    
                    # 用于存储所有数据
                    all_data = []
                    
                    # 读取每个子文件夹
                    for idx, subfolder in enumerate(sorted(subfolders)):
                        # 判断是否为KLA文件夹
                        is_kla_folder = 'kla' in subfolder.lower()
                        
                        if is_kla_folder:
                            # KLA文件夹：读取jianchu.csv文件
                            csv_path = os.path.join(folder_path, subfolder, 'jianchu.csv')
                            
                            if not os.path.exists(csv_path):
                                st.warning(f"未找到 {subfolder}/jianchu.csv")
                                continue
                            
                            # 读取CSV
                            df = pd.read_csv(csv_path)
                            
                            # KLA数据不过滤nDefectType
                            # 使用XREL和YREL作为坐标列
                            if 'XREL' in df.columns and 'YREL' in df.columns:
                                x_col = 'XREL'
                                y_col = 'YREL'
                                
                                x_data = pd.to_numeric(df[x_col], errors='coerce')
                                y_data = pd.to_numeric(df[y_col], errors='coerce')
                                
                                # KLA数据没有DW1O_BGMean
                                dw1o_bgmean = None
                            else:
                                st.warning(f"{subfolder}: jianchu.csv缺少XREL/YREL列")
                                continue
                        else:
                            # 普通文件夹：查找BlobFeatures.csv或BlobFeatures*.csv文件
                            csv_files = glob.glob(os.path.join(folder_path, subfolder, 'BlobFeatures*.csv'))
                            
                            if not csv_files:
                                st.warning(f"未找到 {subfolder}/BlobFeatures*.csv")
                                continue
                            
                            csv_path = csv_files[0]  # 使用找到的第一个文件
                            # 读取CSV
                            df = pd.read_csv(csv_path)
                            
                            # 根据用户选择过滤nDefectType为1000和10001的数据
                            if filter_special_types and 'nDefectType' in df.columns:
                                df = df[(df['nDefectType'] != 1000) & (df['nDefectType'] != 10001)]
                            
                            # 获取第4列和第5列（索引3和4）
                            if df.shape[1] >= 5:
                                x_col = df.columns[3]
                                y_col = df.columns[4]
                                
                                x_data = pd.to_numeric(df[x_col], errors='coerce')
                                y_data = pd.to_numeric(df[y_col], errors='coerce')
                                
                                # 获取DW1O_BGMean列（如果存在）
                                dw1o_bgmean = None
                                if 'DW1O_BGMean' in df.columns:
                                    dw1o_bgmean = pd.to_numeric(df['DW1O_BGMean'], errors='coerce')
                            else:
                                st.warning(f"{subfolder}: BlobFeatures.csv列数不足")
                                continue
                        
                        # 过滤有效数据（对KLA和CASI数据都适用）
                        valid_mask = pd.notna(x_data) & pd.notna(y_data)
                        x_valid = x_data[valid_mask]
                        y_valid = y_data[valid_mask]
                        
                        if len(x_valid) > 0:
                            # 计算DW1O_BGMean统计值（排除0值）
                            bgmean_stats = {}
                            if dw1o_bgmean is not None:
                                dw1o_valid = dw1o_bgmean[valid_mask].dropna()
                                # 排除值为0的数据
                                dw1o_valid = dw1o_valid[dw1o_valid != 0]
                                if len(dw1o_valid) > 0:
                                    bgmean_stats = {
                                        'min': dw1o_valid.min(),
                                        'max': dw1o_valid.max(),
                                        'mean': dw1o_valid.mean()
                                    }
                            
                            # 存储数据用于统计
                            all_data.append({
                                'folder': subfolder,
                                'count': len(x_valid),
                                'x_data': x_valid,
                                'y_data': y_valid,
                                'bgmean_stats': bgmean_stats
                            })
                            
                            # 创建悬浮文本
                            hover_text = [
                                f"<b>{subfolder}</b><br>X: {x:.2f}<br>Y: {y:.2f}"
                                for x, y in zip(x_valid, y_valid)
                            ]
                            
                            # 添加散点图
                            color = colors[idx % len(colors)]
                            fig.add_trace(go.Scatter(
                                x=x_valid,
                                y=y_valid,
                                mode='markers',
                                name=f'{subfolder} ({len(x_valid)})',
                                marker=dict(
                                    size=point_size,
                                    color=color,
                                    line=dict(width=0.5, color='white'),
                                    opacity=0.7
                                ),
                                hovertext=hover_text,
                                hoverinfo='text'
                            ))
                    
                    if not all_data:
                        st.error("未找到有效的数据文件")
                    else:
                        # 添加晶圆边界圆
                        theta = np.linspace(0, 2*np.pi, 100)
                        circle_x = center_x + plot_range * np.cos(theta)
                        circle_y = center_y + plot_range * np.sin(theta)
                        
                        fig.add_trace(go.Scatter(
                            x=circle_x,
                            y=circle_y,
                            mode='lines',
                            name='晶圆边界',
                            line=dict(color='gray', width=2, dash='dash'),
                            showlegend=True,
                            hoverinfo='skip'
                        ))
                        
                        # 设置布局
                        fig.update_layout(
                            title=dict(
                                text='多文件夹缺陷对比图',
                                x=0.5,
                                xanchor='center',
                                font=dict(size=20)
                            ),
                            xaxis=dict(
                                title='X坐标',
                                range=[center_x - plot_range - 10000, center_x + plot_range + 10000],
                                scaleanchor="y",
                                scaleratio=1,
                                showgrid=show_grid_folder,
                                gridcolor='lightgray'
                            ),
                            yaxis=dict(
                                title='Y坐标',
                                range=[center_y - plot_range - 10000, center_y + plot_range + 10000],
                                showgrid=show_grid_folder,
                                gridcolor='lightgray'
                            ),
                            plot_bgcolor='white',
                            hovermode='closest',
                            width=900,
                            height=900,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02,
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='gray',
                                borderwidth=1
                            )
                        )
                        
                        # 显示图形
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示统计信息
                        st.subheader("统计信息")
                        
                        # 创建统计表格
                        stats_data = []
                        total_defects = sum(d['count'] for d in all_data)
                        
                        for data in all_data:
                            row = {
                                '文件夹': data['folder'],
                                '缺陷数量': data['count'],
                                '占比': f"{(data['count'] / total_defects * 100):.2f}%"
                            }
                            
                            # 添加DW1O_BGMean统计
                            if data['bgmean_stats']:
                                row['BGMean最小值'] = f"{data['bgmean_stats']['min']:.2f}"
                                row['BGMean最大值'] = f"{data['bgmean_stats']['max']:.2f}"
                                row['BGMean均值'] = f"{data['bgmean_stats']['mean']:.2f}"
                            else:
                                row['BGMean最小值'] = 'N/A'
                                row['BGMean最大值'] = 'N/A'
                                row['BGMean均值'] = 'N/A'
                            
                            stats_data.append(row)
                        
                        stats_df = pd.DataFrame(stats_data)
                        
                        # 显示指标卡片
                        cols = st.columns(min(len(all_data) + 1, 5))
                        for i, data in enumerate(all_data):
                            with cols[i % 5]:
                                st.metric(data['folder'], data['count'])
                        
                        if len(all_data) % 5 == 4 or len(all_data) < 5:
                            with cols[len(all_data) % 5 if len(all_data) < 5 else 4]:
                                st.metric("总计", total_defects)
                        else:
                            st.metric("总计", total_defects)
                        
                        # 显示详细统计表
                        st.write("### 详细统计")
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # 柱状图对比
                        st.write("### 缺陷数量对比")
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=[d['folder'] for d in all_data],
                                y=[d['count'] for d in all_data],
                                marker_color=[colors[i % len(colors)] for i in range(len(all_data))],
                                text=[d['count'] for d in all_data],
                                textposition='auto',
                            )
                        ])
                        fig_bar.update_layout(
                            title='各文件夹缺陷数量对比',
                            xaxis_title='文件夹',
                            yaxis_title='缺陷数量',
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # 单独显示每个文件夹 - 每行5个
                        st.write("---")
                        st.subheader("各文件夹单独显示")
                        
                        # 每行显示5个文件夹
                        num_per_row = 5
                        for row_start in range(0, len(all_data), num_per_row):
                            cols = st.columns(num_per_row)
                            row_end = min(row_start + num_per_row, len(all_data))
                            
                            for col_idx, data_idx in enumerate(range(row_start, row_end)):
                                data = all_data[data_idx]
                                
                                with cols[col_idx]:
                                    with st.expander(f"📊 {data['folder']}", expanded=False):
                                        st.write(f"**缺陷数量: {data['count']}**")
                                        
                                        # 创建单独的图形
                                        fig_single = go.Figure()
                                        
                                        # 创建悬浮文本
                                        hover_text = [
                                            f"<b>{data['folder']}</b><br>X: {x:.2f}<br>Y: {y:.2f}"
                                            for x, y in zip(data['x_data'], data['y_data'])
                                        ]
                                        
                                        # 添加散点图
                                        color = colors[data_idx % len(colors)]
                                        fig_single.add_trace(go.Scatter(
                                            x=data['x_data'],
                                            y=data['y_data'],
                                            mode='markers',
                                            name=f"{data['folder']}",
                                            marker=dict(
                                                size=point_size + 2,  # 单独显示时点稍微大一点
                                                color=color,
                                                line=dict(width=0.5, color='white'),
                                                opacity=0.8
                                            ),
                                            hovertext=hover_text,
                                            hoverinfo='text'
                                        ))
                                        
                                        # 添加晶圆边界圆
                                        theta_single = np.linspace(0, 2*np.pi, 100)
                                        circle_x_single = center_x + plot_range * np.cos(theta_single)
                                        circle_y_single = center_y + plot_range * np.sin(theta_single)
                                        
                                        fig_single.add_trace(go.Scatter(
                                            x=circle_x_single,
                                            y=circle_y_single,
                                            mode='lines',
                                            name='晶圆边界',
                                            line=dict(color='gray', width=2, dash='dash'),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                        
                                        # 设置布局
                                        fig_single.update_layout(
                                            title=dict(
                                                text=f'{data["folder"]}',
                                                x=0.5,
                                                xanchor='center',
                                                font=dict(size=14)
                                            ),
                                            xaxis=dict(
                                                title='',
                                                range=[center_x - plot_range - 10000, center_x + plot_range + 10000],
                                                scaleanchor="y",
                                                scaleratio=1,
                                                showgrid=show_grid_folder,
                                                gridcolor='lightgray',
                                                showticklabels=False
                                            ),
                                            yaxis=dict(
                                                title='',
                                                range=[center_y - plot_range - 10000, center_y + plot_range + 10000],
                                                showgrid=show_grid_folder,
                                                gridcolor='lightgray',
                                                showticklabels=False
                                            ),
                                            plot_bgcolor='white',
                                            hovermode='closest',
                                            height=300,
                                            showlegend=False,
                                            margin=dict(l=20, r=20, t=40, b=20)
                                        )
                                        
                                        # 显示图形
                                        st.plotly_chart(fig_single, use_container_width=True)
                                        
                                        # 显示该文件夹的统计信息
                                        st.metric("缺陷数", data['count'])
                                        st.caption(f"X: {data['x_data'].min():.0f}~{data['x_data'].max():.0f}")
                                        st.caption(f"Y: {data['y_data'].min():.0f}~{data['y_data'].max():.0f}")
                                        
                                        # 显示BGMean统计
                                        if data['bgmean_stats']:
                                            st.caption(f"BGMean: {data['bgmean_stats']['min']:.2f}~{data['bgmean_stats']['max']:.2f}")
                                            st.caption(f"BGMean均值: {data['bgmean_stats']['mean']:.2f}")
                
        except Exception as e:
            st.error(f"处理文件夹时出错: {str(e)}")
            st.exception(e)
    
    elif folder_path:
        st.error("文件夹路径不存在，请检查路径是否正确")

    # 新增：缺陷坐标匹配和SNR对比分析
    st.write("---")
    st.header("📍 缺陷坐标匹配与SNR对比")
    
    st.markdown("""
    ### 功能说明
    对多个文件夹中的 `jianchu.csv` 文件进行坐标匹配，比较相同缺陷在不同工况下的DW1O_TotalSNR值。
    
    **功能特点：**
    - 匹配范围：50个单位（可调整）
    - 自动排除文件夹名称包含"KLA"的文件
    - 输出匹配结果表格，包含坐标、各工况SNR值
    - 支持导出为CSV文件
    """)
    
    # 输入文件夹路径
    match_folder_path = st.text_input("输入主文件夹路径（用于匹配分析）", 
                                      placeholder=r"例如: D:\data\wafer_folders",
                                      key="match_folder_path")
    
    # 匹配参数
    col1, col2 = st.columns(2)
    with col1:
        match_threshold = st.number_input("匹配距离阈值", value=50.0, min_value=1.0, max_value=500.0, 
                                         help="两个缺陷之间的最大距离，超过此距离则不视为同一缺陷")
    with col2:
        show_unmatched = st.checkbox("显示未匹配的缺陷", value=False,
                                     help="是否在结果中包含只在单个工况中出现的缺陷")
    
    if match_folder_path and os.path.exists(match_folder_path):
        if st.button("开始匹配分析", type="primary", key="match_analysis_btn"):
            try:
                # 获取所有子文件夹，排除包含"KLA"的文件夹
                all_subfolders = [f for f in os.listdir(match_folder_path) 
                                 if os.path.isdir(os.path.join(match_folder_path, f))]
                
                subfolders = [f for f in all_subfolders if 'KLA' not in f.upper()]
                kla_folders = [f for f in all_subfolders if 'KLA' in f.upper()]
                
                if not subfolders:
                    st.warning("未找到有效的子文件夹（排除KLA文件夹后）")
                else:
                    st.info(f"找到 {len(subfolders)} 个有效文件夹，已排除 {len(kla_folders)} 个KLA文件夹")
                    
                    if kla_folders:
                        with st.expander("已排除的KLA文件夹"):
                            st.write(kla_folders)
                    
                    # 读取所有文件夹的数据
                    folder_data = {}
                    
                    with st.spinner("正在读取数据..."):
                        for subfolder in sorted(subfolders):
                            csv_path = os.path.join(match_folder_path, subfolder, 'jianchu.csv')
                            
                            if os.path.exists(csv_path):
                                df = pd.read_csv(csv_path)
                                
                                # 获取第4列和第5列作为坐标
                                if df.shape[1] >= 5:
                                    x_col = df.columns[3]
                                    y_col = df.columns[4]
                                    
                                    # 获取DW1O_TotalSNR、DW1O_MaxOrg、DW1O_BGDev列
                                    snr_col = 'DW1O_TotalSNR' if 'DW1O_TotalSNR' in df.columns else None
                                    maxorg_col = 'DW1O_MaxOrg' if 'DW1O_MaxOrg' in df.columns else None
                                    bgdev_col = 'DW1O_BGDev' if 'DW1O_BGDev' in df.columns else None
                                    
                                    # 创建数据字典
                                    data_dict = {
                                        'x': pd.to_numeric(df[x_col], errors='coerce'),
                                        'y': pd.to_numeric(df[y_col], errors='coerce'),
                                    }
                                    
                                    if snr_col:
                                        data_dict['snr'] = pd.to_numeric(df[snr_col], errors='coerce')
                                    else:
                                        data_dict['snr'] = None
                                    
                                    if maxorg_col:
                                        data_dict['maxorg'] = pd.to_numeric(df[maxorg_col], errors='coerce')
                                    else:
                                        data_dict['maxorg'] = None
                                    
                                    if bgdev_col:
                                        data_dict['bgdev'] = pd.to_numeric(df[bgdev_col], errors='coerce')
                                    else:
                                        data_dict['bgdev'] = None
                                    
                                    # 创建DataFrame并过滤有效数据
                                    temp_df = pd.DataFrame(data_dict)
                                    temp_df = temp_df.dropna(subset=['x', 'y'])
                                    
                                    folder_data[subfolder] = temp_df
                                    st.success(f"✓ {subfolder}: {len(temp_df)} 个缺陷")
                    
                    if len(folder_data) < 2:
                        st.warning("需要至少2个文件夹的数据才能进行匹配")
                    else:
                        # 自定义排序函数：将包含KLA的文件夹排到最后
                        def sort_folders_kla_last(folders):
                            non_kla = sorted([f for f in folders if 'KLA' not in f.upper()])
                            kla = sorted([f for f in folders if 'KLA' in f.upper()])
                            return non_kla + kla
                        
                        # 执行匹配
                        with st.spinner("正在匹配缺陷..."):
                            # 使用第一个文件夹作为基准（非KLA）
                            sorted_folders = sort_folders_kla_last(folder_data.keys())
                            base_folder = sorted_folders[0]
                            base_data = folder_data[base_folder]
                            
                            # 存储匹配结果
                            match_results = []
                            
                            # 对基准文件夹中的每个缺陷进行匹配
                            for idx, row in base_data.iterrows():
                                base_x = row['x']
                                base_y = row['y']
                                base_snr = row['snr']
                                base_maxorg = row['maxorg']
                                base_bgdev = row['bgdev']
                                
                                result = {
                                    'X坐标': base_x,
                                    'Y坐标': base_y,
                                    f'{base_folder}_SNR': base_snr if pd.notna(base_snr) else None,
                                    f'{base_folder}_MaxOrg': base_maxorg if pd.notna(base_maxorg) else None,
                                    f'{base_folder}_BGDev': base_bgdev if pd.notna(base_bgdev) else None
                                }
                                
                                matched_count = 1  # 至少匹配到基准文件夹本身
                                
                                # 在其他文件夹中查找匹配的缺陷
                                for other_folder in sorted_folders:
                                    if other_folder == base_folder:
                                        continue
                                    
                                    other_data = folder_data[other_folder]
                                    
                                    # 计算距离
                                    distances = np.sqrt(
                                        (other_data['x'] - base_x)**2 + 
                                        (other_data['y'] - base_y)**2
                                    )
                                    
                                    # 找到最近的匹配
                                    if len(distances) > 0:
                                        min_dist_idx = distances.idxmin()
                                        min_dist = distances[min_dist_idx]
                                        
                                        if min_dist <= match_threshold:
                                            matched_snr = other_data.loc[min_dist_idx, 'snr']
                                            matched_maxorg = other_data.loc[min_dist_idx, 'maxorg']
                                            matched_bgdev = other_data.loc[min_dist_idx, 'bgdev']
                                            result[f'{other_folder}_SNR'] = matched_snr if pd.notna(matched_snr) else None
                                            result[f'{other_folder}_MaxOrg'] = matched_maxorg if pd.notna(matched_maxorg) else None
                                            result[f'{other_folder}_BGDev'] = matched_bgdev if pd.notna(matched_bgdev) else None
                                            # result[f'{other_folder}_距离'] = min_dist
                                            matched_count += 1
                                        else:
                                            result[f'{other_folder}_SNR'] = None
                                            result[f'{other_folder}_MaxOrg'] = None
                                            result[f'{other_folder}_BGDev'] = None
                                            # result[f'{other_folder}_距离'] = None
                                    else:
                                        result[f'{other_folder}_SNR'] = None
                                        result[f'{other_folder}_MaxOrg'] = None
                                        result[f'{other_folder}_BGDev'] = None
                                        # result[f'{other_folder}_距离'] = None
                                
                                # 根据设置决定是否添加此结果
                                if show_unmatched or matched_count > 1:
                                    result['匹配数量'] = matched_count
                                    match_results.append(result)
                            
                            # 创建结果DataFrame
                            if match_results:
                                results_df = pd.DataFrame(match_results)
                                
                                # 重新排列列的顺序（KLA文件夹列在最后）
                                cols = ['X坐标', 'Y坐标', '匹配数量']
                                for folder in sorted_folders:
                                    if f'{folder}_SNR' in results_df.columns:
                                        cols.append(f'{folder}_SNR')
                                    if f'{folder}_MaxOrg' in results_df.columns:
                                        cols.append(f'{folder}_MaxOrg')
                                    if f'{folder}_BGDev' in results_df.columns:
                                        cols.append(f'{folder}_BGDev')
                                    if f'{folder}_距离' in results_df.columns:
                                        cols.append(f'{folder}_距离')
                                
                                results_df = results_df[cols]
                                
                                # 显示统计信息
                                st.subheader("匹配统计")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("总缺陷数（基准）", len(base_data))
                                with col2:
                                    st.metric("匹配结果数", len(results_df))
                                with col3:
                                    fully_matched = len(results_df[results_df['匹配数量'] == len(folder_data)])
                                    st.metric("完全匹配", fully_matched)
                                with col4:
                                    partial_matched = len(results_df[results_df['匹配数量'] > 1]) - fully_matched
                                    st.metric("部分匹配", partial_matched)
                                
                                # 显示结果表格
                                st.subheader("匹配结果详细表格")
                                st.dataframe(results_df, use_container_width=True, height=400)
                                
                                # 提供CSV下载
                                csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 下载匹配结果（CSV）",
                                    data=csv,
                                    file_name=f"defect_match_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # SNR对比分析
                                st.subheader("SNR对比分析")
                                
                                # 只分析完全匹配的缺陷
                                fully_matched_df = results_df[results_df['匹配数量'] == len(folder_data)].copy()
                                
                                if len(fully_matched_df) > 0:
                                    st.write(f"完全匹配的缺陷数量: {len(fully_matched_df)}")
                                    
                                    # 创建SNR对比箱线图
                                    snr_columns = [col for col in results_df.columns if col.endswith('_SNR')]
                                    
                                    if snr_columns:
                                        # 准备绘图数据
                                        plot_data = []
                                        for col in snr_columns:
                                            folder_name = col.replace('_SNR', '')
                                            snr_values = fully_matched_df[col].dropna()
                                            for val in snr_values:
                                                plot_data.append({
                                                    '工况': folder_name,
                                                    'SNR': val
                                                })
                                        
                                        if plot_data:
                                            plot_df = pd.DataFrame(plot_data)
                                            
                                            # 箱线图（KLA文件夹在最后）
                                            fig_box = go.Figure()
                                            for folder in sorted_folders:
                                                folder_snr = plot_df[plot_df['工况'] == folder]['SNR']
                                                fig_box.add_trace(go.Box(
                                                    y=folder_snr,
                                                    name=folder,
                                                    boxmean='sd'
                                                ))
                                            
                                            fig_box.update_layout(
                                                title='各工况SNR分布对比（完全匹配缺陷）',
                                                yaxis_title='DW1O_TotalSNR',
                                                xaxis_title='工况',
                                                showlegend=False,
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig_box, use_container_width=True)
                                            
                                            # 统计摘要 - SNR（KLA文件夹在最后）
                                            st.write("### SNR统计摘要")
                                            summary_data = []
                                            for folder in sorted_folders:
                                                folder_snr = plot_df[plot_df['工况'] == folder]['SNR']
                                                if len(folder_snr) > 0:
                                                    summary_data.append({
                                                        '工况': folder,
                                                        '平均值': folder_snr.mean(),
                                                        '中位数': folder_snr.median(),
                                                        '标准差': folder_snr.std(),
                                                        '最小值': folder_snr.min(),
                                                        '最大值': folder_snr.max()
                                                    })
                                            
                                            summary_df = pd.DataFrame(summary_data)
                                            st.dataframe(summary_df.round(2), use_container_width=True)
                                    
                                    # MaxOrg对比分析
                                    maxorg_columns = [col for col in results_df.columns if col.endswith('_MaxOrg')]
                                    
                                    if maxorg_columns:
                                        st.write("### DW1O_MaxOrg对比分析")
                                        
                                        # 准备绘图数据
                                        plot_data_maxorg = []
                                        for col in maxorg_columns:
                                            folder_name = col.replace('_MaxOrg', '')
                                            maxorg_values = fully_matched_df[col].dropna()
                                            for val in maxorg_values:
                                                plot_data_maxorg.append({
                                                    '工况': folder_name,
                                                    'MaxOrg': val
                                                })
                                        
                                        if plot_data_maxorg:
                                            plot_df_maxorg = pd.DataFrame(plot_data_maxorg)
                                            
                                            # 箱线图（KLA文件夹在最后）
                                            fig_maxorg = go.Figure()
                                            for folder in sorted_folders:
                                                folder_maxorg = plot_df_maxorg[plot_df_maxorg['工况'] == folder]['MaxOrg']
                                                fig_maxorg.add_trace(go.Box(
                                                    y=folder_maxorg,
                                                    name=folder,
                                                    boxmean='sd'
                                                ))
                                            
                                            fig_maxorg.update_layout(
                                                title='各工况MaxOrg分布对比（完全匹配缺陷）',
                                                yaxis_title='DW1O_MaxOrg',
                                                xaxis_title='工况',
                                                showlegend=False,
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig_maxorg, use_container_width=True)
                                            
                                            # 统计摘要（KLA文件夹在最后）
                                            summary_maxorg = []
                                            for folder in sorted_folders:
                                                folder_maxorg = plot_df_maxorg[plot_df_maxorg['工况'] == folder]['MaxOrg']
                                                if len(folder_maxorg) > 0:
                                                    summary_maxorg.append({
                                                        '工况': folder,
                                                        '平均值': folder_maxorg.mean(),
                                                        '中位数': folder_maxorg.median(),
                                                        '标准差': folder_maxorg.std(),
                                                        '最小值': folder_maxorg.min(),
                                                        '最大值': folder_maxorg.max()
                                                    })
                                            
                                            summary_maxorg_df = pd.DataFrame(summary_maxorg)
                                            st.dataframe(summary_maxorg_df.round(2), use_container_width=True)
                                    
                                    # BGDev对比分析
                                    bgdev_columns = [col for col in results_df.columns if col.endswith('_BGDev')]
                                    
                                    if bgdev_columns:
                                        st.write("### DW1O_BGDev对比分析")
                                        
                                        # 准备绘图数据
                                        plot_data_bgdev = []
                                        for col in bgdev_columns:
                                            folder_name = col.replace('_BGDev', '')
                                            bgdev_values = fully_matched_df[col].dropna()
                                            for val in bgdev_values:
                                                plot_data_bgdev.append({
                                                    '工况': folder_name,
                                                    'BGDev': val
                                                })
                                        
                                        if plot_data_bgdev:
                                            plot_df_bgdev = pd.DataFrame(plot_data_bgdev)
                                            
                                            # 箱线图（KLA文件夹在最后）
                                            fig_bgdev = go.Figure()
                                            for folder in sorted_folders:
                                                folder_bgdev = plot_df_bgdev[plot_df_bgdev['工况'] == folder]['BGDev']
                                                fig_bgdev.add_trace(go.Box(
                                                    y=folder_bgdev,
                                                    name=folder,
                                                    boxmean='sd'
                                                ))
                                            
                                            fig_bgdev.update_layout(
                                                title='各工况BGDev分布对比（完全匹配缺陷）',
                                                yaxis_title='DW1O_BGDev',
                                                xaxis_title='工况',
                                                showlegend=False,
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig_bgdev, use_container_width=True)
                                            
                                            # 统计摘要（KLA文件夹在最后）
                                            summary_bgdev = []
                                            for folder in sorted_folders:
                                                folder_bgdev = plot_df_bgdev[plot_df_bgdev['工况'] == folder]['BGDev']
                                                if len(folder_bgdev) > 0:
                                                    summary_bgdev.append({
                                                        '工况': folder,
                                                        '平均值': folder_bgdev.mean(),
                                                        '中位数': folder_bgdev.median(),
                                                        '标准差': folder_bgdev.std(),
                                                        '最小值': folder_bgdev.min(),
                                                        '最大值': folder_bgdev.max()
                                                    })
                                            
                                            summary_bgdev_df = pd.DataFrame(summary_bgdev)
                                            st.dataframe(summary_bgdev_df.round(2), use_container_width=True)
                                else:
                                    st.info("没有在所有工况中都匹配到的缺陷")
                            else:
                                st.warning("未找到匹配的缺陷")
                
            except Exception as e:
                st.error(f"匹配分析时出错: {str(e)}")
                st.exception(e)
    
    elif match_folder_path:
        st.error("文件夹路径不存在，请检查路径是否正确")

    # 新增：CASI与KLA匹配分析（基于您的match函数逻辑）
    st.write("---")
    st.header("🔍 CASI与KLA匹配分析")
    
    st.markdown("""
    ### 功能说明
    对多个子文件夹进行CASI与KLA的匹配分析，统计过检、漏检和正确检出。
    
    **两种数据输入方式：**
    - **方式1（原有方式）**：主文件夹包含多个子文件夹，每个子文件夹包含对应的CSV文件
      - 不含"KLA"的子文件夹中的 `BlobFeatures.csv` (CASI数据)
      - 含"KLA"的子文件夹中的 `jianchu.csv` (KLA数据)
    - **方式2（新增方式）**：单个文件夹包含所有CSV文件
      - 名称中包含"kla"的CSV文件为KLA数据
      - 其他CSV文件为CASI数据
    
    **匹配逻辑：**
    - CASI数据自动过滤nDefectType=1000和10001
    - 与KLA数据进行坐标匹配
    - 匹配结果分类：
      - 0 = 过检（CASI有但KLA无，且为非特殊类型）
      - -2 = 过检（CASI有但KLA无，但为特殊类型nDefectType=1000/10001）
      - 1 = 正确检出（一对一）
      - 3 = 正确检出（多CASI对一KLA，非特殊类型）
      - 4, 5 = 正确检出（一对多、多对多）
      - 2 = 漏检（KLA有但CASI无，或CASI为特殊类型）
      - -3 = 多CASI对一KLA中的特殊类型
    """)
    
    # 选择输入方式
    st.subheader("选择数据输入方式")
    input_mode = st.radio(
        "选择输入方式",
        ["方式1：选择主文件夹（原有方式）", "方式2：选择单个文件夹（包含所有CSV）"],
        key="kla_input_mode"
    )
    
    # 输入文件夹路径
    if input_mode == "方式1：选择主文件夹（原有方式）":
        kla_match_folder = st.text_input("输入主文件夹路径（用于KLA匹配）", 
                                         placeholder=r"例如: D:\data\wafer_folders",
                                         key="kla_match_folder")
        st.info("📁 方式1：主文件夹包含多个子文件夹，每个子文件夹包含BlobFeatures.csv或jianchu.csv")
    else:
        kla_match_folder = st.text_input("输入包含所有CSV文件的文件夹路径", 
                                         placeholder=r"例如: D:\data\csv_files",
                                         key="kla_match_folder_single")
        st.info("📁 方式2：文件夹内包含多个CSV文件，其中名称包含'kla'的为KLA数据，其他CSV为CASI数据")
    
    # 匹配参数
    col1, col2 = st.columns(2)
    with col1:
        kla_match_threshold = st.number_input("KLA匹配距离阈值", value=200.0, min_value=1.0, max_value=10000.0,
                                             help="CASI和KLA之间的最大匹配距离")
    with col2:
        block_size_param = st.number_input("分块大小", value=10000.0, min_value=1000.0,
                                          help="用于坐标分块处理（如果需要）")
    
    if kla_match_folder and os.path.exists(kla_match_folder):
        if st.button("开始KLA匹配分析", type="primary", key="kla_match_btn"):
            try:
                from scipy.spatial import KDTree
                
                # 根据输入方式处理
                if input_mode == "方式1：选择主文件夹（原有方式）":
                    # 原有方式：获取所有子文件夹
                    all_subfolders = [f for f in os.listdir(kla_match_folder) 
                                     if os.path.isdir(os.path.join(kla_match_folder, f))]
                    
                    # 分离CASI和KLA文件夹
                    casi_folders = [f for f in all_subfolders if 'KLA' not in f.upper()]
                    kla_folders = [f for f in all_subfolders if 'KLA' in f.upper()]
                else:
                    # 新方式：读取文件夹内的所有CSV文件
                    all_csv_files = [f for f in os.listdir(kla_match_folder) 
                                    if f.endswith('.csv') and os.path.isfile(os.path.join(kla_match_folder, f))]
                    
                    # 找出包含'kla'的文件（KLA数据）
                    kla_csv_files = [f for f in all_csv_files if 'kla' in f.lower()]
                    # 其他CSV文件为CASI数据
                    casi_csv_files = [f for f in all_csv_files if 'kla' not in f.lower()]
                    
                    # 为了保持后续逻辑一致，创建虚拟的文件夹名称
                    casi_folders = [f.replace('.csv', '') for f in casi_csv_files]
                    kla_folders = [f.replace('.csv', '') for f in kla_csv_files]
                    
                    st.info(f"📄 找到 {len(casi_csv_files)} 个CASI CSV文件，{len(kla_csv_files)} 个KLA CSV文件")
                
                if not casi_folders:
                    st.warning("未找到CASI数据")
                elif not kla_folders:
                    st.warning("未找到KLA数据")
                else:
                    if input_mode == "方式1：选择主文件夹（原有方式）":
                        st.info(f"找到 {len(casi_folders)} 个CASI文件夹，{len(kla_folders)} 个KLA文件夹")
                    else:
                        st.info(f"找到 {len(casi_folders)} 个CASI CSV文件，{len(kla_folders)} 个KLA CSV文件")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if input_mode == "方式1：选择主文件夹（原有方式）":
                            st.write("**CASI文件夹：**")
                        else:
                            st.write("**CASI CSV文件：**")
                        st.write(casi_folders)
                    with col2:
                        if input_mode == "方式1：选择主文件夹（原有方式）":
                            st.write("**KLA文件夹：**")
                        else:
                            st.write("**KLA CSV文件：**")
                        st.write(kla_folders)
                    
                    # 存储所有匹配结果
                    all_match_results = []
                    
                    # 初始化 session_state 中的匹配结果
                    if 'kla_match_results' not in st.session_state:
                        st.session_state.kla_match_results = []
                    
                    with st.spinner("正在执行KLA匹配..."):
                        # 对每个CASI文件夹进行处理
                        for casi_folder in sorted(casi_folders):
                            # 根据输入方式读取CASI文件
                            if input_mode == "方式1：选择主文件夹（原有方式）":
                                # 原有方式：查找子文件夹中的BlobFeatures文件
                                casi_csv_path = None
                                folder_path_full = os.path.join(kla_match_folder, casi_folder)
                                
                                # 查找包含BlobFeatures的CSV文件
                                for fname in os.listdir(folder_path_full):
                                    if 'BlobFeatures' in fname and fname.endswith('.csv'):
                                        casi_csv_path = os.path.join(folder_path_full, fname)
                                        break
                                
                                # 如果没找到BlobFeatures文件，尝试使用BlobFeatures.csv
                                if casi_csv_path is None:
                                    casi_csv_path = os.path.join(folder_path_full, 'BlobFeatures.csv')
                                
                                if not os.path.exists(casi_csv_path):
                                    st.warning(f"未找到 {casi_folder}/BlobFeatures*.csv")
                                    continue
                            else:
                                # 新方式：直接读取CSV文件
                                casi_csv_path = os.path.join(kla_match_folder, f"{casi_folder}.csv")
                                if not os.path.exists(casi_csv_path):
                                    st.warning(f"未找到 {casi_csv_path}")
                                    continue
                            
                            # 读取CASI数据（BlobFeatures）
                            casi_df = pd.read_csv(casi_csv_path)
                            casi_df.columns = casi_df.columns.str.strip()
                            
                            # **重要：过滤nDefectType不等于1000和10001的数据**
                            original_count = len(casi_df)
                            if 'nDefectType' in casi_df.columns:
                                # 先记录原始数据用于特殊类型判断
                                casi_df_full = casi_df.copy()
                                # 标记特殊类型（用于后续匹配逻辑）
                                casi_df['is_special_type'] = casi_df['nDefectType'].isin([1000, 10001])
                            else:
                                casi_df_full = casi_df.copy()
                                casi_df['is_special_type'] = False
                            
                            filtered_count = len(casi_df)
                            st.info(f"{casi_folder}: 原始数据 {original_count} 条，包含特殊类型标记")
                            
                            # 统计基础检出个数（全部数据）
                            blob_count = original_count
                            # 获取CASI坐标列
                            cas_x_col = None
                            cas_y_col = None
                            for x_candidate in ['dCenterXCartisian', 'dCenterXCartesian', 'XREL', 'cx']:
                                if x_candidate in casi_df.columns:
                                    cas_x_col = x_candidate
                                    break
                            for y_candidate in ['dCenterYCartisian', 'dCenterYCartesian', 'YREL', 'cy']:
                                if y_candidate in casi_df.columns:
                                    cas_y_col = y_candidate
                                    break
                            
                            if cas_x_col is None or cas_y_col is None:
                                st.warning(f"{casi_folder}: 未找到坐标列")
                                continue
                            
                            # 新增：计算到(150000, 150000)的距离
                            casi_df['distance_to_center'] = np.sqrt(
                                (casi_df[cas_x_col] - 150000)**2 + 
                                (casi_df[cas_y_col] - 150000)**2
                            )
                            # 标记距离>=147mm(147000um)的点
                            casi_df['is_edge_point'] = casi_df['distance_to_center'] >= 147000
                            
                            # 处理每个KLA文件夹
                            for kla_folder in sorted(kla_folders):
                                # 根据输入方式读取KLA文件
                                if input_mode == "方式1：选择主文件夹（原有方式）":
                                    # 原有方式：从子文件夹读取jianchu.csv
                                    kla_csv_path = os.path.join(kla_match_folder, kla_folder, 'jianchu.csv')
                                else:
                                    # 新方式：直接读取CSV文件
                                    kla_csv_path = os.path.join(kla_match_folder, f"{kla_folder}.csv")
                                
                                if not os.path.exists(kla_csv_path):
                                    continue
                                
                                # 读取KLA数据
                                kla_df = pd.read_csv(kla_csv_path)
                                kla_df.columns = kla_df.columns.str.strip()
                                
                                if not {'XREL', 'YREL'}.issubset(kla_df.columns):
                                    st.warning(f"{kla_folder}: 缺少XREL/YREL列")
                                    continue
                                
                                # 准备匹配数据（包含多个MaxOrg列用于识别污染）
                                # 检查是否有DW1O_MaxOrg、DW2O_MaxOrg、DN1O_MaxOrg列
                                maxorg_cols = []
                                if 'DW1O_MaxOrg' in casi_df.columns:
                                    maxorg_cols.append('DW1O_MaxOrg')
                                if 'DW2O_MaxOrg' in casi_df.columns:
                                    maxorg_cols.append('DW2O_MaxOrg')
                                if 'DN1O_MaxOrg' in casi_df.columns:
                                    maxorg_cols.append('DN1O_MaxOrg')
                                
                                has_maxorg = len(maxorg_cols) > 0
                                
                                # 检查是否有DW1O_Size和DW2O_Size列（用于过检尺寸分布统计）
                                size_cols = []
                                if 'DW1O_Size' in casi_df.columns:
                                    size_cols.append('DW1O_Size')
                                if 'DW2O_Size' in casi_df.columns:
                                    size_cols.append('DW2O_Size')
                                if 'DN1O_Size' in casi_df.columns:
                                    size_cols.append('DN1O_Size')
                                has_size_cols = len(size_cols) > 0
                                
                                # 检查是否有DW1O通道的SubRow和MainRow列（用于通道比值分析）
                                dw1o_channel_cols = []
                                if 'DW1O_SubRow1Max' in casi_df.columns:
                                    dw1o_channel_cols.append('DW1O_SubRow1Max')
                                if 'DW1O_SubRow2Max' in casi_df.columns:
                                    dw1o_channel_cols.append('DW1O_SubRow2Max')
                                if 'DW1O_MainRowMax' in casi_df.columns:
                                    dw1o_channel_cols.append('DW1O_MainRowMax')
                                has_dw1o_channels = len(dw1o_channel_cols) == 3  # 需要三个列都存在
                                
                                # 检查是否有BGMean列（用于背景均值分析）
                                bgmean_cols = []
                                if 'DW1O_BGMean' in casi_df.columns:
                                    bgmean_cols.append('DW1O_BGMean')
                                if 'DW2O_BGMean' in casi_df.columns:
                                    bgmean_cols.append('DW2O_BGMean')
                                if 'DN1O_BGMean' in casi_df.columns:
                                    bgmean_cols.append('DN1O_BGMean')
                                if 'DW1O_BGDev' in casi_df.columns:
                                    bgmean_cols.append('DW1O_BGDev')
                                if 'DW2O_BGDev' in casi_df.columns:
                                    bgmean_cols.append('DW2O_BGDev')
                                if 'DN1O_BGDev' in casi_df.columns:
                                    bgmean_cols.append('DN1O_BGDev')
                                has_bgmean_cols = len(bgmean_cols) > 0
                                
                                # 检查是否有TotalSNR列（用于SNR分析）
                                totalsnr_cols = []
                                if 'DW1O_TotalSNR' in casi_df.columns:
                                    totalsnr_cols.append('DW1O_TotalSNR')
                                if 'DW2O_TotalSNR' in casi_df.columns:
                                    totalsnr_cols.append('DW2O_TotalSNR')
                                if 'DN1O_TotalSNR' in casi_df.columns:
                                    totalsnr_cols.append('DN1O_TotalSNR')
                                has_totalsnr_cols = len(totalsnr_cols) > 0
                                
                                # 准备CASI工作数据，包含is_special_type列和尺寸列
                                cols_to_read = [cas_x_col, cas_y_col, 'is_special_type', 'is_edge_point']
                                if has_maxorg:
                                    cols_to_read += maxorg_cols
                                if has_size_cols:
                                    cols_to_read += size_cols
                                if has_dw1o_channels:
                                    cols_to_read += dw1o_channel_cols
                                if has_bgmean_cols:
                                    cols_to_read += bgmean_cols
                                if has_totalsnr_cols:
                                    cols_to_read += totalsnr_cols
                                
                                casi_work = casi_df[cols_to_read].copy()
                                # 重命名坐标列
                                casi_work.rename(columns={cas_x_col: 'XREL', cas_y_col: 'YREL'}, inplace=True)
                                
                                # 转换MaxOrg列为数值
                                if has_maxorg:
                                    for col in maxorg_cols:
                                        casi_work[col] = pd.to_numeric(casi_work[col], errors='coerce')
                                
                                # 转换Size列为数值
                                if has_size_cols:
                                    for col in size_cols:
                                        casi_work[col] = pd.to_numeric(casi_work[col], errors='coerce')
                                
                                # 转换DW1O通道列为数值
                                if has_dw1o_channels:
                                    for col in dw1o_channel_cols:
                                        casi_work[col] = pd.to_numeric(casi_work[col], errors='coerce')
                                
                                # 转换BGMean列为数值
                                if has_bgmean_cols:
                                    for col in bgmean_cols:
                                        casi_work[col] = pd.to_numeric(casi_work[col], errors='coerce')
                                
                                # 转换TotalSNR列为数值
                                if has_totalsnr_cols:
                                    for col in totalsnr_cols:
                                        casi_work[col] = pd.to_numeric(casi_work[col], errors='coerce')
                                
                                casi_work = casi_work.dropna(subset=['XREL', 'YREL']).reset_index(drop=True)
                                
                                # 确保is_special_type列存在
                                if 'is_special_type' not in casi_work.columns:
                                    casi_work['is_special_type'] = False
                                
                                # 确保is_edge_point列存在
                                if 'is_edge_point' not in casi_work.columns:
                                    casi_work['is_edge_point'] = False
                                
                                # 读取KLA数据，包含DSIZE列用于尺寸统计
                                if 'DSIZE' in kla_df.columns:
                                    kla_work = kla_df[['XREL', 'YREL', 'DSIZE']].copy()
                                    kla_work['DSIZE'] = pd.to_numeric(kla_work['DSIZE'], errors='coerce')
                                else:
                                    kla_work = kla_df[['XREL', 'YREL']].copy()
                                    kla_work['DSIZE'] = np.nan
                                kla_work = kla_work.dropna(subset=['XREL', 'YREL']).reset_index(drop=True)
                                
                                # 初始化匹配结果列
                                casi_match_result = np.full(len(casi_work), np.nan)
                                kla_matched = np.zeros(len(kla_work), dtype=bool)  # KLA是否被非特殊CASI匹配
                                kla_miss_type = np.zeros(len(kla_work), dtype=int)  # 0=正确检出, 1=基础漏检, 2=分类漏检
                                
                                # 辅助函数：判断是否为特殊类型
                                def _cas_is_special(idx: int) -> bool:
                                    if idx >= len(casi_work):
                                        return False
                                    return bool(casi_work.at[idx, 'is_special_type'])
                                
                                # 构建两个KDTree：一个包含所有CASI，一个只包含非特殊CASI
                                if len(casi_work) > 0 and len(kla_work) > 0:
                                    # 所有CASI的坐标（用于判断基础漏检 vs 分类漏检）
                                    casi_pts_all = casi_work[['XREL', 'YREL']].to_numpy()
                                    tree_casi_all = KDTree(casi_pts_all)
                                    
                                    # 只包含非特殊类型的CASI（用于正常匹配）
                                    non_special_mask = ~casi_work['is_special_type'].values
                                    non_special_indices = np.where(non_special_mask)[0]
                                    
                                    kla_pts = kla_work[['XREL', 'YREL']].to_numpy()
                                    tree_kla = KDTree(kla_pts)
                                    
                                    # 先标记特殊类型的CASI为-2
                                    for casi_idx in range(len(casi_work)):
                                        if _cas_is_special(casi_idx):
                                            casi_match_result[casi_idx] = -2
                                    
                                    if len(non_special_indices) > 0:
                                        casi_pts_non_special = casi_work.loc[non_special_indices, ['XREL', 'YREL']].to_numpy()
                                        tree_casi_non_special = KDTree(casi_pts_non_special)
                                        
                                        # ===== 第一步：遍历KLA，判断漏检类型 =====
                                        for kla_idx in range(len(kla_pts)):
                                            kla_pt = kla_pts[kla_idx]
                                            
                                            # 在非特殊CASI中查找匹配
                                            casi_non_special_indices_in_tree = tree_casi_non_special.query_ball_point(kla_pt, r=kla_match_threshold)
                                            
                                            if len(casi_non_special_indices_in_tree) == 0:
                                                # KLA附近没有非特殊CASI
                                                kla_matched[kla_idx] = False
                                                
                                                # 进一步判断：附近是否有特殊类型的CASI
                                                casi_all_indices = tree_casi_all.query_ball_point(kla_pt, r=kla_match_threshold)
                                                
                                                if len(casi_all_indices) == 0:
                                                    # 附近完全没有CASI -> 基础漏检
                                                    kla_miss_type[kla_idx] = 1
                                                else:
                                                    # 附近有CASI，但都是特殊类型 -> 分类漏检
                                                    kla_miss_type[kla_idx] = 2
                                                continue
                                            
                                            # 有非特殊CASI匹配 -> 正确检出
                                            kla_matched[kla_idx] = True
                                            kla_miss_type[kla_idx] = 0
                                            
                                            # 映射回原始索引
                                            casi_idx_list = [non_special_indices[i] for i in casi_non_special_indices_in_tree]
                                            
                                            if len(casi_idx_list) == 1:
                                                # 一对一匹配
                                                ci = casi_idx_list[0]
                                                casi_match_result[ci] = 1
                                            else:
                                                # 多CASI对一KLA
                                                for ci in casi_idx_list:
                                                    casi_match_result[ci] = 3
                                        
                                        # ===== 第二步：遍历非特殊CASI，识别过检 =====
                                        for tree_idx, casi_idx in enumerate(non_special_indices):
                                            casi_pt = casi_pts_non_special[tree_idx]
                                            kla_idx_list = tree_kla.query_ball_point(casi_pt, r=kla_match_threshold)
                                            
                                            cur = casi_match_result[casi_idx]
                                            
                                            if len(kla_idx_list) == 0:
                                                # CASI附近没有KLA -> 过检
                                                casi_match_result[casi_idx] = 0
                                                continue
                                            
                                            # 细化1->4, 3->5（一CASI对多KLA）
                                            if pd.notna(cur):
                                                cur_int = int(cur)
                                                if cur_int == 1 and len(kla_idx_list) > 1:
                                                    casi_match_result[casi_idx] = 4
                                                elif cur_int == 3 and len(kla_idx_list) > 1:
                                                    casi_match_result[casi_idx] = 5
                                            elif len(kla_idx_list) > 1:
                                                casi_match_result[casi_idx] = 4
                                        
                                        # 处理未匹配的非特殊CASI -> 过检
                                        for casi_idx in non_special_indices:
                                            if np.isnan(casi_match_result[casi_idx]):
                                                casi_match_result[casi_idx] = 0
                                
                                # 统计结果
                                n_overdetect_true = np.sum(casi_match_result == 0)  # 真过检（CASI附近真的没有KLA）
                                n_correct_casi = np.sum(np.isin(casi_match_result, [1, 3, 4, 5]))  # CASI侧的正确检出
                                n_miss_basic = np.sum(kla_miss_type == 1)  # 基础漏检
                                n_miss_classified = np.sum(kla_miss_type == 2)  # 分类漏检
                                n_miss = n_miss_basic + n_miss_classified  # 总漏检
                                
                                # 统计DSIZE尺寸信息（正确检出和漏检的缺陷）
                                dsize_correct_list = []
                                dsize_miss_list = []
                                
                                # 定义尺寸区间（nm）- DSIZE需要乘以1000
                                size_bins = list(range(26, 101))  # 26nm到100nm，每1nm一个区间
                                size_bin_labels = [f"{i}nm" for i in size_bins]
                                
                                # 初始化按尺寸区间的统计字典
                                size_stats = {
                                    'bins': size_bins,
                                    'correct_count': {i: 0 for i in size_bins},
                                    'miss_count': {i: 0 for i in size_bins},
                                    'total_count': {i: 0 for i in size_bins}
                                }
                                
                                if 'DSIZE' in kla_work.columns and len(kla_work) > 0:
                                    # 统计正确检出和漏检的DSIZE
                                    for kla_idx in range(len(kla_work)):
                                        dsize_val = kla_work.loc[kla_idx, 'DSIZE']
                                        if pd.notna(dsize_val):
                                            dsize_nm = dsize_val * 1000  # 转换为nm
                                            
                                            # 找到对应的尺寸区间
                                            size_bin = int(round(dsize_nm))
                                            
                                            if 26 <= size_bin <= 100:
                                                size_stats['total_count'][size_bin] += 1
                                                
                                                # 判断是正确检出还是漏检
                                                if kla_matched[kla_idx]:
                                                    # 正确检出（KLA被匹配到）
                                                    dsize_correct_list.append(dsize_val)
                                                    size_stats['correct_count'][size_bin] += 1
                                                else:
                                                    # 漏检（KLA未被匹配）
                                                    dsize_miss_list.append(dsize_val)
                                                    size_stats['miss_count'][size_bin] += 1
                                
                                # 计算DSIZE统计值
                                dsize_correct_avg = np.mean(dsize_correct_list) if len(dsize_correct_list) > 0 else 0
                                dsize_correct_min = np.min(dsize_correct_list) if len(dsize_correct_list) > 0 else 0
                                dsize_correct_max = np.max(dsize_correct_list) if len(dsize_correct_list) > 0 else 0
                                
                                dsize_miss_avg = np.mean(dsize_miss_list) if len(dsize_miss_list) > 0 else 0
                                dsize_miss_min = np.min(dsize_miss_list) if len(dsize_miss_list) > 0 else 0
                                dsize_miss_max = np.max(dsize_miss_list) if len(dsize_miss_list) > 0 else 0
                                
                                # 统计过检中污染的数量（DW1O_MaxOrg或DW2O_MaxOrg或DN1O_MaxOrg == 65532）
                                n_contamination = 0
                                if has_maxorg:
                                    overdetect_indices = np.where(casi_match_result == 0)[0]  # 真过检的索引
                                    for idx in overdetect_indices:
                                        is_contamination = False
                                        # 检查所有可用的MaxOrg列
                                        for maxorg_col in maxorg_cols:
                                            if maxorg_col in casi_work.columns:
                                                maxorg_val = casi_work.loc[idx, maxorg_col]
                                                if pd.notna(maxorg_val) and maxorg_val == 65532:
                                                    is_contamination = True
                                                    break  # 只要有一个为65532就算污染
                                        if is_contamination:
                                            n_contamination += 1
                                
                                # 计算去除污染后的真过检数量
                                n_overdetect_true_clean = n_overdetect_true - n_contamination
                                
                                # 统计过检数据的DW1O_Size和DW2O_Size尺寸分布
                                overdetect_size_stats = {
                                    'has_size_data': has_size_cols,
                                    'dw1o_size': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'count_200000': 0},
                                    'dw2o_size': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'count_200000': 0}
                                }
                                
                                if has_size_cols and n_overdetect_true > 0:
                                    # 获取真过检数据的索引
                                    overdetect_indices = np.where(casi_match_result == 0)[0]
                                    
                                    # 统计DW1O_Size
                                    if 'DW1O_Size' in casi_work.columns:
                                        dw1o_values = []
                                        dw1o_count_200000 = 0
                                        for idx in overdetect_indices:
                                            val = casi_work.loc[idx, 'DW1O_Size']
                                            if pd.notna(val) and val > 0:  # 排除无效值和0
                                                if val == 200000.00:  # 单独统计200000的
                                                    dw1o_count_200000 += 1
                                                else:  # 其他值正常统计
                                                    dw1o_values.append(val)
                                        
                                        overdetect_size_stats['dw1o_size']['count_200000'] = dw1o_count_200000
                                        if len(dw1o_values) > 0:
                                            overdetect_size_stats['dw1o_size']['values'] = dw1o_values
                                            overdetect_size_stats['dw1o_size']['mean'] = np.mean(dw1o_values)
                                            overdetect_size_stats['dw1o_size']['min'] = np.min(dw1o_values)
                                            overdetect_size_stats['dw1o_size']['max'] = np.max(dw1o_values)
                                            overdetect_size_stats['dw1o_size']['std'] = np.std(dw1o_values)
                                    
                                    # 统计DW2O_Size
                                    if 'DW2O_Size' in casi_work.columns:
                                        dw2o_values = []
                                        dw2o_count_200000 = 0
                                        for idx in overdetect_indices:
                                            val = casi_work.loc[idx, 'DW2O_Size']
                                            if pd.notna(val) and val > 0:  # 排除无效值和0
                                                if val == 200000.00:  # 单独统计200000的
                                                    dw2o_count_200000 += 1
                                                else:  # 其他值正常统计
                                                    dw2o_values.append(val)
                                        
                                        overdetect_size_stats['dw2o_size']['count_200000'] = dw2o_count_200000
                                        if len(dw2o_values) > 0:
                                            overdetect_size_stats['dw2o_size']['values'] = dw2o_values
                                            overdetect_size_stats['dw2o_size']['mean'] = np.mean(dw2o_values)
                                            overdetect_size_stats['dw2o_size']['min'] = np.min(dw2o_values)
                                            overdetect_size_stats['dw2o_size']['max'] = np.max(dw2o_values)
                                            overdetect_size_stats['dw2o_size']['std'] = np.std(dw2o_values)
                                
                                total_casi = len(casi_work)
                                total_kla = len(kla_work)
                                
                                # 新增：统计边缘点数量（距离>=147mm的过检点）
                                # 找出所有过检点（match_result == 0）中的边缘点
                                overdetect_edge_count = 0
                                if 'is_edge_point' in casi_work.columns:
                                    overdetect_indices = np.where(casi_match_result == 0)[0]
                                    for idx in overdetect_indices:
                                        if casi_work.loc[idx, 'is_edge_point']:
                                            overdetect_edge_count += 1
                                
                                # 计算CASI分类后检出数（不包含1000和10001的特殊类型，也不包含距离>=147的边缘过检点）
                                # 原始分类后检出数
                                casi_detected_count_raw = np.sum(~casi_work['is_special_type']) if 'is_special_type' in casi_work.columns else total_casi
                                # 去除边缘过检点后的分类后检出数
                                casi_detected_count = casi_detected_count_raw - overdetect_edge_count
                                
                                # **正确的统计逻辑**：
                                # 1. 正确检出 = KLA总数 - 漏检总数
                                # 2. 过检(0) = CASI分类后检出数 - 正确检出（总过检）
                                # 3. 真过检 = CASI附近真的没有KLA的（casi_match_result == 0）
                                # 4. 去除污染过检 = 真过检 - 污染数量
                                # 5. 去除边缘点后的真过检 = 真过检 - 边缘过检点数量
                                # 6. 漏检分为：基础漏检 和 分类漏检
                                # 验证：CASI分类后检出数 = 过检(0) + 正确检出
                                
                                n_correct = total_kla - n_miss  # 正确检出数 = KLA总数 - 漏检总数
                                n_overdetect = casi_detected_count - n_correct  # 过检(0) = CASI分类后检出数 - 正确检出
                                # 真过检需要减去边缘点
                                n_overdetect_true_filtered = n_overdetect_true - overdetect_edge_count
                                n_overdetect_clean = n_overdetect_true_clean - overdetect_edge_count  # 去除污染和边缘点的过检
                                n_miss_total = n_miss  # 漏检总数
                                n_miss_from_special = n_miss_classified  # 分类漏检（旧字段名保持兼容）
                                
                                # 验证：CASI分类后检出数应该等于过检(0)+正确检出
                                expected_casi = n_overdetect + n_correct
                                if abs(expected_casi - casi_detected_count) > 1:
                                    st.warning(f"⚠️ 验证失败：CASI分类后检出数({casi_detected_count}) ≠ 过检({n_overdetect}) + 正检({n_correct}) = {expected_casi}")
                                
                                # 新增：统计DW1O_MaxOrg和DW2O_MaxOrg比值分布（去除0值）
                                maxorg_ratio_stats = {
                                    'has_maxorg_data': False,
                                    '过检': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                    '正确检出': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0}
                                }
                                
                                # 检查是否有DW1O_MaxOrg和DW2O_MaxOrg列
                                has_dw1o_maxorg = 'DW1O_MaxOrg' in casi_work.columns
                                has_dw2o_maxorg = 'DW2O_MaxOrg' in casi_work.columns
                                
                                if has_dw1o_maxorg and has_dw2o_maxorg:
                                    maxorg_ratio_stats['has_maxorg_data'] = True
                                    
                                    # 计算每种类型的MaxOrg比值
                                    for idx in range(len(casi_work)):
                                        dw1o_val = casi_work.loc[idx, 'DW1O_MaxOrg']
                                        dw2o_val = casi_work.loc[idx, 'DW2O_MaxOrg']
                                        result = casi_match_result[idx]
                                        
                                        # 跳过0值和无效值
                                        if pd.notna(dw1o_val) and pd.notna(dw2o_val) and dw1o_val != 0 and dw2o_val != 0:
                                            ratio = dw1o_val / dw2o_val
                                            
                                            # 根据匹配结果分类（只统计过检和正确检出）
                                            if result == 0:
                                                # 过检：排除边缘点
                                                is_edge = casi_work.loc[idx, 'is_edge_point'] if 'is_edge_point' in casi_work.columns else False
                                                if not is_edge:
                                                    maxorg_ratio_stats['过检']['ratios'].append(ratio)
                                            elif result in [1, 3, 4, 5]:
                                                maxorg_ratio_stats['正确检出']['ratios'].append(ratio)
                                    
                                    # 计算统计值
                                    for defect_type in ['过检', '正确检出']:
                                        ratios = maxorg_ratio_stats[defect_type]['ratios']
                                        if len(ratios) > 0:
                                            maxorg_ratio_stats[defect_type]['mean'] = np.mean(ratios)
                                            maxorg_ratio_stats[defect_type]['min'] = np.min(ratios)
                                            maxorg_ratio_stats[defect_type]['max'] = np.max(ratios)
                                            maxorg_ratio_stats[defect_type]['std'] = np.std(ratios)
                                            maxorg_ratio_stats[defect_type]['median'] = np.median(ratios)
                                
                                # 新增：统计过检和正确检出中MaxOrg=65532的情况
                                maxorg_65532_stats = {
                                    'has_maxorg_cols': False,
                                    '过检': {
                                        '总数': 0,
                                        '三个都是65532': 0,
                                        'DW1O和DW2O是65532但DN1O不是': 0,
                                        'DW1O是65532但DW2O和DN1O不是': 0,
                                        'DW2O是65532但DW1O和DN1O不是': 0,
                                        'DN1O是65532但DW1O和DW2O不是': 0,
                                        'DW1O和DN1O是65532但DW2O不是': 0,
                                        'DW2O和DN1O是65532但DW1O不是': 0,
                                        '都不是65532': 0
                                    },
                                    '正确检出': {
                                        '总数': 0,
                                        '三个都是65532': 0,
                                        'DW1O和DW2O是65532但DN1O不是': 0,
                                        'DW1O是65532但DW2O和DN1O不是': 0,
                                        'DW2O是65532但DW1O和DN1O不是': 0,
                                        'DN1O是65532但DW1O和DW2O不是': 0,
                                        'DW1O和DN1O是65532但DW2O不是': 0,
                                        'DW2O和DN1O是65532但DW1O不是': 0,
                                        '都不是65532': 0
                                    }
                                }
                                
                                # 检查是否有三个MaxOrg列
                                has_dw1o_maxorg = 'DW1O_MaxOrg' in casi_work.columns
                                has_dw2o_maxorg = 'DW2O_MaxOrg' in casi_work.columns
                                has_dn1o_maxorg = 'DN1O_MaxOrg' in casi_work.columns
                                
                                if has_dw1o_maxorg and has_dw2o_maxorg and has_dn1o_maxorg:
                                    maxorg_65532_stats['has_maxorg_cols'] = True
                                    
                                    # 分析过检和正确检出的缺陷
                                    for idx in range(len(casi_work)):
                                        result = casi_match_result[idx]
                                        
                                        # 只分析过检(0)和正确检出(1,3,4,5)
                                        if result == 0:
                                            # 过检：排除边缘点
                                            is_edge = casi_work.loc[idx, 'is_edge_point'] if 'is_edge_point' in casi_work.columns else False
                                            if is_edge:
                                                continue
                                            defect_type = '过检'
                                        elif result in [1, 3, 4, 5]:
                                            defect_type = '正确检出'
                                        else:
                                            continue
                                        
                                        maxorg_65532_stats[defect_type]['总数'] += 1
                                        
                                        # 获取三个MaxOrg值
                                        dw1o_maxorg = casi_work.loc[idx, 'DW1O_MaxOrg']
                                        dw2o_maxorg = casi_work.loc[idx, 'DW2O_MaxOrg']
                                        dn1o_maxorg = casi_work.loc[idx, 'DN1O_MaxOrg']
                                        
                                        # 判断是否为65532
                                        is_dw1o_65532 = (pd.notna(dw1o_maxorg) and dw1o_maxorg == 65532)
                                        is_dw2o_65532 = (pd.notna(dw2o_maxorg) and dw2o_maxorg == 65532)
                                        is_dn1o_65532 = (pd.notna(dn1o_maxorg) and dn1o_maxorg == 65532)
                                        
                                        # 统计各种情况
                                        if is_dw1o_65532 and is_dw2o_65532 and is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['三个都是65532'] += 1
                                        elif is_dw1o_65532 and is_dw2o_65532 and not is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['DW1O和DW2O是65532但DN1O不是'] += 1
                                        elif is_dw1o_65532 and not is_dw2o_65532 and is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['DW1O和DN1O是65532但DW2O不是'] += 1
                                        elif not is_dw1o_65532 and is_dw2o_65532 and is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['DW2O和DN1O是65532但DW1O不是'] += 1
                                        elif is_dw1o_65532 and not is_dw2o_65532 and not is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['DW1O是65532但DW2O和DN1O不是'] += 1
                                        elif not is_dw1o_65532 and is_dw2o_65532 and not is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['DW2O是65532但DW1O和DN1O不是'] += 1
                                        elif not is_dw1o_65532 and not is_dw2o_65532 and is_dn1o_65532:
                                            maxorg_65532_stats[defect_type]['DN1O是65532但DW1O和DW2O不是'] += 1
                                        else:  # 都不是65532
                                            maxorg_65532_stats[defect_type]['都不是65532'] += 1
                                
                                # 新增：统计DW1O通道的三个比值分布（去除0值）
                                dw1o_ratio_stats = {
                                    'has_dw1o_data': False,
                                    '过检': {
                                        'SubRow1/SubRow2': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'MainRow/SubRow1': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'MainRow/SubRow2': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0}
                                    },
                                    '正确检出': {
                                        'SubRow1/SubRow2': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'MainRow/SubRow1': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'MainRow/SubRow2': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0}
                                    },
                                    'KLA检出': {
                                        'SubRow1/SubRow2': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'MainRow/SubRow1': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'MainRow/SubRow2': {'ratios': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0}
                                    }
                                }
                                
                                # 检查是否有DW1O通道的三个列
                                has_subrow1 = 'DW1O_SubRow1Max' in casi_work.columns
                                has_subrow2 = 'DW1O_SubRow2Max' in casi_work.columns
                                has_mainrow = 'DW1O_MainRowMax' in casi_work.columns
                                
                                if has_subrow1 and has_subrow2 and has_mainrow:
                                    dw1o_ratio_stats['has_dw1o_data'] = True
                                    
                                    # 计算每种类型的DW1O比值
                                    for idx in range(len(casi_work)):
                                        subrow1_val = casi_work.loc[idx, 'DW1O_SubRow1Max']
                                        subrow2_val = casi_work.loc[idx, 'DW1O_SubRow2Max']
                                        mainrow_val = casi_work.loc[idx, 'DW1O_MainRowMax']
                                        result = casi_match_result[idx]
                                        
                                        # 跳过0值和无效值
                                        if pd.notna(subrow1_val) and pd.notna(subrow2_val) and pd.notna(mainrow_val):
                                            # 根据匹配结果分类（只统计过检和正确检出）
                                            if result == 0:
                                                defect_type = '过检'
                                            elif result in [1, 3, 4, 5]:
                                                defect_type = '正确检出'
                                            else:
                                                continue
                                            
                                            # 计算三个比值（去除0值）
                                            if subrow1_val != 0 and subrow2_val != 0:
                                                ratio1 = subrow1_val / subrow2_val
                                                dw1o_ratio_stats[defect_type]['SubRow1/SubRow2']['ratios'].append(ratio1)
                                            
                                            if mainrow_val != 0 and subrow1_val != 0:
                                                ratio2 = mainrow_val / subrow1_val
                                                dw1o_ratio_stats[defect_type]['MainRow/SubRow1']['ratios'].append(ratio2)
                                            
                                            if mainrow_val != 0 and subrow2_val != 0:
                                                ratio3 = mainrow_val / subrow2_val
                                                dw1o_ratio_stats[defect_type]['MainRow/SubRow2']['ratios'].append(ratio3)
                                            
                                            # KLA检出 = 正确检出 + 漏检
                                            if result in [1, 3, 4, 5] or result == 2:
                                                if subrow1_val != 0 and subrow2_val != 0:
                                                    dw1o_ratio_stats['KLA检出']['SubRow1/SubRow2']['ratios'].append(subrow1_val / subrow2_val)
                                                if mainrow_val != 0 and subrow1_val != 0:
                                                    dw1o_ratio_stats['KLA检出']['MainRow/SubRow1']['ratios'].append(mainrow_val / subrow1_val)
                                                if mainrow_val != 0 and subrow2_val != 0:
                                                    dw1o_ratio_stats['KLA检出']['MainRow/SubRow2']['ratios'].append(mainrow_val / subrow2_val)
                                    
                                    # 计算统计值
                                    for defect_type in ['过检', '正确检出', 'KLA检出']:
                                        for ratio_name in ['SubRow1/SubRow2', 'MainRow/SubRow1', 'MainRow/SubRow2']:
                                            ratios = dw1o_ratio_stats[defect_type][ratio_name]['ratios']
                                            if len(ratios) > 0:
                                                dw1o_ratio_stats[defect_type][ratio_name]['mean'] = np.mean(ratios)
                                                dw1o_ratio_stats[defect_type][ratio_name]['min'] = np.min(ratios)
                                                dw1o_ratio_stats[defect_type][ratio_name]['max'] = np.max(ratios)
                                                dw1o_ratio_stats[defect_type][ratio_name]['std'] = np.std(ratios)
                                                dw1o_ratio_stats[defect_type][ratio_name]['median'] = np.median(ratios)
                                
                                # 提取每个类型的坐标数据（用于共有率分析）
                                # 对于CASI数据，按匹配结果分类：0=过检，1/3/4/5=正确检出，2=漏检
                                coord_data = {
                                    '过检': [],      # match_result == 0
                                    '正确检出': [],  # match_result in [1, 3, 4, 5]
                                    '漏检': []       # match_result == 2 或 kla_matched == False
                                }
                                
                                # 获取nDefectID列（如果存在）
                                has_ndefectid = 'nDefectID' in casi_df.columns
                                has_ndefecttype = 'nDefectType' in casi_df.columns
                                
                                # 定义需要提取的特征列（三个通道）
                                feature_cols = {
                                    'DW1O': ['DW1O_MaxOrg', 'DW1O_BGMean', 'DW1O_BGDev', 'DW1O_Size', 'DW1O_TotalSNR', 'DW1O_MapSNR'],
                                    'DW2O': ['DW2O_MaxOrg', 'DW2O_BGMean', 'DW2O_BGDev', 'DW2O_Size', 'DW2O_TotalSNR', 'DW2O_MapSNR'],
                                    'DN1O': ['DN1O_MaxOrg', 'DN1O_BGMean', 'DN1O_BGDev', 'DN1O_Size', 'DN1O_TotalSNR', 'DN1O_MapSNR']
                                }
                                
                                # CASI的过检和正确检出坐标（增加nDefectID、nDefectType和特征数据）
                                for idx in range(len(casi_work)):
                                    x = casi_work.loc[idx, 'XREL']
                                    y = casi_work.loc[idx, 'YREL']
                                    result = casi_match_result[idx]
                                    
                                    # 获取nDefectID（如果存在）
                                    defect_id = casi_df.loc[idx, 'nDefectID'] if has_ndefectid and idx < len(casi_df) else None
                                    
                                    # 获取nDefectType（如果存在）
                                    defect_type_value = casi_df.loc[idx, 'nDefectType'] if has_ndefecttype and idx < len(casi_df) else None
                                    
                                    # 提取特征数据
                                    features = {}
                                    for channel, cols in feature_cols.items():
                                        for col in cols:
                                            if col in casi_df.columns and idx < len(casi_df):
                                                features[col] = casi_df.loc[idx, col]
                                            else:
                                                features[col] = None
                                    
                                    # 数据格式：(x, y, defect_id, features_dict, defect_type_value)
                                    data_tuple = (x, y, defect_id, features, defect_type_value)
                                    
                                    if result == 0:
                                        # 过检：排除边缘点（距离>=147的点）
                                        is_edge = casi_work.loc[idx, 'is_edge_point'] if 'is_edge_point' in casi_work.columns else False
                                        if not is_edge:
                                            coord_data['过检'].append(data_tuple)
                                    elif result in [1, 3, 4, 5]:
                                        coord_data['正确检出'].append(data_tuple)
                                    # 注意：result==2的CASI不添加到漏检，因为漏检统计基于KLA
                                
                                # 新增：统计BGMean值分布（过检和正确检出，去除0值）
                                bgmean_stats = {
                                    'has_bgmean_data': False,
                                    '过检': {
                                        'DW1O_BGMean': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DW2O_BGMean': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DN1O_BGMean': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DW1O_BGDev': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DW2O_BGDev': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DN1O_BGDev': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0}
                                    },
                                    '正确检出': {
                                        'DW1O_BGMean': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DW2O_BGMean': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DN1O_BGMean': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DW1O_BGDev': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DW2O_BGDev': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0},
                                        'DN1O_BGDev': {'values': [], 'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'median': 0}
                                    }
                                }
                                
                                # 检查是否有BGMean和BGDev列
                                has_dw1o_bgmean = 'DW1O_BGMean' in casi_work.columns
                                has_dw2o_bgmean = 'DW2O_BGMean' in casi_work.columns
                                has_dn1o_bgmean = 'DN1O_BGMean' in casi_work.columns
                                has_dw1o_bgdev = 'DW1O_BGDev' in casi_work.columns
                                has_dw2o_bgdev = 'DW2O_BGDev' in casi_work.columns
                                has_dn1o_bgdev = 'DN1O_BGDev' in casi_work.columns
                                
                                if has_dw1o_bgmean or has_dw2o_bgmean or has_dn1o_bgmean or has_dw1o_bgdev or has_dw2o_bgdev or has_dn1o_bgdev:
                                    bgmean_stats['has_bgmean_data'] = True
                                    
                                    # 提取过检和正确检出的BGMean和BGDev值（去除0值）
                                    for idx in range(len(casi_work)):
                                        result = casi_match_result[idx]
                                        
                                        # 只统计过检和正确检出
                                        if result == 0:
                                            defect_type = '过检'
                                        elif result in [1, 3, 4, 5]:
                                            defect_type = '正确检出'
                                        else:
                                            continue
                                        
                                        # 收集DW1O_BGMean值
                                        if has_dw1o_bgmean:
                                            dw1o_bgmean_val = casi_work.loc[idx, 'DW1O_BGMean']
                                            if pd.notna(dw1o_bgmean_val) and dw1o_bgmean_val != 0:
                                                bgmean_stats[defect_type]['DW1O_BGMean']['values'].append(dw1o_bgmean_val)
                                        
                                        # 收集DW2O_BGMean值
                                        if has_dw2o_bgmean:
                                            dw2o_bgmean_val = casi_work.loc[idx, 'DW2O_BGMean']
                                            if pd.notna(dw2o_bgmean_val) and dw2o_bgmean_val != 0:
                                                bgmean_stats[defect_type]['DW2O_BGMean']['values'].append(dw2o_bgmean_val)
                                        
                                        # 收集DN1O_BGMean值
                                        if has_dn1o_bgmean:
                                            dn1o_bgmean_val = casi_work.loc[idx, 'DN1O_BGMean']
                                            if pd.notna(dn1o_bgmean_val) and dn1o_bgmean_val != 0:
                                                bgmean_stats[defect_type]['DN1O_BGMean']['values'].append(dn1o_bgmean_val)
                                        
                                        # 收集DW1O_BGDev值
                                        if has_dw1o_bgdev:
                                            dw1o_bgdev_val = casi_work.loc[idx, 'DW1O_BGDev']
                                            if pd.notna(dw1o_bgdev_val) and dw1o_bgdev_val != 0:
                                                bgmean_stats[defect_type]['DW1O_BGDev']['values'].append(dw1o_bgdev_val)
                                        
                                        # 收集DW2O_BGDev值
                                        if has_dw2o_bgdev:
                                            dw2o_bgdev_val = casi_work.loc[idx, 'DW2O_BGDev']
                                            if pd.notna(dw2o_bgdev_val) and dw2o_bgdev_val != 0:
                                                bgmean_stats[defect_type]['DW2O_BGDev']['values'].append(dw2o_bgdev_val)
                                        
                                        # 收集DN1O_BGDev值
                                        if has_dn1o_bgdev:
                                            dn1o_bgdev_val = casi_work.loc[idx, 'DN1O_BGDev']
                                            if pd.notna(dn1o_bgdev_val) and dn1o_bgdev_val != 0:
                                                bgmean_stats[defect_type]['DN1O_BGDev']['values'].append(dn1o_bgdev_val)
                                    
                                    # 计算统计值
                                    for defect_type in ['过检', '正确检出']:
                                        for bg_name in ['DW1O_BGMean', 'DW2O_BGMean', 'DN1O_BGMean', 'DW1O_BGDev', 'DW2O_BGDev', 'DN1O_BGDev']:
                                            values = bgmean_stats[defect_type][bg_name]['values']
                                            if len(values) > 0:
                                                bgmean_stats[defect_type][bg_name]['mean'] = np.mean(values)
                                                bgmean_stats[defect_type][bg_name]['min'] = np.min(values)
                                                bgmean_stats[defect_type][bg_name]['max'] = np.max(values)
                                                bgmean_stats[defect_type][bg_name]['std'] = np.std(values)
                                                bgmean_stats[defect_type][bg_name]['median'] = np.median(values)
                                
                                # 新增：统计TotalSNR值按尺寸分布（过检和正确检出，每2nm一个区间，从26nm开始）
                                totalsnr_size_stats = {
                                    'has_snr_data': False,
                                    'size_bins': [],  # 尺寸区间列表，如 [26, 28, 30, ...]
                                    '过检': {},  # 每个尺寸区间的SNR值字典
                                    '正确检出': {}  # 每个尺寸区间的SNR值字典
                                }
                                
                                # 检查是否同时有Size和TotalSNR列
                                has_dw1o_size = 'DW1O_Size' in casi_work.columns
                                has_dw2o_size = 'DW2O_Size' in casi_work.columns
                                has_dn1o_size = 'DN1O_Size' in casi_work.columns
                                has_dw1o_snr = 'DW1O_TotalSNR' in casi_work.columns
                                has_dw2o_snr = 'DW2O_TotalSNR' in casi_work.columns
                                has_dn1o_snr = 'DN1O_TotalSNR' in casi_work.columns
                                
                                # 需要至少有一组Size和SNR列
                                if (has_dw1o_size and has_dw1o_snr) or (has_dw2o_size and has_dw2o_snr) or (has_dn1o_size and has_dn1o_snr):
                                    totalsnr_size_stats['has_snr_data'] = True
                                    
                                    # 定义尺寸区间：从26开始，每2nm一个区间
                                    size_bins = list(range(26, 201, 2))  # 26, 28, 30, ..., 200
                                    totalsnr_size_stats['size_bins'] = size_bins
                                    
                                    # 初始化每个尺寸区间的数据字典
                                    for size_bin in size_bins:
                                        for defect_type in ['过检', '正确检出']:
                                            if defect_type not in totalsnr_size_stats:
                                                totalsnr_size_stats[defect_type] = {}
                                            totalsnr_size_stats[defect_type][size_bin] = {
                                                'count': 0,
                                                'coords': [],  # (x, y, dw1o_size, dw2o_size, dn1o_size)
                                                'DW1O_TotalSNR': [],
                                                'DW2O_TotalSNR': [],
                                                'DN1O_TotalSNR': []
                                            }
                                    
                                    # 收集每个缺陷的数据
                                    for idx in range(len(casi_work)):
                                        result = casi_match_result[idx]
                                        
                                        # 只统计过检和正确检出
                                        if result == 0:
                                            defect_type = '过检'
                                        elif result in [1, 3, 4, 5]:
                                            defect_type = '正确检出'
                                        else:
                                            continue
                                        
                                        # 获取三个通道的尺寸值（用于决定归入哪个区间）
                                        dw1o_size = casi_work.loc[idx, 'DW1O_Size'] if has_dw1o_size else np.nan
                                        dw2o_size = casi_work.loc[idx, 'DW2O_Size'] if has_dw2o_size else np.nan
                                        dn1o_size = casi_work.loc[idx, 'DN1O_Size'] if has_dn1o_size else np.nan
                                        
                                        # 获取坐标
                                        x_coord = casi_work.loc[idx, 'XREL']
                                        y_coord = casi_work.loc[idx, 'YREL']
                                        
                                        # 使用DW1O_Size作为主要尺寸判断标准（如果没有则用DW2O或DN1O）
                                        primary_size = dw1o_size if pd.notna(dw1o_size) else (dw2o_size if pd.notna(dw2o_size) else dn1o_size)
                                        
                                        if pd.notna(primary_size) and primary_size < 200000:  # 排除200000的异常值
                                            # 找到对应的尺寸区间（向下取整到最近的偶数）
                                            size_bin = int(primary_size // 2) * 2
                                            
                                            # 确保在统计范围内
                                            if size_bin in size_bins:
                                                # 获取SNR值
                                                dw1o_snr_val = casi_work.loc[idx, 'DW1O_TotalSNR'] if has_dw1o_snr else np.nan
                                                dw2o_snr_val = casi_work.loc[idx, 'DW2O_TotalSNR'] if has_dw2o_snr else np.nan
                                                dn1o_snr_val = casi_work.loc[idx, 'DN1O_TotalSNR'] if has_dn1o_snr else np.nan
                                                
                                                # 收集SNR值
                                                if has_dw1o_snr:
                                                    if pd.notna(dw1o_snr_val):
                                                        totalsnr_size_stats[defect_type][size_bin]['DW1O_TotalSNR'].append(dw1o_snr_val)
                                                
                                                if has_dw2o_snr:
                                                    if pd.notna(dw2o_snr_val):
                                                        totalsnr_size_stats[defect_type][size_bin]['DW2O_TotalSNR'].append(dw2o_snr_val)
                                                
                                                if has_dn1o_snr:
                                                    if pd.notna(dn1o_snr_val):
                                                        totalsnr_size_stats[defect_type][size_bin]['DN1O_TotalSNR'].append(dn1o_snr_val)
                                                
                                                # 记录坐标、尺寸和SNR信息
                                                totalsnr_size_stats[defect_type][size_bin]['coords'].append({
                                                    'x': x_coord,
                                                    'y': y_coord,
                                                    'dw1o_size': dw1o_size if pd.notna(dw1o_size) else 0,
                                                    'dw2o_size': dw2o_size if pd.notna(dw2o_size) else 0,
                                                    'dn1o_size': dn1o_size if pd.notna(dn1o_size) else 0,
                                                    'dw1o_snr': dw1o_snr_val if pd.notna(dw1o_snr_val) else 0,
                                                    'dw2o_snr': dw2o_snr_val if pd.notna(dw2o_snr_val) else 0,
                                                    'dn1o_snr': dn1o_snr_val if pd.notna(dn1o_snr_val) else 0
                                                })
                                                totalsnr_size_stats[defect_type][size_bin]['count'] += 1
                                
                                # KLA的漏检坐标
                                # 注意：KLA数据没有nDefectID和特征数据，用None表示
                                for idx in range(len(kla_work)):
                                    # 漏检：kla_matched == False（KLA附近没有非特殊类型的CASI）
                                    if not kla_matched[idx]:
                                        x = kla_work.loc[idx, 'XREL']
                                        y = kla_work.loc[idx, 'YREL']
                                        # KLA数据格式保持一致：(x, y, None, {})
                                        coord_data['漏检'].append((x, y, None, {}))
                                
                                all_match_results.append({
                                    'CASI文件夹': casi_folder,
                                    'KLA文件夹': kla_folder,
                                    # '基础检出个数': int(blob_count),
                                    'CASI总数': total_casi,
                                    'CASI分类后检出数': int(casi_detected_count),  # 不包含1000和10001，也不包含距离>=147的边缘过检点
                                    'KLA总数': total_kla,
                                    '过检(0)': int(n_overdetect),  # CASI分类后检出数 - 正确检出
                                    '真过检': int(n_overdetect_true_filtered),  # CASI附近真的没有KLA的，去除边缘点
                                    '过检（去除污染）': int(n_overdetect_clean),  # 真过检 - 污染 - 边缘点
                                    '过检-边缘点数': int(overdetect_edge_count),  # 距离>=147的过检点
                                    '正确检出(1,3,4,5)': int(n_correct),
                                    '漏检-基础检': int(n_miss_basic),  # KLA附近完全没有CASI
                                    '漏检-分类': int(n_miss_classified),  # KLA附近有CASI但都是1000/10001
                                    '漏检总数': int(n_miss_total),
                                    # '多对一-特殊(-3)': int(n_multi_special),
                                    '过检率': f"{n_overdetect/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '真过检率': f"{n_overdetect_true_filtered/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '过检率（去除污染）': f"{n_overdetect_clean/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '检出率': f"{n_correct/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '漏检率-基础': f"{n_miss_basic/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '漏检率-分类': f"{n_miss_classified/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '漏检率（总）': f"{n_miss_total/total_kla*100:.2f}%" if total_kla > 0 else "0%",
                                    '正确检出DSIZE均值': f"{dsize_correct_avg:.6f}",
                                    '正确检出DSIZE最小': f"{dsize_correct_min:.6f}",
                                    '正确检出DSIZE最大': f"{dsize_correct_max:.6f}",
                                    '漏检DSIZE均值': f"{dsize_miss_avg:.6f}",
                                    '漏检DSIZE最小': f"{dsize_miss_min:.6f}",
                                    '漏检DSIZE最大': f"{dsize_miss_max:.6f}",
                                    '过检DW1O_Size均值': f"{overdetect_size_stats['dw1o_size']['mean']:.2f}" if overdetect_size_stats['dw1o_size']['mean'] > 0 else "N/A",
                                    '过检DW1O_Size最小': f"{overdetect_size_stats['dw1o_size']['min']:.2f}" if overdetect_size_stats['dw1o_size']['min'] > 0 else "N/A",
                                    '过检DW1O_Size最大': f"{overdetect_size_stats['dw1o_size']['max']:.2f}" if overdetect_size_stats['dw1o_size']['max'] > 0 else "N/A",
                                    '过检DW2O_Size均值': f"{overdetect_size_stats['dw2o_size']['mean']:.2f}" if overdetect_size_stats['dw2o_size']['mean'] > 0 else "N/A",
                                    '过检DW2O_Size最小': f"{overdetect_size_stats['dw2o_size']['min']:.2f}" if overdetect_size_stats['dw2o_size']['min'] > 0 else "N/A",
                                    '过检DW2O_Size最大': f"{overdetect_size_stats['dw2o_size']['max']:.2f}" if overdetect_size_stats['dw2o_size']['max'] > 0 else "N/A",
                                    'DW1O_BGMean': f"{bgmean_stats['过检']['DW1O_BGMean']['mean']:.2f}" if len(bgmean_stats['过检']['DW1O_BGMean']['values']) > 0 else "N/A",
                                    'DW1O_BGDev': f"{bgmean_stats['过检']['DW1O_BGDev']['mean']:.2f}" if len(bgmean_stats['过检']['DW1O_BGDev']['values']) > 0 else "N/A",
                                    'DW2O_BGMean': f"{bgmean_stats['过检']['DW2O_BGMean']['mean']:.2f}" if len(bgmean_stats['过检']['DW2O_BGMean']['values']) > 0 else "N/A",
                                    'DW2O_BGDev': f"{bgmean_stats['过检']['DW2O_BGDev']['mean']:.2f}" if len(bgmean_stats['过检']['DW2O_BGDev']['values']) > 0 else "N/A",
                                    'DN1O_BGMean': f"{bgmean_stats['过检']['DN1O_BGMean']['mean']:.2f}" if len(bgmean_stats['过检']['DN1O_BGMean']['values']) > 0 else "N/A",
                                    'DN1O_BGDev': f"{bgmean_stats['过检']['DN1O_BGDev']['mean']:.2f}" if len(bgmean_stats['过检']['DN1O_BGDev']['values']) > 0 else "N/A",
                                    'size_stats': size_stats,  # 保存尺寸区间统计信息
                                    'overdetect_size_stats': overdetect_size_stats,  # 保存过检尺寸统计信息
                                    'maxorg_ratio_stats': maxorg_ratio_stats,  # 保存MaxOrg比值统计信息
                                    'maxorg_65532_stats': maxorg_65532_stats,  # 保存MaxOrg=65532统计信息
                                    'dw1o_ratio_stats': dw1o_ratio_stats,  # 保存DW1O通道比值统计信息
                                    'bgmean_stats': bgmean_stats,  # 保存BGMean值统计信息
                                    'totalsnr_size_stats': totalsnr_size_stats,  # 保存TotalSNR按尺寸分布统计信息
                                    'coord_data': coord_data  # 保存每种类型的坐标数据，用于共有率分析
                                })
                                
                                st.success(f"✓ {casi_folder} vs {kla_folder}: 过检(0)={n_overdetect}(真过检={n_overdetect_true}, 去污染={n_overdetect_clean}), "
                                          f"正确={n_correct}, 漏检={n_miss_total}(基础={n_miss_basic}, 分类={n_miss_classified})")
                    
                    if all_match_results:
                        # 保存到 session_state 供共有率分析使用
                        st.session_state.kla_match_results = all_match_results
                        
                        # 显示结果表格
                        st.markdown('<a name="过漏检统计"></a>', unsafe_allow_html=True)
                        st.subheader("📊 KLA匹配结果汇总（过漏检统计）")
                        
                        results_df = pd.DataFrame(all_match_results)
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        # 提供下载 - 需要清理数据以避免numpy对象和复杂结构
                        # 创建仅用于导出的DataFrame，排除复杂的嵌套数据
                        export_columns = [
                            'CASI文件夹', 'KLA文件夹', 'CASI总数', 'CASI分类后检出数', 'KLA总数',
                            '过检(0)', '真过检', '过检（去除污染）', '正确检出(1,3,4,5)', 
                            '漏检-基础检', '漏检-分类', '漏检总数',
                            '过检率', '真过检率', '过检率（去除污染）', '检出率', 
                            '漏检率-基础', '漏检率-分类', '漏检率（总）',
                            '正确检出DSIZE均值', '正确检出DSIZE最小', '正确检出DSIZE最大',
                            '漏检DSIZE均值', '漏检DSIZE最小', '漏检DSIZE最大',
                            '过检DW1O_Size均值', '过检DW1O_Size最小', '过检DW1O_Size最大',
                            '过检DW2O_Size均值', '过检DW2O_Size最小', '过检DW2O_Size最大',
                            'DW1O_BGMean', 'DW1O_BGDev', 'DW2O_BGMean', 'DW2O_BGDev', 'DN1O_BGMean', 'DN1O_BGDev'
                        ]
                        # 只导出存在的列
                        export_cols_available = [col for col in export_columns if col in results_df.columns]
                        results_df_export = results_df[export_cols_available].copy()
                        
                        # 确保所有数据都是基本类型（字符串或数值）
                        for col in results_df_export.columns:
                            if results_df_export[col].dtype == 'object':
                                results_df_export[col] = results_df_export[col].astype(str)
                        
                        csv_output = results_df_export.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载匹配结果（CSV）",
                            data=csv_output,
                            file_name=f"kla_match_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # 新增：简化版汇总表
                        st.write("---")
                        st.subheader("📊 KLA匹配结果汇总（简化版）")
                        
                        # 添加说明
                        st.info("""
                        **过检分类说明：**
                        - **过检(0)**：CASI分类后检出数 - 正确检出数（总过检）
                        - **真过检**：CASI附近真的没有KLA的缺陷
                        - **去除污染过检**：真过检 - 污染数量（MaxOrg=65532）
                        
                        **漏检分类说明：**
                        - **基础漏检**：KLA附近完全没有CASI检出（任何类型都没有）
                        - **分类漏检**：KLA附近有CASI检出，但都是特殊类型（nDefectType=1000或10001）
                        - **漏检总数** = 基础漏检 + 分类漏检
                        """)
                        
                        # 说明：CASI分类后检出数已经在数据中正确计算（不包含1000和10001）
                        results_df_simplified = results_df.copy()
                        # 验证：过检(0) + 正确检出 = CASI分类后检出数
                        # 验证：正确检出 + 漏检总数 = KLA总数
                        
                        # 只保留指定的列
                        simplified_columns = [
                            'CASI文件夹', 'KLA文件夹', 'CASI总数', 'CASI分类后检出数', 'KLA总数',
                            '过检(0)', '正确检出(1,3,4,5)', 
                             '漏检总数',
                            '过检率', '检出率', '漏检率（总）'
                        ]
                        
                        # 只选择存在的列
                        simplified_cols_available = [col for col in simplified_columns if col in results_df_simplified.columns]
                        results_df_simplified = results_df_simplified[simplified_cols_available].copy()
                        
                        # 显示简化表格
                        st.dataframe(results_df_simplified, use_container_width=True, height=400)
                        
                        # 添加数据验证
                        st.write("**📊 数据验证：**")
                        all_checks_pass = True
                        for idx, row in results_df_simplified.iterrows():
                            casi_detected = row['CASI分类后检出数']
                            overdetect = row['过检(0)']
                            correct = row['正确检出(1,3,4,5)']
                            miss = row['漏检总数']
                            kla_total = row['KLA总数']
                            
                            # 验证1：正确检出 = CASI分类后检出数 - 过检
                            expected_correct = casi_detected - overdetect
                            check1_pass = abs(expected_correct - correct) < 0.01
                            
                            # 验证2：正确检出 + 漏检 = KLA总数
                            sum_check = correct + miss
                            check2_pass = abs(sum_check - kla_total) < 0.01
                            
                            status1 = "✅" if check1_pass else "❌"
                            status2 = "✅" if check2_pass else "❌"
                            
                            if not (check1_pass and check2_pass):
                                all_checks_pass = False
                            
                            st.write(f"{row['CASI文件夹']} vs {row['KLA文件夹']}: "
                                   f"{status1} 正检={casi_detected}-{overdetect}={expected_correct}(实际:{correct}) | "
                                   f"{status2} 正检({correct})+漏检({miss})={sum_check}(KLA:{kla_total})")
                        
                        if all_checks_pass:
                            st.success("✅ 所有数据验证通过！")
                        else:
                            st.error("❌ 部分数据验证失败，请检查！")
                        
                        # 提供简化版下载
                        csv_simplified = results_df_simplified.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载简化版匹配结果（CSV）",
                            data=csv_simplified,
                            file_name=f"kla_match_results_simplified_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # 可视化分析
                        st.write("### 📈 可视化分析")
                        
                        # 堆叠柱状图
                        fig_stack = go.Figure()
                        
                        results_df['组合'] = results_df['CASI文件夹'] + '\nvs\n' + results_df['KLA文件夹']
                        
                        fig_stack.add_trace(go.Bar(
                            name='过检',
                            x=results_df['组合'],
                            y=results_df['过检(0)'],
                            marker_color="#F5BC02"
                        ))
                        fig_stack.add_trace(go.Bar(
                            name='正确检出',
                            x=results_df['组合'],
                            y=results_df['正确检出(1,3,4,5)'],
                            marker_color="#09A84C"
                        ))
                        fig_stack.add_trace(go.Bar(
                            name='漏检',
                            x=results_df['组合'],
                            y=results_df['漏检总数'],
                            marker_color="#3508FF"
                        ))
                        
                        fig_stack.update_layout(
                            title='KLA匹配结果对比',
                            barmode='stack',
                            xaxis_title='CASI vs KLA',
                            yaxis_title='缺陷数量',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_stack, use_container_width=True)
                        
                        # 新增：过漏检缺陷分布图
                        st.write("### 🗺️ 过漏检缺陷分布图")
                        st.markdown("""
                        晶圆图显示缺陷的分类情况：
                        - 🟢 **绿色**：正确检出的缺陷
                        - 🔴 **红色**：漏检的缺陷
                        - 🟡 **黄色**：过检的缺陷
                        """)
                        
                        # 网格显示选项
                        show_grid_wafer = st.checkbox("显示背景网格", value=False, key="show_grid_wafer_defect", help="控制晶圆图中是否显示背景网格线")
                        
                        # 为每个组合创建晶圆图
                        for idx, row in results_df.iterrows():
                            with st.expander(f"📊 {row['组合']} - 缺陷分布图", expanded=False):
                                # 获取该组合的坐标数据
                                coord_data = row['coord_data']
                                
                                # 创建晶圆图
                                fig_wafer = go.Figure()
                                
                                # 添加正确检出的点（绿色）
                                if '正确检出' in coord_data and len(coord_data['正确检出']) > 0:
                                    correct_coords = coord_data['正确检出']
                                    x_coords = [coord[0] for coord in correct_coords]
                                    y_coords = [coord[1] for coord in correct_coords]
                                    
                                    fig_wafer.add_trace(go.Scatter(
                                        x=x_coords,
                                        y=y_coords,
                                        mode='markers',
                                        name=f'正确检出 ({len(correct_coords)})',
                                        marker=dict(
                                            size=6,
                                            color="#06AC4B",
                                            opacity=0.6,
                                            # line=dict(width=0.5, color='white')
                                        ),
                                        hovertemplate='<b>正确检出</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                                    ))
                                
                                # 添加漏检的点（红色）
                                if '漏检' in coord_data and len(coord_data['漏检']) > 0:
                                    miss_coords = coord_data['漏检']
                                    x_coords = [coord[0] for coord in miss_coords]
                                    y_coords = [coord[1] for coord in miss_coords]
                                    
                                    fig_wafer.add_trace(go.Scatter(
                                        x=x_coords,
                                        y=y_coords,
                                        mode='markers',
                                        name=f'漏检 ({len(miss_coords)})',
                                        marker=dict(
                                            size=6,
                                            color="#3508FF",
                                            opacity=0.6,
                                            # line=dict(width=0.5, color='white')
                                        ),
                                        hovertemplate='<b>漏检</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                                    ))
                                
                                # 添加过检的点（黄色）
                                if '过检' in coord_data and len(coord_data['过检']) > 0:
                                    over_coords = coord_data['过检']
                                    x_coords = [coord[0] for coord in over_coords]
                                    y_coords = [coord[1] for coord in over_coords]
                                    
                                    fig_wafer.add_trace(go.Scatter(
                                        x=x_coords,
                                        y=y_coords,
                                        mode='markers',
                                        name=f'过检 ({len(over_coords)})',
                                        marker=dict(
                                            size=6,
                                            color="#F50202",
                                            opacity=0.6,
                                            # line=dict(width=0.5, color='darkgoldenrod')
                                        ),
                                        hovertemplate='<b>过检</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                                    ))
                                
                                # 添加晶圆边界圆（以150000为中心，半径150000）
                                theta = np.linspace(0, 2*np.pi, 100)
                                circle_x = 150000 + 150000 * np.cos(theta)
                                circle_y = 150000 + 150000 * np.sin(theta)
                                
                                fig_wafer.add_trace(go.Scatter(
                                    x=circle_x,
                                    y=circle_y,
                                    mode='lines',
                                    name='晶圆边界',
                                    line=dict(color='black', width=2),
                                    showlegend=True,
                                    hoverinfo='skip'
                                ))
                                
                                # 设置布局
                                fig_wafer.update_layout(
                                    title=dict(
                                        text=f'{row["组合"]} - 缺陷分布',
                                        x=0.5,
                                        xanchor='center',
                                        font=dict(size=16)
                                    ),
                                    xaxis=dict(
                                        title='X坐标',
                                        range=[0, 300000],
                                        scaleanchor="y",
                                        scaleratio=1,
                                        showgrid=show_grid_wafer,
                                        gridcolor='lightgray'
                                    ),
                                    yaxis=dict(
                                        title='Y坐标',
                                        range=[0, 300000],
                                        showgrid=show_grid_wafer,
                                        gridcolor='lightgray'
                                    ),
                                    plot_bgcolor='white',
                                    hovermode='closest',
                                    width=800,
                                    height=800,
                                    legend=dict(
                                        orientation="v",
                                        yanchor="top",
                                        y=1,
                                        xanchor="left",
                                        x=1.02,
                                        bgcolor='rgba(255,255,255,0.9)',
                                        bordercolor='gray',
                                        borderwidth=1
                                    )
                                )
                                
                                st.plotly_chart(fig_wafer, use_container_width=True)
                                
                                # 添加保存图表功能
                                col_save1, col_save2 = st.columns(2)
                                with col_save1:
                                    # 保存为HTML
                                    html_buffer = fig_wafer.to_html(include_plotlyjs='cdn')
                                    st.download_button(
                                        label="📥 下载为HTML",
                                        data=html_buffer,
                                        file_name=f"缺陷分布_{row['CASI文件夹']}_vs_{row['KLA文件夹']}.html",
                                        mime="text/html",
                                        key=f"download_html_{idx}"
                                    )
                                with col_save2:
                                    # 保存为PNG（需要安装kaleido）- 不带坐标系
                                    # 创建一个用于导出的图表副本，隐藏坐标轴
                                    fig_export = go.Figure(fig_wafer)
                                    fig_export.update_layout(
                                        xaxis=dict(
                                            visible=False,
                                            range=[0, 300000],
                                            scaleanchor="y",
                                            scaleratio=1
                                        ),
                                        yaxis=dict(
                                            visible=False,
                                            range=[0, 300000]
                                        ),
                                        showlegend=False,
                                        title=None,
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        margin=dict(l=50, r=50, t=50, b=50)  # 增加白色边距
                                    )
                                    
                                    # 尝试导出PNG（高清：1200x1200基础尺寸，scale=3输出3600x3600）
                                    try:
                                        img_bytes = fig_export.to_image(
                                            format="png", 
                                            width=1200,      # 宽度增加到1200
                                            height=1200,     # 高度增加到1200
                                            scale=3          # 3倍缩放，输出3600x3600的高清图
                                        )
                                        st.download_button(
                                            label="📥 下载为PNG（高清）",
                                            data=img_bytes,
                                            file_name=f"缺陷分布_{row['CASI文件夹']}_vs_{row['KLA文件夹']}_HD.png",
                                            mime="image/png",
                                            key=f"download_png_{idx}"
                                        )
                                    except Exception as e:
                                        # 如果kaleido未安装，显示安装提示
                                        st.button(
                                            "� 下载为PNG（无坐标）",
                                            disabled=True,
                                            key=f"download_png_disabled_{idx}",
                                            help="需要安装 kaleido"
                                        )
                                        st.caption("💡 需要安装: `pip install kaleido`")
                                
                                # 显示该组合的统计信息
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("正确检出", len(coord_data.get('正确检出', [])))
                                with col_b:
                                    st.metric("漏检", len(coord_data.get('漏检', [])))
                                with col_c:
                                    st.metric("过检", len(coord_data.get('过检', [])))
                        
                        # 统计摘要
                        st.write("### 📋 统计摘要")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            total_overdetect = results_df['过检(0)'].sum()
                            st.metric("总过检数", total_overdetect)
                        with col2:
                            total_overdetect_clean = results_df['过检（去除污染）'].sum()
                            st.metric("总过检数（去污染）", total_overdetect_clean)
                        with col3:
                            total_correct = results_df['正确检出(1,3,4,5)'].sum()
                            st.metric("总正确检出", total_correct)
                        with col4:
                            total_miss_total = results_df['漏检总数'].sum()
                            st.metric("漏检总数", total_miss_total)
                        with col5:
                            avg_detect_rate = results_df['正确检出(1,3,4,5)'].sum() / results_df['CASI总数'].sum() * 100 if results_df['CASI总数'].sum() > 0 else 0
                            st.metric("平均检出率", f"{avg_detect_rate:.2f}%")
                        
                        # 添加去除污染后的对比图
                        st.write("### 📊 去除污染后的匹配结果对比")
                        fig_stack_clean = go.Figure()
                        
                        fig_stack_clean.add_trace(go.Bar(
                            name='过检（去污染）',
                            x=results_df['组合'],
                            y=results_df['过检（去除污染）'],
                            marker_color='#FF6B6B'
                        ))
                        fig_stack_clean.add_trace(go.Bar(
                            name='正确检出',
                            x=results_df['组合'],
                            y=results_df['正确检出(1,3,4,5)'],
                            marker_color='#4ECDC4'
                        ))
                        fig_stack_clean.add_trace(go.Bar(
                            name='漏检',
                            x=results_df['组合'],
                            y=results_df['漏检总数'],
                            marker_color='#FFE66D'
                        ))
                        
                        fig_stack_clean.update_layout(
                            title='KLA匹配结果对比（去除污染后）',
                            barmode='stack',
                            xaxis_title='CASI vs KLA',
                            yaxis_title='缺陷数量',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_stack_clean, use_container_width=True)
                        
                        # 新增：DSIZE尺寸检出率分析
                        st.write("---")
                        st.markdown('<a name="DSIZE尺寸分析"></a>', unsafe_allow_html=True)
                        st.subheader("📏 DSIZE尺寸检出率分析")
                        
                        with st.expander("📏 查看DSIZE尺寸分析详情", expanded=False):
                            st.markdown("""
                            分析正确检出和漏检缺陷的DSIZE尺寸分布，帮助了解不同尺寸缺陷的检出情况。
                            """)
                        
                        # 转换DSIZE列为数值类型
                        for col in ['正确检出DSIZE均值', '正确检出DSIZE最小', '正确检出DSIZE最大', 
                                   '漏检DSIZE均值', '漏检DSIZE最小', '漏检DSIZE最大']:
                            if col in results_df.columns:
                                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                        
                        # DSIZE均值对比图
                        st.write("### 📊 正确检出 vs 漏检的DSIZE均值对比")
                        fig_dsize_avg = go.Figure()
                        
                        fig_dsize_avg.add_trace(go.Bar(
                            name='正确检出DSIZE均值',
                            x=results_df['组合'],
                            y=results_df['正确检出DSIZE均值'],
                            marker_color='#4ECDC4',
                            text=results_df['正确检出DSIZE均值'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"),
                            textposition='auto',
                        ))
                        
                        fig_dsize_avg.add_trace(go.Bar(
                            name='漏检DSIZE均值',
                            x=results_df['组合'],
                            y=results_df['漏检DSIZE均值'],
                            marker_color='#FFE66D',
                            text=results_df['漏检DSIZE均值'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"),
                            textposition='auto',
                        ))
                        
                        fig_dsize_avg.update_layout(
                            title='正确检出 vs 漏检的DSIZE均值对比',
                            xaxis_title='CASI vs KLA',
                            yaxis_title='DSIZE均值',
                            barmode='group',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_dsize_avg, use_container_width=True)
                        
                        # DSIZE范围对比图
                        st.write("### 📊 DSIZE尺寸范围对比")
                        
                        fig_dsize_range = go.Figure()
                        
                        # 正确检出的DSIZE范围
                        fig_dsize_range.add_trace(go.Scatter(
                            x=results_df['组合'],
                            y=results_df['正确检出DSIZE最大'],
                            mode='lines+markers',
                            name='正确检出最大值',
                            line=dict(color='#4ECDC4', width=2),
                            marker=dict(size=8)
                        ))
                        
                        fig_dsize_range.add_trace(go.Scatter(
                            x=results_df['组合'],
                            y=results_df['正确检出DSIZE均值'],
                            mode='lines+markers',
                            name='正确检出均值',
                            line=dict(color='#45B7D1', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        fig_dsize_range.add_trace(go.Scatter(
                            x=results_df['组合'],
                            y=results_df['正确检出DSIZE最小'],
                            mode='lines+markers',
                            name='正确检出最小值',
                            line=dict(color='#A2D9CE', width=2),
                            marker=dict(size=8)
                        ))
                        
                        # 漏检的DSIZE范围
                        fig_dsize_range.add_trace(go.Scatter(
                            x=results_df['组合'],
                            y=results_df['漏检DSIZE最大'],
                            mode='lines+markers',
                            name='漏检最大值',
                            line=dict(color='#FFE66D', width=2),
                            marker=dict(size=8)
                        ))
                        
                        fig_dsize_range.add_trace(go.Scatter(
                            x=results_df['组合'],
                            y=results_df['漏检DSIZE均值'],
                            mode='lines+markers',
                            name='漏检均值',
                            line=dict(color='#FFA07A', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        fig_dsize_range.add_trace(go.Scatter(
                            x=results_df['组合'],
                            y=results_df['漏检DSIZE最小'],
                            mode='lines+markers',
                            name='漏检最小值',
                            line=dict(color='#FF6B6B', width=2),
                            marker=dict(size=8)
                        ))
                        
                        fig_dsize_range.update_layout(
                            title='DSIZE尺寸范围对比（正确检出 vs 漏检）',
                            xaxis_title='CASI vs KLA',
                            yaxis_title='DSIZE值',
                            height=600,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            )
                        )
                        
                        st.plotly_chart(fig_dsize_range, use_container_width=True)
                        
                        # DSIZE统计摘要表
                        st.write("### 📋 DSIZE统计摘要")
                        
                        summary_dsize = pd.DataFrame({
                            'CASI-KLA组合': results_df['组合'],
                            '正确检出数': results_df['正确检出(1,3,4,5)'],
                            '正确检出DSIZE均值': results_df['正确检出DSIZE均值'].apply(lambda x: f"{x:.6f}" if pd.notna(x) and x > 0 else "N/A"),
                            '正确检出DSIZE范围': results_df.apply(lambda row: f"{row['正确检出DSIZE最小']:.6f} - {row['正确检出DSIZE最大']:.6f}" 
                                                            if pd.notna(row['正确检出DSIZE最小']) and row['正确检出DSIZE最小'] > 0 else "N/A", axis=1),
                            '漏检数': results_df['漏检总数'],
                            '漏检DSIZE均值': results_df['漏检DSIZE均值'].apply(lambda x: f"{x:.6f}" if pd.notna(x) and x > 0 else "N/A"),
                            '漏检DSIZE范围': results_df.apply(lambda row: f"{row['漏检DSIZE最小']:.6f} - {row['漏检DSIZE最大']:.6f}" 
                                                        if pd.notna(row['漏检DSIZE最小']) and row['漏检DSIZE最小'] > 0 else "N/A", axis=1)
                        })
                        
                        st.dataframe(summary_dsize, use_container_width=True)
                        
                        # 新增：过检数据的DW1O_Size和DW2O_Size尺寸分布分析
                        st.write("---")
                        st.markdown('<a name="过检尺寸分布"></a>', unsafe_allow_html=True)
                        st.subheader("📊 过检数据尺寸分布分析（DW1O_Size & DW2O_Size）")
                        
                        with st.expander("📐 查看过检尺寸分布详情", expanded=False):
                            st.markdown("""
                            分析过检数据的DW1O_Size和DW2O_Size尺寸分布，帮助了解过检缺陷的尺寸特征。
                            - **DW1O_Size**：过检缺陷在DW1O通道的尺寸测量值（单位：nm，无需转换）
                            - **DW2O_Size**：过检缺陷在DW2O通道的尺寸测量值（单位：nm，无需转换）
                            - 仅统计CASI过检数据，不涉及KLA数据对比
                            """)
                        
                        # 检查是否有过检尺寸数据
                        has_overdetect_size_data = any(
                            result.get('overdetect_size_stats', {}).get('has_size_data', False) 
                            for result in all_match_results
                        )
                        
                        if has_overdetect_size_data:
                            # 创建过检尺寸统计表
                            overdetect_size_summary = []
                            for result in all_match_results:
                                casi_name = result['CASI文件夹']
                                kla_name = result['KLA文件夹']
                                n_overdetect = result['过检(0)']
                                overdetect_stats = result.get('overdetect_size_stats', {})
                                
                                if overdetect_stats.get('has_size_data', False):
                                    dw1o = overdetect_stats.get('dw1o_size', {})
                                    dw2o = overdetect_stats.get('dw2o_size', {})
                                    
                                    row = {
                                        'CASI-KLA组合': f"{casi_name} vs {kla_name}",
                                        '过检数量': n_overdetect,
                                        'DW1O_Size样本数': len(dw1o.get('values', [])),
                                        'DW1O_Size为200000数量': dw1o.get('count_200000', 0),
                                        'DW1O_Size均值': f"{dw1o.get('mean', 0):.2f}" if dw1o.get('mean', 0) > 0 else "N/A",
                                        'DW1O_Size最小': f"{dw1o.get('min', 0):.2f}" if dw1o.get('min', 0) > 0 else "N/A",
                                        'DW1O_Size最大': f"{dw1o.get('max', 0):.2f}" if dw1o.get('max', 0) > 0 else "N/A",
                                        'DW1O_Size标准差': f"{dw1o.get('std', 0):.2f}" if dw1o.get('std', 0) > 0 else "N/A",
                                        'DW2O_Size样本数': len(dw2o.get('values', [])),
                                        'DW2O_Size为200000数量': dw2o.get('count_200000', 0),
                                        'DW2O_Size均值': f"{dw2o.get('mean', 0):.2f}" if dw2o.get('mean', 0) > 0 else "N/A",
                                        'DW2O_Size最小': f"{dw2o.get('min', 0):.2f}" if dw2o.get('min', 0) > 0 else "N/A",
                                        'DW2O_Size最大': f"{dw2o.get('max', 0):.2f}" if dw2o.get('max', 0) > 0 else "N/A",
                                        'DW2O_Size标准差': f"{dw2o.get('std', 0):.2f}" if dw2o.get('std', 0) > 0 else "N/A"
                                    }
                                    overdetect_size_summary.append(row)
                            
                            if overdetect_size_summary:
                                st.write("### 📋 过检尺寸统计摘要")
                                overdetect_size_df = pd.DataFrame(overdetect_size_summary)
                                st.dataframe(overdetect_size_df, use_container_width=True)
                                
                                # 提供CSV下载
                                csv_overdetect_size = overdetect_size_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 下载过检尺寸统计（CSV）",
                                    data=csv_overdetect_size,
                                    file_name=f"overdetect_size_stats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # DW1O_Size和DW2O_Size分布箱线图对比
                                st.write("### 📊 过检尺寸分布箱线图（CASI数据）")
                                
                                st.info("以下图表展示的是CASI过检数据的DW1O_Size和DW2O_Size尺寸分布，单位为nm，无需转换。")
                                
                                # 为每个CASI-KLA组合创建箱线图
                                for result in all_match_results:
                                    casi_name = result['CASI文件夹']
                                    kla_name = result['KLA文件夹']
                                    overdetect_stats = result.get('overdetect_size_stats', {})
                                    
                                    if not overdetect_stats.get('has_size_data', False):
                                        continue
                                    
                                    dw1o_values = overdetect_stats.get('dw1o_size', {}).get('values', [])
                                    dw2o_values = overdetect_stats.get('dw2o_size', {}).get('values', [])
                                    
                                    if len(dw1o_values) > 0 or len(dw2o_values) > 0:
                                        st.write(f"#### {casi_name} vs {kla_name}")
                                        
                                        fig_box = go.Figure()
                                        
                                        if len(dw1o_values) > 0:
                                            fig_box.add_trace(go.Box(
                                                y=dw1o_values,
                                                name='DW1O_Size (nm)',
                                                marker_color='#4ECDC4',
                                                boxmean='sd'  # 显示均值和标准差
                                            ))
                                        
                                        if len(dw2o_values) > 0:
                                            fig_box.add_trace(go.Box(
                                                y=dw2o_values,
                                                name='DW2O_Size (nm)',
                                                marker_color='#FF6B6B',
                                                boxmean='sd'
                                            ))
                                        
                                        fig_box.update_layout(
                                            title=f'CASI过检数据尺寸分布 - {casi_name} vs {kla_name}',
                                            yaxis_title='尺寸值 (nm)',
                                            height=500,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(fig_box, use_container_width=True)
                                        
                                        # 显示统计信息
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            dw1o_stats = overdetect_stats.get('dw1o_size', {})
                                            if len(dw1o_values) > 0:
                                                st.write("**DW1O_Size统计：**")
                                                st.write(f"- 样本数（排除200000）: {len(dw1o_values)}")
                                                st.write(f"- 200000数量: {dw1o_stats.get('count_200000', 0)}")
                                                st.write(f"- 均值: {np.mean(dw1o_values):.2f}")
                                                st.write(f"- 中位数: {np.median(dw1o_values):.2f}")
                                                st.write(f"- 标准差: {np.std(dw1o_values):.2f}")
                                                st.write(f"- 范围: {np.min(dw1o_values):.2f} - {np.max(dw1o_values):.2f}")
                                        
                                        with col2:
                                            dw2o_stats = overdetect_stats.get('dw2o_size', {})
                                            if len(dw2o_values) > 0:
                                                st.write("**DW2O_Size统计：**")
                                                st.write(f"- 样本数（排除200000）: {len(dw2o_values)}")
                                                st.write(f"- 200000数量: {dw2o_stats.get('count_200000', 0)}")
                                                st.write(f"- 均值: {np.mean(dw2o_values):.2f}")
                                                st.write(f"- 中位数: {np.median(dw2o_values):.2f}")
                                                st.write(f"- 标准差: {np.std(dw2o_values):.2f}")
                                                st.write(f"- 范围: {np.min(dw2o_values):.2f} - {np.max(dw2o_values):.2f}")
                                        
                                        # 直方图分布
                                        st.write("**尺寸分布直方图：**")
                                        
                                        fig_hist = go.Figure()
                                        
                                        if len(dw1o_values) > 0:
                                            fig_hist.add_trace(go.Histogram(
                                                x=dw1o_values,
                                                name='DW1o_size (nm)',
                                                marker_color='#4ECDC4',
                                                opacity=0.7,
                                                nbinsx=30
                                            ))
                                        
                                        if len(dw2o_values) > 0:
                                            fig_hist.add_trace(go.Histogram(
                                                x=dw2o_values,
                                                name='DW2O_Size (nm)',
                                                marker_color='#FF6B6B',
                                                opacity=0.7,
                                                nbinsx=30
                                            ))
                                        
                                        fig_hist.update_layout(
                                            title=f'CASI过检尺寸分布直方图 - {casi_name} vs {kla_name}',
                                            xaxis_title='尺寸值 (nm)',
                                            yaxis_title='频数',
                                            barmode='overlay',
                                            height=400,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                        
                                        st.write("---")
                                
                                # 总体对比（所有组合）
                                st.write("### 📊 总体过检尺寸分布对比（CASI数据）")
                                
                                # 收集所有DW1O和DW2O数据，同时统计200000的数量
                                all_dw1o_values = []
                                all_dw2o_values = []
                                total_dw1o_count_200000 = 0
                                total_dw2o_count_200000 = 0
                                
                                for result in all_match_results:
                                    overdetect_stats = result.get('overdetect_size_stats', {})
                                    if overdetect_stats.get('has_size_data', False):
                                        all_dw1o_values.extend(overdetect_stats.get('dw1o_size', {}).get('values', []))
                                        all_dw2o_values.extend(overdetect_stats.get('dw2o_size', {}).get('values', []))
                                        total_dw1o_count_200000 += overdetect_stats.get('dw1o_size', {}).get('count_200000', 0)
                                        total_dw2o_count_200000 += overdetect_stats.get('dw2o_size', {}).get('count_200000', 0)

                                if len(all_dw1o_values) > 0 or len(all_dw2o_values) > 0:
                                    st.info("📊 该图表汇总所有组合的CASI过检数据，尺寸单位为nm，无需转换。已排除200000的数据。")
                                    
                                    fig_overall = go.Figure()
                                    
                                    if len(all_dw1o_values) > 0:
                                        fig_overall.add_trace(go.Box(
                                            y=all_dw1o_values,
                                            name=f'DW1O_Size (nm) (n={len(all_dw1o_values)})',
                                            marker_color='#4ECDC4',
                                            boxmean='sd'
                                        ))
                                    
                                    if len(all_dw2o_values) > 0:
                                        fig_overall.add_trace(go.Box(
                                            y=all_dw2o_values,
                                            name=f'DW2O_Size (nm) (n={len(all_dw2o_values)})',
                                            marker_color='#FF6B6B',
                                            boxmean='sd'
                                        ))
                                    
                                    fig_overall.update_layout(
                                        title='CASI过检数据总体尺寸分布对比',
                                        yaxis_title='尺寸值 (nm)',
                                        height=500,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_overall, use_container_width=True)
                                    
                                    # 总体统计摘要
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if len(all_dw1o_values) > 0:
                                            st.write("**DW1O_Size总体统计 (nm)：**")
                                            st.metric("总样本数（排除200000）", len(all_dw1o_values))
                                            st.metric("200000数量", total_dw1o_count_200000)
                                            st.metric("均值 (nm)", f"{np.mean(all_dw1o_values):.2f}")
                                            st.metric("中位数 (nm)", f"{np.median(all_dw1o_values):.2f}")
                                            st.metric("标准差 (nm)", f"{np.std(all_dw1o_values):.2f}")
                                    
                                    with col2:
                                        if len(all_dw2o_values) > 0:
                                            st.write("**DW2O_Size总体统计 (nm)：**")
                                            st.metric("总样本数（排除200000）", len(all_dw2o_values))
                                            st.metric("200000数量", total_dw2o_count_200000)
                                            st.metric("均值 (nm)", f"{np.mean(all_dw2o_values):.2f}")
                                            st.metric("中位数 (nm)", f"{np.median(all_dw2o_values):.2f}")
                                            st.metric("标准差 (nm)", f"{np.std(all_dw2o_values):.2f}")
                            else:
                                st.info("所有组合均无过检尺寸数据")
                        else:
                            st.info("BlobFeatures文件中未找到DW1O_Size或DW2O_Size列，无法进行过检尺寸分布分析")
                        
                        # 新增：按尺寸区间的详细检出统计（26nm-100nm）
                        st.write("---")
                        st.markdown('<a name="按尺寸区间统计"></a>', unsafe_allow_html=True)
                        st.subheader("📊 按尺寸区间的详细检出统计（26nm-100nm，DSIZE×1000）")
                        
                        with st.expander("📊 查看按尺寸区间统计详情", expanded=False):
                            st.markdown("""
                            按每1nm为一个区间，统计26nm到100nm范围内各尺寸的检出情况。
                            - **总数**：该尺寸区间的KLA缺陷总数
                            - **正确检出**：该尺寸区间被正确检出的缺陷数
                            - **漏检**：该尺寸区间未被检出的缺陷数
                            - **检出率**：正确检出数 / 总数 × 100%
                            """)
                        
                        # 为每个CASI-KLA组合生成详细统计表
                        for idx, result in enumerate(all_match_results):
                            casi_name = result['CASI文件夹']
                            kla_name = result['KLA文件夹']
                            size_stats = result.get('size_stats', None)
                            
                            if size_stats is None:
                                continue
                            
                            st.write(f"#### {casi_name} vs {kla_name}")
                            
                            # 构建详细统计表
                            detail_rows = []
                            for size_nm in size_stats['bins']:
                                total = size_stats['total_count'][size_nm]
                                correct = size_stats['correct_count'][size_nm]
                                miss = size_stats['miss_count'][size_nm]
                                detect_rate = (correct / total * 100) if total > 0 else 0
                                
                                # 只显示有数据的区间
                                if total > 0:
                                    detail_rows.append({
                                        '尺寸区间(nm)': f"{size_nm}nm",
                                        'KLA总数': total,
                                        '正确检出': correct,
                                        '漏检': miss,
                                        '检出率': f"{detect_rate:.2f}%"
                                    })
                            
                            if detail_rows:
                                detail_df = pd.DataFrame(detail_rows)
                                st.dataframe(detail_df, use_container_width=True, height=400)
                                
                                # 提供CSV下载
                                csv_detail = detail_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label=f"📥 下载 {casi_name}-{kla_name} 详细统计（CSV）",
                                    data=csv_detail,
                                    file_name=f"size_detail_{casi_name}_{kla_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key=f"download_detail_{idx}"
                                )
                                
                                # 绘制检出率曲线图
                                fig_detect_rate = go.Figure()
                                
                                fig_detect_rate.add_trace(go.Scatter(
                                    x=[row['尺寸区间(nm)'] for row in detail_rows],
                                    y=[float(row['检出率'].rstrip('%')) for row in detail_rows],
                                    mode='lines+markers',
                                    name='检出率',
                                    line=dict(color='#4ECDC4', width=2),
                                    marker=dict(size=6),
                                    text=[f"总数:{row['KLA总数']}<br>正确:{row['正确检出']}<br>漏检:{row['漏检']}" 
                                          for row in detail_rows],
                                    hovertemplate='<b>%{x}</b><br>检出率: %{y:.2f}%<br>%{text}<extra></extra>'
                                ))
                                
                                fig_detect_rate.update_layout(
                                    title=f'{casi_name} vs {kla_name} - 各尺寸区间检出率',
                                    xaxis_title='尺寸区间',
                                    yaxis_title='检出率 (%)',
                                    yaxis=dict(range=[0, 105]),
                                    height=500,
                                    hovermode='closest'
                                )
                                
                                st.plotly_chart(fig_detect_rate, use_container_width=True)
                                
                                # 绘制堆叠柱状图：正确检出 vs 漏检
                                fig_stack_size = go.Figure()
                                
                                fig_stack_size.add_trace(go.Bar(
                                    x=[row['尺寸区间(nm)'] for row in detail_rows],
                                    y=[row['正确检出'] for row in detail_rows],
                                    name='正确检出',
                                    marker_color='#4ECDC4',
                                    text=[row['正确检出'] for row in detail_rows],
                                    textposition='inside'
                                ))
                                
                                fig_stack_size.add_trace(go.Bar(
                                    x=[row['尺寸区间(nm)'] for row in detail_rows],
                                    y=[row['漏检'] for row in detail_rows],
                                    name='漏检',
                                    marker_color='#FFE66D',
                                    text=[row['漏检'] for row in detail_rows],
                                    textposition='inside'
                                ))
                                
                                fig_stack_size.update_layout(
                                    title=f'{casi_name} vs {kla_name} - 各尺寸区间缺陷分布',
                                    xaxis_title='尺寸区间',
                                    yaxis_title='缺陷数量',
                                    barmode='stack',
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_stack_size, use_container_width=True)
                            else:
                                st.info(f"{casi_name} vs {kla_name}: 26nm-100nm范围内无数据")
                            
                            st.write("---")
                        
                        # 新增：DSIZE与DW1O_Size对比分析（仅针对正确检出的缺陷）
                        st.write("---")
                        st.subheader("📏 DSIZE与DW1O_Size对比分析（正确检出缺陷）")
                        
                        st.markdown("""
                        对所有匹配后判断为正确检出的缺陷，对比KLA文件中的DSIZE与各子文件夹的DW1O_Size。
                        - DSIZE需要乘以1000进行单位转换
                        - 仅分析匹配结果为1、3、4、5（正确检出）的缺陷
                        """)
                        
                        with st.spinner("正在分析DSIZE和DW1O_Size..."):
                            size_comparison_data = []
                            
                            # 对每个CASI文件夹重新读取数据并提取正确检出的缺陷
                            for casi_folder in sorted(casi_folders):
                                casi_csv_path = os.path.join(kla_match_folder, casi_folder, 'jianchu.csv')
                                
                                if not os.path.exists(casi_csv_path):
                                    continue
                                
                                # 读取CASI数据
                                casi_df = pd.read_csv(casi_csv_path)
                                casi_df.columns = casi_df.columns.str.strip()
                                
                                # 获取CASI坐标列
                                cas_x_col = None
                                cas_y_col = None
                                for x_candidate in ['dCenterXCartisian', 'dCenterXCartesian', 'XREL', 'cx']:
                                    if x_candidate in casi_df.columns:
                                        cas_x_col = x_candidate
                                        break
                                for y_candidate in ['dCenterYCartisian', 'dCenterYCartesian', 'YREL', 'cy']:
                                    if y_candidate in casi_df.columns:
                                        cas_y_col = y_candidate
                                        break
                                
                                # 检查DW1O_Size列是否存在
                                if 'DW1O_Size' not in casi_df.columns:
                                    st.warning(f"{casi_folder}: 缺少DW1O_Size列")
                                    continue
                                
                                if cas_x_col is None or cas_y_col is None:
                                    continue
                                
                                # 对每个KLA文件夹进行匹配
                                for kla_folder in sorted(kla_folders):
                                    kla_csv_path = os.path.join(kla_match_folder, kla_folder, 'jianchu.csv')
                                    
                                    if not os.path.exists(kla_csv_path):
                                        continue
                                    
                                    # 读取KLA数据（保持完整精度，特别是DSIZE列）
                                    kla_df = pd.read_csv(kla_csv_path, dtype={'DSIZE': float}, float_precision='high')
                                    kla_df.columns = kla_df.columns.str.strip()
                                    
                                    # 检查KLA必需的列
                                    if not {'XREL', 'YREL', 'DSIZE'}.issubset(kla_df.columns):
                                        st.warning(f"{kla_folder}: 缺少XREL/YREL/DSIZE列")
                                        continue
                                    
                                    # 准备数据
                                    casi_work = casi_df[[cas_x_col, cas_y_col, 'DW1O_Size']].copy()
                                    casi_work.columns = ['XREL', 'YREL', 'DW1O_Size']
                                    casi_work['XREL'] = pd.to_numeric(casi_work['XREL'], errors='coerce')
                                    casi_work['YREL'] = pd.to_numeric(casi_work['YREL'], errors='coerce')
                                    casi_work['DW1O_Size'] = pd.to_numeric(casi_work['DW1O_Size'], errors='coerce')
                                    casi_work = casi_work.dropna(subset=['XREL', 'YREL']).reset_index(drop=True)
                                    
                                    kla_work = kla_df[['XREL', 'YREL', 'DSIZE']].copy()
                                    kla_work['XREL'] = pd.to_numeric(kla_work['XREL'], errors='coerce')
                                    kla_work['YREL'] = pd.to_numeric(kla_work['YREL'], errors='coerce')
                                    kla_work['DSIZE'] = pd.to_numeric(kla_work['DSIZE'], errors='coerce', downcast=None)
                                    kla_work = kla_work.dropna(subset=['XREL', 'YREL']).reset_index(drop=True)
                                    
                                    # 构建KDTree并执行匹配
                                    if len(casi_work) > 0 and len(kla_work) > 0:
                                        casi_pts = casi_work[['XREL', 'YREL']].to_numpy()
                                        kla_pts = kla_work[['XREL', 'YREL']].to_numpy()
                                        
                                        tree_casi = KDTree(casi_pts)
                                        tree_kla = KDTree(kla_pts)
                                        
                                        # 初始化匹配结果
                                        casi_match_result = np.full(len(casi_work), np.nan)
                                        casi_matched_kla_idx = np.full(len(casi_work), -1, dtype=int)
                                        
                                        # KLA -> CASI 匹配
                                        for kla_idx in range(len(kla_pts)):
                                            kla_pt = kla_pts[kla_idx]
                                            casi_idx_list = tree_casi.query_ball_point(kla_pt, r=kla_match_threshold)
                                            
                                            if len(casi_idx_list) == 1:
                                                casi_match_result[casi_idx_list[0]] = 1
                                                casi_matched_kla_idx[casi_idx_list[0]] = kla_idx
                                            elif len(casi_idx_list) > 1:
                                                casi_match_result[casi_idx_list[0]] = 3
                                                casi_matched_kla_idx[casi_idx_list[0]] = kla_idx
                                                for ci in casi_idx_list[1:]:
                                                    if np.isnan(casi_match_result[ci]):
                                                        casi_match_result[ci] = 0
                                        
                                        # CASI -> KLA 匹配（细化）
                                        for casi_idx in range(len(casi_pts)):
                                            casi_pt = casi_pts[casi_idx]
                                            kla_idx_list = tree_kla.query_ball_point(casi_pt, r=kla_match_threshold)
                                            
                                            if len(kla_idx_list) == 0:
                                                if np.isnan(casi_match_result[casi_idx]):
                                                    casi_match_result[casi_idx] = 0
                                            elif len(kla_idx_list) > 1:
                                                if casi_match_result[casi_idx] == 1:
                                                    casi_match_result[casi_idx] = 4
                                                    if casi_matched_kla_idx[casi_idx] == -1:
                                                        casi_matched_kla_idx[casi_idx] = kla_idx_list[0]
                                                elif casi_match_result[casi_idx] == 3:
                                                    casi_match_result[casi_idx] = 5
                                            elif len(kla_idx_list) == 1:
                                                if casi_matched_kla_idx[casi_idx] == -1:
                                                    casi_matched_kla_idx[casi_idx] = kla_idx_list[0]
                                        
                                        # 提取正确检出的缺陷（匹配结果为1、3、4、5）
                                        correct_detect_mask = np.isin(casi_match_result, [1, 3, 4, 5])
                                        
                                        for casi_idx in np.where(correct_detect_mask)[0]:
                                            kla_idx = casi_matched_kla_idx[casi_idx]
                                            
                                            if kla_idx >= 0 and kla_idx < len(kla_work):
                                                casi_dw1o_size = casi_work.loc[casi_idx, 'DW1O_Size']
                                                kla_dsize = kla_work.loc[kla_idx, 'DSIZE']
                                                
                                                # 过滤条件：排除DW1O_Size为200000.00的缺陷，且两个值都有效
                                                if pd.notna(casi_dw1o_size) and pd.notna(kla_dsize) and casi_dw1o_size != 200000.00:
                                                    kla_dsize_converted = kla_dsize * 1000  # DSIZE乘以1000
                                                    
                                                    size_comparison_data.append({
                                                        'CASI文件夹': casi_folder,
                                                        'KLA文件夹': kla_folder,
                                                        'X坐标': casi_work.loc[casi_idx, 'XREL'],
                                                        'Y坐标': casi_work.loc[casi_idx, 'YREL'],
                                                        'KLA_DSIZE': kla_dsize,
                                                        'KLA_DSIZE_x1000': kla_dsize_converted,
                                                        'CASI_DW1O_Size': casi_dw1o_size,
                                                        'Size差异': casi_dw1o_size - kla_dsize_converted,
                                                        'Size差异率(%)': ((casi_dw1o_size - kla_dsize_converted) / kla_dsize_converted * 100) if kla_dsize_converted != 0 else np.nan
                                                    })
                            
                            # 显示对比结果
                            if size_comparison_data:
                                size_comp_df = pd.DataFrame(size_comparison_data)
                                
                                st.success(f"找到 {len(size_comp_df)} 个正确检出的缺陷具有完整的尺寸数据")
                                
                                # 显示对比表格（DSIZE显示更多小数位）
                                st.write("### 📊 DSIZE与DW1O_Size对比表格")
                                
                                # 创建格式化的显示DataFrame
                                display_df = size_comp_df.copy()
                                display_df['KLA_DSIZE'] = display_df['KLA_DSIZE'].apply(lambda x: f"{x:.6f}")  # 显示6位小数
                                display_df['KLA_DSIZE_x1000'] = display_df['KLA_DSIZE_x1000'].apply(lambda x: f"{x:.3f}")  # 显示3位小数
                                display_df['CASI_DW1O_Size'] = display_df['CASI_DW1O_Size'].apply(lambda x: f"{x:.2f}")
                                display_df['Size差异'] = display_df['Size差异'].apply(lambda x: f"{x:.2f}")
                                display_df['Size差异率(%)'] = display_df['Size差异率(%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                                display_df['X坐标'] = display_df['X坐标'].apply(lambda x: f"{x:.2f}")
                                display_df['Y坐标'] = display_df['Y坐标'].apply(lambda x: f"{x:.2f}")
                                
                                st.dataframe(display_df, use_container_width=True, height=400)
                                
                                # 提供CSV下载（保持完整精度）
                                csv_size = size_comp_df.to_csv(index=False, encoding='utf-8-sig', float_format='%.6f')
                                st.download_button(
                                    label="📥 下载尺寸对比数据（CSV）",
                                    data=csv_size,
                                    file_name=f"size_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # 对比趋势图（纵坐标为KLA的DSIZE×1000）
                                st.write("### 📈 尺寸对比趋势图")
                                
                                # 为每个CASI-KLA组合创建散点图
                                casi_kla_pairs = size_comp_df.groupby(['CASI文件夹', 'KLA文件夹'])
                                
                                for (casi_name, kla_name), group_df in casi_kla_pairs:
                                    st.write(f"#### {casi_name} vs {kla_name}")
                                    
                                    # 创建散点图：X轴为缺陷索引或序号，Y轴为尺寸值
                                    fig_trend = go.Figure()
                                    
                                    # 添加KLA DSIZE×1000的趋势线
                                    fig_trend.add_trace(go.Scatter(
                                        x=list(range(len(group_df))),
                                        y=group_df['KLA_DSIZE_x1000'],
                                        mode='lines+markers',
                                        name='KLA DSIZE×1000',
                                        marker=dict(size=8, color='#FF6B6B'),
                                        line=dict(width=2, color='#FF6B6B')
                                    ))
                                    
                                    # 添加CASI DW1O_Size的趋势线
                                    fig_trend.add_trace(go.Scatter(
                                        x=list(range(len(group_df))),
                                        y=group_df['CASI_DW1O_Size'],
                                        mode='lines+markers',
                                        name='CASI DW1O_Size',
                                        marker=dict(size=8, color='#4ECDC4'),
                                        line=dict(width=2, color='#4ECDC4')
                                    ))
                                    
                                    fig_trend.update_layout(
                                        title=f'尺寸对比趋势图 - {casi_name} vs {kla_name}',
                                        xaxis_title='缺陷编号',
                                        yaxis_title='尺寸值（KLA DSIZE×1000）',
                                        height=500,
                                        hovermode='x unified',
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        )
                                    )
                                    
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                    
                                    # 显示统计信息
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("平均KLA DSIZE×1000", f"{group_df['KLA_DSIZE_x1000'].mean():.2f}")
                                    with col2:
                                        st.metric("平均CASI DW1O_Size", f"{group_df['CASI_DW1O_Size'].mean():.2f}")
                                    with col3:
                                        st.metric("平均差异", f"{group_df['Size差异'].mean():.2f}")
                                    with col4:
                                        avg_diff_rate = group_df['Size差异率(%)'].mean()
                                        st.metric("平均差异率", f"{avg_diff_rate:.2f}%")
                                
                                # 总体统计分析
                                st.write("### 📊 总体统计分析")
                                
                                # 箱线图对比
                                fig_box_size = go.Figure()
                                
                                fig_box_size.add_trace(go.Box(
                                    y=size_comp_df['KLA_DSIZE_x1000'],
                                    name='KLA DSIZE×1000',
                                    marker_color='#FF6B6B'
                                ))
                                
                                fig_box_size.add_trace(go.Box(
                                    y=size_comp_df['CASI_DW1O_Size'],
                                    name='CASI DW1O_Size',
                                    marker_color='#4ECDC4'
                                ))
                                
                                fig_box_size.update_layout(
                                    title='尺寸分布对比（箱线图）',
                                    yaxis_title='尺寸值',
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_box_size, use_container_width=True)
                                
                                # 散点图：KLA vs CASI
                                fig_scatter = go.Figure()
                                
                                # 按CASI-KLA组合分组着色
                                for (casi_name, kla_name), group_df in casi_kla_pairs:
                                    fig_scatter.add_trace(go.Scatter(
                                        x=group_df['KLA_DSIZE_x1000'],
                                        y=group_df['CASI_DW1O_Size'],
                                        mode='markers',
                                        name=f'{casi_name} vs {kla_name}',
                                        marker=dict(size=8, opacity=0.7),
                                        text=[f"X:{x:.0f}, Y:{y:.0f}" for x, y in zip(group_df['X坐标'], group_df['Y坐标'])],
                                        hovertemplate='<b>%{fullData.name}</b><br>KLA DSIZE×1000: %{x:.2f}<br>CASI DW1O_Size: %{y:.2f}<br>%{text}<extra></extra>'
                                    ))
                                
                                # 添加对角线（理想情况：两者相等）
                                min_val = min(size_comp_df['KLA_DSIZE_x1000'].min(), size_comp_df['CASI_DW1O_Size'].min())
                                max_val = max(size_comp_df['KLA_DSIZE_x1000'].max(), size_comp_df['CASI_DW1O_Size'].max())
                                
                                fig_scatter.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='理想匹配线',
                                    line=dict(color='gray', dash='dash', width=2),
                                    showlegend=True,
                                    hoverinfo='skip'
                                ))
                                
                                fig_scatter.update_layout(
                                    title='KLA DSIZE×1000 vs CASI DW1O_Size 散点图',
                                    xaxis_title='KLA DSIZE×1000',
                                    yaxis_title='CASI DW1O_Size',
                                    height=600,
                                    hovermode='closest'
                                )
                                
                                st.plotly_chart(fig_scatter, use_container_width=True)
                                
                                # 统计摘要表
                                st.write("### 📋 统计摘要")
                                summary_stats = pd.DataFrame({
                                    '指标': ['平均值', '中位数', '标准差', '最小值', '最大值'],
                                    'KLA DSIZE×1000': [
                                        size_comp_df['KLA_DSIZE_x1000'].mean(),
                                        size_comp_df['KLA_DSIZE_x1000'].median(),
                                        size_comp_df['KLA_DSIZE_x1000'].std(),
                                        size_comp_df['KLA_DSIZE_x1000'].min(),
                                        size_comp_df['KLA_DSIZE_x1000'].max()
                                    ],
                                    'CASI DW1O_Size': [
                                        size_comp_df['CASI_DW1O_Size'].mean(),
                                        size_comp_df['CASI_DW1O_Size'].median(),
                                        size_comp_df['CASI_DW1O_Size'].std(),
                                        size_comp_df['CASI_DW1O_Size'].min(),
                                        size_comp_df['CASI_DW1O_Size'].max()
                                    ],
                                    'Size差异': [
                                        size_comp_df['Size差异'].mean(),
                                        size_comp_df['Size差异'].median(),
                                        size_comp_df['Size差异'].std(),
                                        size_comp_df['Size差异'].min(),
                                        size_comp_df['Size差异'].max()
                                    ]
                                })
                                
                                st.dataframe(summary_stats.round(2), use_container_width=True)
                                
                            else:
                                st.info("未找到具有完整尺寸数据的正确检出缺陷")
                        
                        # 新增：DW1O_MaxOrg / DW2O_MaxOrg 比值分布分析
                        st.write("---")
                        st.markdown('<a name="MaxOrg比值分析"></a>', unsafe_allow_html=True)
                        st.subheader("📊 DW1O_MaxOrg / DW2O_MaxOrg 比值分布分析")
                        
                        with st.expander("🔢 查看MaxOrg比值分析详情", expanded=False):
                            st.markdown("""
                            分析过检、漏检和正确检出三种类型缺陷的DW1O_MaxOrg与DW2O_MaxOrg比值分布。
                            - **比值 = DW1O_MaxOrg / DW2O_MaxOrg**
                            - **已去除值为0的数据**
                            - 帮助了解不同检出状态下的通道特征差异
                            """)
                        
                        # 检查是否有MaxOrg比值数据
                        has_maxorg_ratio_data = any(
                            result.get('maxorg_ratio_stats', {}).get('has_maxorg_data', False) 
                            for result in all_match_results
                        )
                        
                        if has_maxorg_ratio_data:
                            # 为每个CASI-KLA组合生成统计表和分布图
                            for idx, result in enumerate(all_match_results):
                                casi_name = result['CASI文件夹']
                                kla_name = result['KLA文件夹']
                                maxorg_stats = result.get('maxorg_ratio_stats', {})
                                
                                if not maxorg_stats.get('has_maxorg_data', False):
                                    continue
                                
                                st.write(f"#### {casi_name} vs {kla_name}")
                                
                                # 创建统计摘要表
                                summary_rows = []
                                for defect_type in ['过检', '漏检', '正确检出']:
                                    type_stats = maxorg_stats[defect_type]
                                    if len(type_stats['ratios']) > 0:
                                        summary_rows.append({
                                            '缺陷类型': defect_type,
                                            '样本数': len(type_stats['ratios']),
                                            '均值': f"{type_stats['mean']:.4f}",
                                            '中位数': f"{type_stats['median']:.4f}",
                                            '标准差': f"{type_stats['std']:.4f}",
                                            '最小值': f"{type_stats['min']:.4f}",
                                            '最大值': f"{type_stats['max']:.4f}"
                                        })
                                
                                if summary_rows:
                                    summary_df = pd.DataFrame(summary_rows)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    # 提供CSV下载
                                    csv_maxorg = summary_df.to_csv(index=False, encoding='utf-8-sig')
                                    st.download_button(
                                        label=f"📥 下载 {casi_name}-{kla_name} MaxOrg比值统计（CSV）",
                                        data=csv_maxorg,
                                        file_name=f"maxorg_ratio_{casi_name}_{kla_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        key=f"download_maxorg_{idx}"
                                    )
                                    
                                    # 绘制箱线图对比
                                    st.write("**比值分布箱线图：**")
                                    fig_maxorg_box = go.Figure()
                                    
                                    for defect_type in ['过检', '漏检', '正确检出']:
                                        ratios = maxorg_stats[defect_type]['ratios']
                                        if len(ratios) > 0:
                                            fig_maxorg_box.add_trace(go.Box(
                                                y=ratios,
                                                name=f'{defect_type} (n={len(ratios)})',
                                                boxmean='sd'
                                            ))
                                    
                                    fig_maxorg_box.update_layout(
                                        title=f'{casi_name} vs {kla_name} - DW1O_MaxOrg/DW2O_MaxOrg 比值分布',
                                        yaxis_title='比值 (DW1O_MaxOrg / DW2O_MaxOrg)',
                                        height=500,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_maxorg_box, use_container_width=True)
                                    
                                    # 绘制直方图分布 - 分别显示
                                    st.write("**比值分布直方图（分别显示）：**")
                                    
                                    colors_map = {'过检': '#FF6B6B', '漏检': '#FFE66D', '正确检出': '#4ECDC4'}
                                    
                                    # 为每种类型单独绘制直方图
                                    cols_hist = st.columns(3)
                                    
                                    for col_idx, defect_type in enumerate(['过检', '漏检', '正确检出']):
                                        ratios = maxorg_stats[defect_type]['ratios']
                                        if len(ratios) > 0:
                                            with cols_hist[col_idx]:
                                                fig_hist_single = go.Figure()
                                                
                                                fig_hist_single.add_trace(go.Histogram(
                                                    x=ratios,
                                                    marker_color=colors_map[defect_type],
                                                    opacity=0.8,
                                                    nbinsx=30
                                                ))
                                                
                                                fig_hist_single.update_layout(
                                                    title=f'{defect_type}<br>(n={len(ratios)})',
                                                    xaxis_title='比值',
                                                    yaxis_title='频数',
                                                    height=400,
                                                    showlegend=False,
                                                    margin=dict(t=60, b=40, l=40, r=20)
                                                )
                                                
                                                st.plotly_chart(fig_hist_single, use_container_width=True)
                                    
                                    # 统计信息对比
                                    st.write("**详细统计对比：**")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.write("**过检：**")
                                        overdetect_stats = maxorg_stats['过检']
                                        if len(overdetect_stats['ratios']) > 0:
                                            st.metric("样本数", len(overdetect_stats['ratios']))
                                            st.metric("均值", f"{overdetect_stats['mean']:.4f}")
                                            st.metric("中位数", f"{overdetect_stats['median']:.4f}")
                                            st.metric("标准差", f"{overdetect_stats['std']:.4f}")
                                        else:
                                            st.info("无数据")
                                    
                                    with col2:
                                        st.write("**漏检：**")
                                        miss_stats = maxorg_stats['漏检']
                                        if len(miss_stats['ratios']) > 0:
                                            st.metric("样本数", len(miss_stats['ratios']))
                                            st.metric("均值", f"{miss_stats['mean']:.4f}")
                                            st.metric("中位数", f"{miss_stats['median']:.4f}")
                                            st.metric("标准差", f"{miss_stats['std']:.4f}")
                                        else:
                                            st.info("无数据")
                                    
                                    with col3:
                                        st.write("**正确检出：**")
                                        correct_stats = maxorg_stats['正确检出']
                                        if len(correct_stats['ratios']) > 0:
                                            st.metric("样本数", len(correct_stats['ratios']))
                                            st.metric("均值", f"{correct_stats['mean']:.4f}")
                                            st.metric("中位数", f"{correct_stats['median']:.4f}")
                                            st.metric("标准差", f"{correct_stats['std']:.4f}")
                                        else:
                                            st.info("无数据")
                                    
                                    st.write("---")
                                else:
                                    st.info(f"{casi_name} vs {kla_name}: 无有效的MaxOrg比值数据")
                            
                            # 总体对比（所有组合汇总）
                            st.write("### 📊 总体MaxOrg比值分布对比（所有组合汇总）")
                            
                            # 收集所有组合的数据
                            all_overdetect_ratios = []
                            all_miss_ratios = []
                            all_correct_ratios = []
                            
                            for result in all_match_results:
                                maxorg_stats = result.get('maxorg_ratio_stats', {})
                                if maxorg_stats.get('has_maxorg_data', False):
                                    all_overdetect_ratios.extend(maxorg_stats['过检']['ratios'])
                                    all_miss_ratios.extend(maxorg_stats['漏检']['ratios'])
                                    all_correct_ratios.extend(maxorg_stats['正确检出']['ratios'])
                            
                            if any([all_overdetect_ratios, all_miss_ratios, all_correct_ratios]):
                                # 总体箱线图
                                fig_overall_maxorg = go.Figure()
                                
                                if len(all_overdetect_ratios) > 0:
                                    fig_overall_maxorg.add_trace(go.Box(
                                        y=all_overdetect_ratios,
                                        name=f'过检 (n={len(all_overdetect_ratios)})',
                                        marker_color='#FF6B6B',
                                        boxmean='sd'
                                    ))
                                
                                if len(all_miss_ratios) > 0:
                                    fig_overall_maxorg.add_trace(go.Box(
                                        y=all_miss_ratios,
                                        name=f'漏检 (n={len(all_miss_ratios)})',
                                        marker_color='#FFE66D',
                                        boxmean='sd'
                                    ))
                                
                                if len(all_correct_ratios) > 0:
                                    fig_overall_maxorg.add_trace(go.Box(
                                        y=all_correct_ratios,
                                        name=f'正确检出 (n={len(all_correct_ratios)})',
                                        marker_color='#4ECDC4',
                                        boxmean='sd'
                                    ))
                                
                                fig_overall_maxorg.update_layout(
                                    title='总体DW1O_MaxOrg/DW2O_MaxOrg 比值分布对比',
                                    yaxis_title='比值 (DW1O_MaxOrg / DW2O_MaxOrg)',
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_overall_maxorg, use_container_width=True)
                                
                                # 总体统计摘要
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if len(all_overdetect_ratios) > 0:
                                        st.write("**总体过检统计：**")
                                        st.metric("样本数", len(all_overdetect_ratios))
                                        st.metric("均值", f"{np.mean(all_overdetect_ratios):.4f}")
                                        st.metric("中位数", f"{np.median(all_overdetect_ratios):.4f}")
                                        st.metric("标准差", f"{np.std(all_overdetect_ratios):.4f}")
                                
                                with col2:
                                    if len(all_miss_ratios) > 0:
                                        st.write("**总体漏检统计：**")
                                        st.metric("样本数", len(all_miss_ratios))
                                        st.metric("均值", f"{np.mean(all_miss_ratios):.4f}")
                                        st.metric("中位数", f"{np.median(all_miss_ratios):.4f}")
                                        st.metric("标准差", f"{np.std(all_miss_ratios):.4f}")
                                
                                with col3:
                                    if len(all_correct_ratios) > 0:
                                        st.write("**总体正确检出统计：**")
                                        st.metric("样本数", len(all_correct_ratios))
                                        st.metric("均值", f"{np.mean(all_correct_ratios):.4f}")
                                        st.metric("中位数", f"{np.median(all_correct_ratios):.4f}")
                                        st.metric("标准差", f"{np.std(all_correct_ratios):.4f}")
                        else:
                            st.info("未找到DW1O_MaxOrg和DW2O_MaxOrg列，无法进行比值分析")
                        
                        # 新增：MaxOrg=65532统计分析（过检和正确检出）
                        st.write("---")
                        st.markdown('<a name="MaxOrg65532统计"></a>', unsafe_allow_html=True)
                        st.subheader("📊 MaxOrg=65532情况统计（过检和正确检出）")
                        
                        with st.expander("🔍 查看MaxOrg=65532统计详情", expanded=False):
                            # 检查是否有任何组合包含MaxOrg 65532数据
                            has_any_65532_data = any(
                                result.get('maxorg_65532_stats', {}).get('has_maxorg_cols', False)
                                for result in st.session_state.kla_match_results
                            )
                        
                        if has_any_65532_data:
                            # 逐个组合显示
                            for result in st.session_state.kla_match_results:
                                casi_name = result['CASI文件夹']
                                kla_name = result['KLA文件夹']
                                stats_65532 = result.get('maxorg_65532_stats', {})
                                
                                if stats_65532.get('has_maxorg_cols', False):
                                    st.write(f"**{casi_name} vs {kla_name}**")
                                    
                                    # 创建两列布局：左边过检，右边正确检出
                                    col_over, col_correct = st.columns(2)
                                    
                                    with col_over:
                                        st.write("**过检缺陷：**")
                                        overdetect_stats = stats_65532['过检']
                                        if overdetect_stats['总数'] > 0:
                                            over_data = []
                                            over_data.append({'情况': '总数', '数量': overdetect_stats['总数'], '占比': '100.00%'})
                                            over_data.append({
                                                '情况': '三个都是65532',
                                                '数量': overdetect_stats['三个都是65532'],
                                                '占比': f"{overdetect_stats['三个都是65532']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': 'DW1O和DW2O是65532但DN1O不是',
                                                '数量': overdetect_stats['DW1O和DW2O是65532但DN1O不是'],
                                                '占比': f"{overdetect_stats['DW1O和DW2O是65532但DN1O不是']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': 'DW1O和DN1O是65532但DW2O不是',
                                                '数量': overdetect_stats['DW1O和DN1O是65532但DW2O不是'],
                                                '占比': f"{overdetect_stats['DW1O和DN1O是65532但DW2O不是']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': 'DW2O和DN1O是65532但DW1O不是',
                                                '数量': overdetect_stats['DW2O和DN1O是65532但DW1O不是'],
                                                '占比': f"{overdetect_stats['DW2O和DN1O是65532但DW1O不是']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': 'DW1O是65532但DW2O和DN1O不是',
                                                '数量': overdetect_stats['DW1O是65532但DW2O和DN1O不是'],
                                                '占比': f"{overdetect_stats['DW1O是65532但DW2O和DN1O不是']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': 'DW2O是65532但DW1O和DN1O不是',
                                                '数量': overdetect_stats['DW2O是65532但DW1O和DN1O不是'],
                                                '占比': f"{overdetect_stats['DW2O是65532但DW1O和DN1O不是']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': 'DN1O是65532但DW1O和DW2O不是',
                                                '数量': overdetect_stats['DN1O是65532但DW1O和DW2O不是'],
                                                '占比': f"{overdetect_stats['DN1O是65532但DW1O和DW2O不是']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            over_data.append({
                                                '情况': '都不是65532',
                                                '数量': overdetect_stats['都不是65532'],
                                                '占比': f"{overdetect_stats['都不是65532']/overdetect_stats['总数']*100:.2f}%"
                                            })
                                            
                                            over_df = pd.DataFrame(over_data)
                                            st.dataframe(over_df, use_container_width=True, hide_index=True)
                                        else:
                                            st.info("无过检数据")
                                    
                                    with col_correct:
                                        st.write("**正确检出缺陷：**")
                                        correct_stats = stats_65532['正确检出']
                                        if correct_stats['总数'] > 0:
                                            correct_data = []
                                            correct_data.append({'情况': '总数', '数量': correct_stats['总数'], '占比': '100.00%'})
                                            correct_data.append({
                                                '情况': '三个都是65532',
                                                '数量': correct_stats['三个都是65532'],
                                                '占比': f"{correct_stats['三个都是65532']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': 'DW1O和DW2O是65532但DN1O不是',
                                                '数量': correct_stats['DW1O和DW2O是65532但DN1O不是'],
                                                '占比': f"{correct_stats['DW1O和DW2O是65532但DN1O不是']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': 'DW1O和DN1O是65532但DW2O不是',
                                                '数量': correct_stats['DW1O和DN1O是65532但DW2O不是'],
                                                '占比': f"{correct_stats['DW1O和DN1O是65532但DW2O不是']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': 'DW2O和DN1O是65532但DW1O不是',
                                                '数量': correct_stats['DW2O和DN1O是65532但DW1O不是'],
                                                '占比': f"{correct_stats['DW2O和DN1O是65532但DW1O不是']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': 'DW1O是65532但DW2O和DN1O不是',
                                                '数量': correct_stats['DW1O是65532但DW2O和DN1O不是'],
                                                '占比': f"{correct_stats['DW1O是65532但DW2O和DN1O不是']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': 'DW2O是65532但DW1O和DN1O不是',
                                                '数量': correct_stats['DW2O是65532但DW1O和DN1O不是'],
                                                '占比': f"{correct_stats['DW2O是65532但DW1O和DN1O不是']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': 'DN1O是65532但DW1O和DW2O不是',
                                                '数量': correct_stats['DN1O是65532但DW1O和DW2O不是'],
                                                '占比': f"{correct_stats['DN1O是65532但DW1O和DW2O不是']/correct_stats['总数']*100:.2f}%"
                                            })
                                            correct_data.append({
                                                '情况': '都不是65532',
                                                '数量': correct_stats['都不是65532'],
                                                '占比': f"{correct_stats['都不是65532']/correct_stats['总数']*100:.2f}%"
                                            })
                                            
                                            correct_df = pd.DataFrame(correct_data)
                                            st.dataframe(correct_df, use_container_width=True, hide_index=True)
                                        else:
                                            st.info("无正确检出数据")
                                    
                                    st.write("---")
                            
                            # 总体汇总（所有组合）
                            st.write("### 📊 总体MaxOrg=65532统计（所有组合汇总）")
                            
                            # 汇总所有组合的数据
                            total_over_stats = {
                                '总数': 0,
                                '三个都是65532': 0,
                                'DW1O和DW2O是65532但DN1O不是': 0,
                                'DW1O是65532但DW2O和DN1O不是': 0,
                                'DW2O是65532但DW1O和DN1O不是': 0,
                                'DN1O是65532但DW1O和DW2O不是': 0,
                                'DW1O和DN1O是65532但DW2O不是': 0,
                                'DW2O和DN1O是65532但DW1O不是': 0,
                                '都不是65532': 0
                            }
                            
                            total_correct_stats = {
                                '总数': 0,
                                '三个都是65532': 0,
                                'DW1O和DW2O是65532但DN1O不是': 0,
                                'DW1O是65532但DW2O和DN1O不是': 0,
                                'DW2O是65532但DW1O和DN1O不是': 0,
                                'DN1O是65532但DW1O和DW2O不是': 0,
                                'DW1O和DN1O是65532但DW2O不是': 0,
                                'DW2O和DN1O是65532但DW1O不是': 0,
                                '都不是65532': 0
                            }
                            
                            for result in st.session_state.kla_match_results:
                                stats_65532 = result.get('maxorg_65532_stats', {})
                                if stats_65532.get('has_maxorg_cols', False):
                                    for key in total_over_stats.keys():
                                        total_over_stats[key] += stats_65532['过检'][key]
                                        total_correct_stats[key] += stats_65532['正确检出'][key]
                            
                            # 显示汇总表格
                            col_over_total, col_correct_total = st.columns(2)
                            
                            with col_over_total:
                                st.write("**总体过检统计：**")
                                if total_over_stats['总数'] > 0:
                                    total_over_data = []
                                    for situation, count in total_over_stats.items():
                                        if situation == '总数':
                                            total_over_data.append({'情况': situation, '数量': count, '占比': '100.00%'})
                                        else:
                                            total_over_data.append({
                                                '情况': situation,
                                                '数量': count,
                                                '占比': f"{count/total_over_stats['总数']*100:.2f}%"
                                            })
                                    
                                    total_over_df = pd.DataFrame(total_over_data)
                                    st.dataframe(total_over_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("无过检数据")
                            
                            with col_correct_total:
                                st.write("**总体正确检出统计：**")
                                if total_correct_stats['总数'] > 0:
                                    total_correct_data = []
                                    for situation, count in total_correct_stats.items():
                                        if situation == '总数':
                                            total_correct_data.append({'情况': situation, '数量': count, '占比': '100.00%'})
                                        else:
                                            total_correct_data.append({
                                                '情况': situation,
                                                '数量': count,
                                                '占比': f"{count/total_correct_stats['总数']*100:.2f}%"
                                            })
                                    
                                    total_correct_df = pd.DataFrame(total_correct_data)
                                    st.dataframe(total_correct_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("无正确检出数据")
                        else:
                            st.info("未找到DW1O_MaxOrg、DW2O_MaxOrg和DN1O_MaxOrg列，无法进行65532统计分析")
                    
                        # 新增：DW1O通道比值分析（SubRow1Max, SubRow2Max, MainRowMax）
                        st.write("---")
                        st.markdown('<a name="DW1O通道比值"></a>', unsafe_allow_html=True)
                        st.subheader("📊 DW1O通道比值分布分析")
                        
                        with st.expander("📈 查看DW1O通道比值分析详情", expanded=False):
                            # 检查是否有任何组合包含DW1O通道数据
                            has_any_dw1o = any(
                                result.get('dw1o_ratio_stats', {}).get('has_dw1o_data', False)
                                for result in st.session_state.kla_match_results
                            )
                        
                        if has_any_dw1o:
                            # 三个比值类型
                            ratio_names = ['SubRow1/SubRow2', 'MainRow/SubRow1', 'MainRow/SubRow2']
                            
                            # 为每个比值类型创建分析
                            for ratio_name in ratio_names:
                                st.write(f"### 📈 {ratio_name} 比值分析")
                                
                                # 逐个组合显示
                                for result in st.session_state.kla_match_results:
                                    casi_name = result['CASI文件夹']
                                    kla_name = result['KLA文件夹']
                                    dw1o_stats = result.get('dw1o_ratio_stats', {})
                                    
                                    if dw1o_stats.get('has_dw1o_data', False):
                                        st.write(f"**{casi_name} vs {kla_name}**")
                                        
                                        # 汇总表格
                                        summary_data = []
                                        for defect_type in ['过检', '漏检', '正确检出', 'KLA检出']:
                                            stats = dw1o_stats[defect_type][ratio_name]
                                            if len(stats['ratios']) > 0:
                                                summary_data.append({
                                                    '类型': defect_type,
                                                    '样本数': len(stats['ratios']),
                                                    '均值': f"{stats['mean']:.4f}",
                                                    '中位数': f"{stats['median']:.4f}",
                                                    '最小值': f"{stats['min']:.4f}",
                                                    '最大值': f"{stats['max']:.4f}",
                                                    '标准差': f"{stats['std']:.4f}"
                                                })
                                        
                                        if summary_data:
                                            summary_df = pd.DataFrame(summary_data)
                                            st.dataframe(summary_df, use_container_width=True)
                                            
                                            # 箱型图对比
                                            fig_box = go.Figure()
                                            colors = {'过检': '#FF6B6B', '漏检': '#4ECDC4', '正确检出': '#95E1D3', 'KLA检出': '#FFA07A'}
                                            
                                            for defect_type in ['过检', '漏检', '正确检出', 'KLA检出']:
                                                ratios = dw1o_stats[defect_type][ratio_name]['ratios']
                                                if len(ratios) > 0:
                                                    fig_box.add_trace(go.Box(
                                                        y=ratios,
                                                        name=defect_type,
                                                        marker_color=colors[defect_type],
                                                        boxmean='sd'
                                                    ))
                                            
                                            fig_box.update_layout(
                                                title=f'{ratio_name} 比值箱型图对比<br>{casi_name} vs {kla_name}',
                                                yaxis_title='比值',
                                                height=400,
                                                showlegend=True
                                            )
                                            
                                            st.plotly_chart(fig_box, use_container_width=True)
                                            
                                            # 分别列出直方图（4列：过检、漏检、正确检出、KLA检出）
                                            st.write("**各类型比值分布直方图：**")
                                            cols_hist = st.columns(4)
                                            
                                            for idx, defect_type in enumerate(['过检', '漏检', '正确检出', 'KLA检出']):
                                                with cols_hist[idx]:
                                                    ratios = dw1o_stats[defect_type][ratio_name]['ratios']
                                                    if len(ratios) > 0:
                                                        fig_hist_single = go.Figure()
                                                        fig_hist_single.add_trace(go.Histogram(
                                                            x=ratios,
                                                            nbinsx=30,
                                                            marker_color=colors[defect_type],
                                                            opacity=0.8,
                                                            name=defect_type
                                                        ))
                                                        
                                                        fig_hist_single.update_layout(
                                                            title=f'{defect_type}<br>(n={len(ratios)})',
                                                            xaxis_title='比值',
                                                            yaxis_title='频数',
                                                            height=400,
                                                            showlegend=False,
                                                            margin=dict(t=60, b=40, l=40, r=20)
                                                        )
                                                        
                                                        st.plotly_chart(fig_hist_single, use_container_width=True)
                                            
                                            # 统计信息对比
                                            st.write("**详细统计对比：**")
                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            with col1:
                                                st.write("**过检：**")
                                                stats = dw1o_stats['过检'][ratio_name]
                                                if len(stats['ratios']) > 0:
                                                    st.metric("样本数", len(stats['ratios']))
                                                    st.metric("均值", f"{stats['mean']:.4f}")
                                                    st.metric("中位数", f"{stats['median']:.4f}")
                                                    st.metric("标准差", f"{stats['std']:.4f}")
                                                else:
                                                    st.info("无数据")
                                            
                                            with col2:
                                                st.write("**漏检：**")
                                                stats = dw1o_stats['漏检'][ratio_name]
                                                if len(stats['ratios']) > 0:
                                                    st.metric("样本数", len(stats['ratios']))
                                                    st.metric("均值", f"{stats['mean']:.4f}")
                                                    st.metric("中位数", f"{stats['median']:.4f}")
                                                    st.metric("标准差", f"{stats['std']:.4f}")
                                                else:
                                                    st.info("无数据")
                                            
                                            with col3:
                                                st.write("**正确检出：**")
                                                stats = dw1o_stats['正确检出'][ratio_name]
                                                if len(stats['ratios']) > 0:
                                                    st.metric("样本数", len(stats['ratios']))
                                                    st.metric("均值", f"{stats['mean']:.4f}")
                                                    st.metric("中位数", f"{stats['median']:.4f}")
                                                    st.metric("标准差", f"{stats['std']:.4f}")
                                                else:
                                                    st.info("无数据")
                                            
                                            with col4:
                                                st.write("**KLA检出：**")
                                                stats = dw1o_stats['KLA检出'][ratio_name]
                                                if len(stats['ratios']) > 0:
                                                    st.metric("样本数", len(stats['ratios']))
                                                    st.metric("均值", f"{stats['mean']:.4f}")
                                                    st.metric("中位数", f"{stats['median']:.4f}")
                                                    st.metric("标准差", f"{stats['std']:.4f}")
                                                else:
                                                    st.info("无数据")
                                            
                                            st.write("---")
                                        else:
                                            st.info(f"{casi_name} vs {kla_name}: 无有效的{ratio_name}比值数据")
                                
                                # 总体对比（所有组合汇总）
                                st.write(f"#### 📊 总体{ratio_name}比值分布对比（所有组合汇总）")
                                
                                # 收集所有组合的数据
                                all_overdetect_ratios = []
                                all_miss_ratios = []
                                all_correct_ratios = []
                                all_kla_ratios = []
                                
                                for result in st.session_state.kla_match_results:
                                    dw1o_stats = result.get('dw1o_ratio_stats', {})
                                    if dw1o_stats.get('has_dw1o_data', False):
                                        all_overdetect_ratios.extend(dw1o_stats['过检'][ratio_name]['ratios'])
                                        all_miss_ratios.extend(dw1o_stats['漏检'][ratio_name]['ratios'])
                                        all_correct_ratios.extend(dw1o_stats['正确检出'][ratio_name]['ratios'])
                                        all_kla_ratios.extend(dw1o_stats['KLA检出'][ratio_name]['ratios'])
                                
                                if all_overdetect_ratios or all_miss_ratios or all_correct_ratios or all_kla_ratios:
                                    # 箱型图
                                    fig_overall_box = go.Figure()
                                    colors = {'过检': '#FF6B6B', '漏检': '#4ECDC4', '正确检出': '#95E1D3', 'KLA检出': '#FFA07A'}
                                    
                                    if all_overdetect_ratios:
                                        fig_overall_box.add_trace(go.Box(
                                            y=all_overdetect_ratios,
                                            name='过检',
                                            marker_color=colors['过检'],
                                            boxmean='sd'
                                        ))
                                    
                                    if all_miss_ratios:
                                        fig_overall_box.add_trace(go.Box(
                                            y=all_miss_ratios,
                                            name='漏检',
                                            marker_color=colors['漏检'],
                                            boxmean='sd'
                                        ))
                                    
                                    if all_correct_ratios:
                                        fig_overall_box.add_trace(go.Box(
                                            y=all_correct_ratios,
                                            name='正确检出',
                                            marker_color=colors['正确检出'],
                                            boxmean='sd'
                                        ))
                                    
                                    if all_kla_ratios:
                                        fig_overall_box.add_trace(go.Box(
                                            y=all_kla_ratios,
                                            name='KLA检出',
                                            marker_color=colors['KLA检出'],
                                            boxmean='sd'
                                        ))
                                    
                                    fig_overall_box.update_layout(
                                        title=f'总体{ratio_name}比值箱型图对比（所有组合汇总）',
                                        yaxis_title='比值',
                                        height=400,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_overall_box, use_container_width=True)
                                    
                                    # 总体统计摘要
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        if len(all_overdetect_ratios) > 0:
                                            st.write("**总体过检统计：**")
                                            st.metric("样本数", len(all_overdetect_ratios))
                                            st.metric("均值", f"{np.mean(all_overdetect_ratios):.4f}")
                                            st.metric("中位数", f"{np.median(all_overdetect_ratios):.4f}")
                                            st.metric("标准差", f"{np.std(all_overdetect_ratios):.4f}")
                                    
                                    with col2:
                                        if len(all_miss_ratios) > 0:
                                            st.write("**总体漏检统计：**")
                                            st.metric("样本数", len(all_miss_ratios))
                                            st.metric("均值", f"{np.mean(all_miss_ratios):.4f}")
                                            st.metric("中位数", f"{np.median(all_miss_ratios):.4f}")
                                            st.metric("标准差", f"{np.std(all_miss_ratios):.4f}")
                                    
                                    with col3:
                                        if len(all_correct_ratios) > 0:
                                            st.write("**总体正确检出统计：**")
                                            st.metric("样本数", len(all_correct_ratios))
                                            st.metric("均值", f"{np.mean(all_correct_ratios):.4f}")
                                            st.metric("中位数", f"{np.median(all_correct_ratios):.4f}")
                                            st.metric("标准差", f"{np.std(all_correct_ratios):.4f}")
                                    
                                    with col4:
                                        if len(all_kla_ratios) > 0:
                                            st.write("**总体KLA检出统计：**")
                                            st.metric("样本数", len(all_kla_ratios))
                                            st.metric("均值", f"{np.mean(all_kla_ratios):.4f}")
                                            st.metric("中位数", f"{np.median(all_kla_ratios):.4f}")
                                            st.metric("标准差", f"{np.std(all_kla_ratios):.4f}")
                                
                                st.write("---")
                        else:
                            st.info("未找到DW1O_SubRow1Max、DW1O_SubRow2Max和DW1O_MainRowMax列，无法进行比值分析")
                    
                        # 新增：BGMean值分布分析（过检和正确检出，去除0值）
                        st.write("---")
                        st.markdown('<a name="BGMean值分布"></a>', unsafe_allow_html=True)
                        st.subheader("📊 BGMean值分布分析（过检和正确检出）")
                        
                        with st.expander("💡 查看BGMean值分布分析详情", expanded=False):
                            st.markdown("""
                            分析过检和正确检出缺陷的DW1O_BGMean、DW2O_BGMean和DN1O_BGMean值分布。
                            - **BGMean**: 背景均值，反映缺陷周围的背景灰度水平
                            - **已去除值为0的数据**
                            - 仅分析过检和正确检出两种类型，不包括漏检
                            - 帮助了解不同检出状态下的背景特征差异
                            """)
                        
                        # 检查是否有BGMean数据
                        has_any_bgmean_data = any(
                            result.get('bgmean_stats', {}).get('has_bgmean_data', False)
                            for result in st.session_state.kla_match_results
                        )
                        
                        if has_any_bgmean_data:
                            # 三个BGMean通道
                            bgmean_channels = ['DW1O_BGMean', 'DW2O_BGMean', 'DN1O_BGMean']
                            
                            # 为每个BGMean通道创建分析
                            for bgmean_channel in bgmean_channels:
                                st.write(f"### 📈 {bgmean_channel} 值分析")
                                
                                # 逐个组合显示
                                for result in st.session_state.kla_match_results:
                                    casi_name = result['CASI文件夹']
                                    kla_name = result['KLA文件夹']
                                    bgmean_stats = result.get('bgmean_stats', {})
                                    
                                    if bgmean_stats.get('has_bgmean_data', False):
                                        st.write(f"**{casi_name} vs {kla_name}**")
                                        
                                        # 汇总表格
                                        summary_data = []
                                        for defect_type in ['过检', '正确检出']:
                                            stats = bgmean_stats[defect_type][bgmean_channel]
                                            if len(stats['values']) > 0:
                                                summary_data.append({
                                                    '类型': defect_type,
                                                    '样本数': len(stats['values']),
                                                    '均值': f"{stats['mean']:.2f}",
                                                    '中位数': f"{stats['median']:.2f}",
                                                    '最小值': f"{stats['min']:.2f}",
                                                    '最大值': f"{stats['max']:.2f}",
                                                    '标准差': f"{stats['std']:.2f}"
                                                })
                                        
                                        if summary_data:
                                            summary_df = pd.DataFrame(summary_data)
                                            st.dataframe(summary_df, use_container_width=True)
                                            
                                            # 箱型图对比
                                            fig_box = go.Figure()
                                            colors = {'过检': '#FF6B6B', '正确检出': '#4ECDC4'}
                                            
                                            for defect_type in ['过检', '正确检出']:
                                                values = bgmean_stats[defect_type][bgmean_channel]['values']
                                                if len(values) > 0:
                                                    fig_box.add_trace(go.Box(
                                                        y=values,
                                                        name=defect_type,
                                                        marker_color=colors[defect_type],
                                                        boxmean='sd'
                                                    ))
                                            
                                            fig_box.update_layout(
                                                title=f'{bgmean_channel} 值箱型图对比<br>{casi_name} vs {kla_name}',
                                                yaxis_title=bgmean_channel,
                                                height=400,
                                                showlegend=True
                                            )
                                            
                                            st.plotly_chart(fig_box, use_container_width=True)
                                            
                                            # 直方图分布（两列：过检和正确检出）
                                            st.write("**各类型值分布直方图：**")
                                            cols_hist = st.columns(2)
                                            
                                            for idx, defect_type in enumerate(['过检', '正确检出']):
                                                with cols_hist[idx]:
                                                    values = bgmean_stats[defect_type][bgmean_channel]['values']
                                                    if len(values) > 0:
                                                        fig_hist_single = go.Figure()
                                                        fig_hist_single.add_trace(go.Histogram(
                                                            x=values,
                                                            nbinsx=30,
                                                            marker_color=colors[defect_type],
                                                            opacity=0.8,
                                                            name=defect_type
                                                        ))
                                                        
                                                        fig_hist_single.update_layout(
                                                            title=f'{defect_type}<br>(n={len(values)})',
                                                            xaxis_title=bgmean_channel,
                                                            yaxis_title='频数',
                                                            height=400,
                                                            showlegend=False,
                                                            margin=dict(t=60, b=40, l=40, r=20)
                                                        )
                                                        
                                                        st.plotly_chart(fig_hist_single, use_container_width=True)
                                            
                                            # 统计信息对比
                                            st.write("**详细统计对比：**")
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.write("**过检：**")
                                                stats = bgmean_stats['过检'][bgmean_channel]
                                                if len(stats['values']) > 0:
                                                    st.metric("样本数", len(stats['values']))
                                                    st.metric("均值", f"{stats['mean']:.2f}")
                                                    st.metric("中位数", f"{stats['median']:.2f}")
                                                    st.metric("标准差", f"{stats['std']:.2f}")
                                                else:
                                                    st.info("无数据")
                                            
                                            with col2:
                                                st.write("**正确检出：**")
                                                stats = bgmean_stats['正确检出'][bgmean_channel]
                                                if len(stats['values']) > 0:
                                                    st.metric("样本数", len(stats['values']))
                                                    st.metric("均值", f"{stats['mean']:.2f}")
                                                    st.metric("中位数", f"{stats['median']:.2f}")
                                                    st.metric("标准差", f"{stats['std']:.2f}")
                                                else:
                                                    st.info("无数据")
                                            
                                            st.write("---")
                                        else:
                                            st.info(f"{casi_name} vs {kla_name}: 无有效的{bgmean_channel}数据")
                                
                                # 总体对比（所有组合汇总）
                                st.write(f"#### 📊 总体{bgmean_channel}值分布对比（所有组合汇总）")
                                
                                # 收集所有组合的数据
                                all_overdetect_values = []
                                all_correct_values = []
                                
                                for result in st.session_state.kla_match_results:
                                    bgmean_stats = result.get('bgmean_stats', {})
                                    if bgmean_stats.get('has_bgmean_data', False):
                                        all_overdetect_values.extend(bgmean_stats['过检'][bgmean_channel]['values'])
                                        all_correct_values.extend(bgmean_stats['正确检出'][bgmean_channel]['values'])
                                
                                if all_overdetect_values or all_correct_values:
                                    # 箱型图
                                    fig_overall_box = go.Figure()
                                    colors = {'过检': '#FF6B6B', '正确检出': '#4ECDC4'}
                                    
                                    if all_overdetect_values:
                                        fig_overall_box.add_trace(go.Box(
                                            y=all_overdetect_values,
                                            name=f'过检 (n={len(all_overdetect_values)})',
                                            marker_color=colors['过检'],
                                            boxmean='sd'
                                        ))
                                    
                                    if all_correct_values:
                                        fig_overall_box.add_trace(go.Box(
                                            y=all_correct_values,
                                            name=f'正确检出 (n={len(all_correct_values)})',
                                            marker_color=colors['正确检出'],
                                            boxmean='sd'
                                        ))
                                    
                                    fig_overall_box.update_layout(
                                        title=f'总体{bgmean_channel}值箱型图对比（所有组合汇总）',
                                        yaxis_title=bgmean_channel,
                                        height=400,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_overall_box, use_container_width=True)
                                    
                                    # 总体统计摘要
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if len(all_overdetect_values) > 0:
                                            st.write("**总体过检统计：**")
                                            st.metric("样本数", len(all_overdetect_values))
                                            st.metric("均值", f"{np.mean(all_overdetect_values):.2f}")
                                            st.metric("中位数", f"{np.median(all_overdetect_values):.2f}")
                                            st.metric("标准差", f"{np.std(all_overdetect_values):.2f}")
                                    
                                    with col2:
                                        if len(all_correct_values) > 0:
                                            st.write("**总体正确检出统计：**")
                                            st.metric("样本数", len(all_correct_values))
                                            st.metric("均值", f"{np.mean(all_correct_values):.2f}")
                                            st.metric("中位数", f"{np.median(all_correct_values):.2f}")
                                            st.metric("标准差", f"{np.std(all_correct_values):.2f}")
                                
                                st.write("---")
                            
                            # 新增：所有文件BGMean数据汇总表格
                            st.write("---")
                            st.markdown('<a name="BGMean汇总表"></a>', unsafe_allow_html=True)
                            st.subheader("📋 BGMean数据汇总表格（所有文件）")
                            
                            st.markdown("""
                            汇总所有CASI-KLA组合的BGMean统计数据，便于对比分析。
                            - 包含过检和正确检出两种类型
                            - 显示DW1O_BGMean、DW2O_BGMean、DN1O_BGMean的统计值
                            - 已排除值为0的数据
                            """)
                            
                            # 创建汇总数据列表
                            summary_table_data = []
                            
                            for result in st.session_state.kla_match_results:
                                casi_name = result['CASI文件夹']
                                kla_name = result['KLA文件夹']
                                bgmean_stats = result.get('bgmean_stats', {})
                                
                                if bgmean_stats.get('has_bgmean_data', False):
                                    # 对每种缺陷类型创建一行
                                    for defect_type in ['过检', '正确检出']:
                                        row = {
                                            'CASI文件夹': casi_name,
                                            'KLA文件夹': kla_name,
                                            '缺陷类型': defect_type
                                        }
                                        
                                        # 添加DW1O_BGMean统计
                                        dw1o_stats = bgmean_stats[defect_type].get('DW1O_BGMean', {})
                                        if dw1o_stats and len(dw1o_stats.get('values', [])) > 0:
                                            row['DW1O_样本数'] = len(dw1o_stats['values'])
                                            row['DW1O_均值'] = round(dw1o_stats['mean'], 2)
                                            row['DW1O_中位数'] = round(dw1o_stats['median'], 2)
                                            row['DW1O_最小值'] = round(dw1o_stats['min'], 2)
                                            row['DW1O_最大值'] = round(dw1o_stats['max'], 2)
                                            row['DW1O_标准差'] = round(dw1o_stats['std'], 2)
                                        else:
                                            row['DW1O_样本数'] = 0
                                            row['DW1O_均值'] = '-'
                                            row['DW1O_中位数'] = '-'
                                            row['DW1O_最小值'] = '-'
                                            row['DW1O_最大值'] = '-'
                                            row['DW1O_标准差'] = '-'
                                        
                                        # 添加DW2O_BGMean统计
                                        dw2o_stats = bgmean_stats[defect_type].get('DW2O_BGMean', {})
                                        if dw2o_stats and len(dw2o_stats.get('values', [])) > 0:
                                            row['DW2O_样本数'] = len(dw2o_stats['values'])
                                            row['DW2O_均值'] = round(dw2o_stats['mean'], 2)
                                            row['DW2O_中位数'] = round(dw2o_stats['median'], 2)
                                            row['DW2O_最小值'] = round(dw2o_stats['min'], 2)
                                            row['DW2O_最大值'] = round(dw2o_stats['max'], 2)
                                            row['DW2O_标准差'] = round(dw2o_stats['std'], 2)
                                        else:
                                            row['DW2O_样本数'] = 0
                                            row['DW2O_均值'] = '-'
                                            row['DW2O_中位数'] = '-'
                                            row['DW2O_最小值'] = '-'
                                            row['DW2O_最大值'] = '-'
                                            row['DW2O_标准差'] = '-'
                                        
                                        # 添加DN1O_BGMean统计
                                        dn1o_stats = bgmean_stats[defect_type].get('DN1O_BGMean', {})
                                        if dn1o_stats and len(dn1o_stats.get('values', [])) > 0:
                                            row['DN1O_样本数'] = len(dn1o_stats['values'])
                                            row['DN1O_均值'] = round(dn1o_stats['mean'], 2)
                                            row['DN1O_中位数'] = round(dn1o_stats['median'], 2)
                                            row['DN1O_最小值'] = round(dn1o_stats['min'], 2)
                                            row['DN1O_最大值'] = round(dn1o_stats['max'], 2)
                                            row['DN1O_标准差'] = round(dn1o_stats['std'], 2)
                                        else:
                                            row['DN1O_样本数'] = 0
                                            row['DN1O_均值'] = '-'
                                            row['DN1O_中位数'] = '-'
                                            row['DN1O_最小值'] = '-'
                                            row['DN1O_最大值'] = '-'
                                            row['DN1O_标准差'] = '-'
                                        
                                        summary_table_data.append(row)
                            
                            if summary_table_data:
                                # 创建汇总DataFrame
                                summary_table_df = pd.DataFrame(summary_table_data)
                                
                                # 重新排列列的顺序
                                columns_order = [
                                    'CASI文件夹', 'KLA文件夹', '缺陷类型',
                                    'DW1O_样本数', 'DW1O_均值', 'DW1O_中位数', 'DW1O_最小值', 'DW1O_最大值', 'DW1O_标准差',
                                    'DW2O_样本数', 'DW2O_均值', 'DW2O_中位数', 'DW2O_最小值', 'DW2O_最大值', 'DW2O_标准差',
                                    'DN1O_样本数', 'DN1O_均值', 'DN1O_中位数', 'DN1O_最小值', 'DN1O_最大值', 'DN1O_标准差'
                                ]
                                
                                # 只保留存在的列
                                columns_order = [col for col in columns_order if col in summary_table_df.columns]
                                summary_table_df = summary_table_df[columns_order]
                                
                                # 显示汇总表格
                                st.write("### 📊 BGMean统计汇总表")
                                st.dataframe(summary_table_df, use_container_width=True, height=400)
                                
                                # 提供CSV下载
                                csv_summary = summary_table_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 下载BGMean汇总表格（CSV）",
                                    data=csv_summary,
                                    file_name=f"bgmean_summary_all_files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="下载包含所有文件BGMean统计数据的汇总表格"
                                )
                                
                                # 统计摘要
                                st.write("### 📈 汇总统计")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("组合总数", len(st.session_state.kla_match_results))
                                    st.metric("数据行数", len(summary_table_df))
                                
                                with col2:
                                    overdetect_rows = summary_table_df[summary_table_df['缺陷类型'] == '过检']
                                    st.metric("过检数据行", len(overdetect_rows))
                                    if len(overdetect_rows) > 0:
                                        total_samples = overdetect_rows[['DW1O_样本数', 'DW2O_样本数', 'DN1O_样本数']].sum().sum()
                                        st.caption(f"过检总样本数: {total_samples}")
                                
                                with col3:
                                    correct_rows = summary_table_df[summary_table_df['缺陷类型'] == '正确检出']
                                    st.metric("正确检出数据行", len(correct_rows))
                                    if len(correct_rows) > 0:
                                        total_samples = correct_rows[['DW1O_样本数', 'DW2O_样本数', 'DN1O_样本数']].sum().sum()
                                        st.caption(f"正检总样本数: {total_samples}")
                                
                                st.info("💡 提示：表格中'-'表示该项无有效数据。所有统计值已排除BGMean为0的数据。")
                            else:
                                st.warning("没有可汇总的BGMean数据")
                        else:
                            st.info("未找到DW1O_BGMean、DW2O_BGMean或DN1O_BGMean列，无法进行BGMean分析")
                    
                        # 新增：TotalSNR按尺寸分布分析
                        st.write("---")
                        st.markdown('<a name="TotalSNR尺寸分布"></a>', unsafe_allow_html=True)
                        st.subheader("📊 TotalSNR按尺寸分布分析")
                        
                        with st.expander("🔍 查看TotalSNR按尺寸分布分析详情", expanded=False):
                            st.markdown("""
                            分析过检和正确检出缺陷的各通道TotalSNR值在不同尺寸区间的分布。
                            - **尺寸区间**：从26nm开始，每2nm一个区间（26-28, 28-30, ...）
                            - **通道**：DW1O_TotalSNR, DW2O_TotalSNR, DN1O_TotalSNR
                            - **缺陷类型**：过检和正确检出
                            - **尺寸判断**：使用DW1O_Size作为主要判断（如无则使用DW2O或DN1O）
                            """)
                        
                        # 检查是否有任何组合包含SNR数据
                        has_any_snr_data = any(
                            result.get('totalsnr_size_stats', {}).get('has_snr_data', False)
                            for result in st.session_state.kla_match_results
                        )
                        
                        if has_any_snr_data:
                            # 为每个CASI-KLA组合生成分析
                            for idx, result in enumerate(st.session_state.kla_match_results):
                                casi_name = result['CASI文件夹']
                                kla_name = result['KLA文件夹']
                                snr_stats = result.get('totalsnr_size_stats', {})
                                
                                if not snr_stats.get('has_snr_data', False):
                                    continue
                                
                                st.write(f"### {casi_name} vs {kla_name}")
                                
                                # 为每种缺陷类型和通道生成箱线图
                                for defect_type in ['过检', '正确检出']:
                                    st.write(f"#### {defect_type}")
                                    
                                    defect_data = snr_stats[defect_type]
                                    size_bins = snr_stats['size_bins']
                                    
                                    # 为每个通道创建箱线图
                                    for channel in ['DW1O_TotalSNR', 'DW2O_TotalSNR', 'DN1O_TotalSNR']:
                                        st.write(f"**{channel} 分布：**")
                                        
                                        # 收集有数据的尺寸区间
                                        plot_data = []
                                        for size_bin in size_bins:
                                            if size_bin in defect_data:
                                                snr_values = defect_data[size_bin][channel]
                                                if len(snr_values) > 0:
                                                    plot_data.append({
                                                        'size_bin': f"{size_bin}-{size_bin+2}nm",
                                                        'size_bin_num': size_bin,
                                                        'values': snr_values
                                                    })
                                        
                                        if len(plot_data) > 0:
                                            fig_snr = go.Figure()
                                            
                                            for item in plot_data:
                                                fig_snr.add_trace(go.Box(
                                                    y=item['values'],
                                                    name=item['size_bin'],
                                                    boxmean='sd'
                                                ))
                                            
                                            fig_snr.update_layout(
                                                title=f'{channel} - {defect_type} ({casi_name} vs {kla_name})',
                                                xaxis_title='尺寸区间',
                                                yaxis_title=channel,
                                                height=500,
                                                showlegend=True
                                            )
                                            
                                            st.plotly_chart(fig_snr, use_container_width=True)
                                            
                                            # 显示统计表格
                                            stat_rows = []
                                            for item in plot_data:
                                                values = item['values']
                                                stat_rows.append({
                                                    '尺寸区间': item['size_bin'],
                                                    '样本数': len(values),
                                                    '均值': f"{np.mean(values):.2f}",
                                                    '中位数': f"{np.median(values):.2f}",
                                                    '标准差': f"{np.std(values):.2f}",
                                                    '最小值': f"{np.min(values):.2f}",
                                                    '最大值': f"{np.max(values):.2f}"
                                                })
                                            
                                            stat_df = pd.DataFrame(stat_rows)
                                            st.dataframe(stat_df, use_container_width=True)
                                        else:
                                            st.info(f"无{channel}数据")
                                
                                # 生成晶圆图：按尺寸区间显示过检和漏检分布
                                st.write("#### 🗺️ 按尺寸区间的晶圆缺陷分布图")
                                
                                # 选择尺寸区间
                                available_bins = []
                                for size_bin in snr_stats['size_bins']:
                                    total_count_over = snr_stats['过检'].get(size_bin, {}).get('count', 0)
                                    total_count_correct = snr_stats['正确检出'].get(size_bin, {}).get('count', 0)
                                    if total_count_over > 0 or total_count_correct > 0:
                                        available_bins.append(size_bin)
                                
                                if len(available_bins) > 0:
                                    # 使用expander来组织晶圆图，避免页面过长
                                    with st.expander(f"🗺️ 查看晶圆缺陷分布图（共{len(available_bins)}个尺寸区间可选）", expanded=True):
                                        st.info("💡 提示：下方默认显示前3个尺寸区间的晶圆图。如需查看更多，请展开对应的区间。")
                                        
                                        # 为每个可用的尺寸区间创建一个expander
                                        for bin_idx, size_bin in enumerate(available_bins):
                                            # 前3个默认展开，其余默认折叠
                                            is_expanded = bin_idx < 3
                                            
                                            with st.expander(f"📍 尺寸区间：{size_bin}-{size_bin+2}nm", expanded=is_expanded):
                                                # 收集该尺寸区间的过检和正确检出数据
                                                over_coords = snr_stats['过检'].get(size_bin, {}).get('coords', [])
                                                correct_coords = snr_stats['正确检出'].get(size_bin, {}).get('coords', [])
                                                
                                                if len(over_coords) == 0 and len(correct_coords) == 0:
                                                    st.info(f"该尺寸区间无数据")
                                                    continue
                                                
                                                # 创建晶圆图
                                                fig_wafer = go.Figure()
                                                
                                                # 添加晶圆边缘圆形（以150000, 150000为中心）
                                                wafer_center_x = 150000
                                                wafer_center_y = 150000
                                                wafer_radius = 150000  # 晶圆半径
                                                
                                                # 生成圆形轮廓点
                                                theta = np.linspace(0, 2*np.pi, 100)
                                                circle_x = wafer_center_x + wafer_radius * np.cos(theta)
                                                circle_y = wafer_center_y + wafer_radius * np.sin(theta)
                                                
                                                fig_wafer.add_trace(go.Scatter(
                                                    x=circle_x,
                                                    y=circle_y,
                                                    mode='lines',
                                                    name='晶圆边缘',
                                                    line=dict(color='gray', width=2, dash='dash'),
                                                    hoverinfo='skip',
                                                    showlegend=True
                                                ))
                                                
                                                # 添加过检点
                                                if len(over_coords) > 0:
                                                    x_over = [c['x'] for c in over_coords]
                                                    y_over = [c['y'] for c in over_coords]
                                                    hover_text_over = [
                                                        f"过检<br>X: {c['x']:.1f}<br>Y: {c['y']:.1f}<br>" +
                                                        f"DW1O_Size: {c['dw1o_size']:.1f}<br>" +
                                                        f"DW2O_Size: {c['dw2o_size']:.1f}<br>" +
                                                        f"DN1O_Size: {c['dn1o_size']:.1f}<br>" +
                                                        f"DW1O_SNR: {c.get('dw1o_snr', 0):.2f}<br>" +
                                                        f"DW2O_SNR: {c.get('dw2o_snr', 0):.2f}<br>" +
                                                        f"DN1O_SNR: {c.get('dn1o_snr', 0):.2f}"
                                                        for c in over_coords
                                                    ]
                                                    
                                                    fig_wafer.add_trace(go.Scatter(
                                                        x=x_over,
                                                        y=y_over,
                                                        mode='markers',
                                                        name=f'过检 (n={len(over_coords)})',
                                                        marker=dict(size=8, color='red', opacity=0.6),
                                                        text=hover_text_over,
                                                        hovertemplate='%{text}<extra></extra>'
                                                    ))
                                                
                                                # 添加正确检出点
                                                if len(correct_coords) > 0:
                                                    x_correct = [c['x'] for c in correct_coords]
                                                    y_correct = [c['y'] for c in correct_coords]
                                                    hover_text_correct = [
                                                        f"正确检出<br>X: {c['x']:.1f}<br>Y: {c['y']:.1f}<br>" +
                                                        f"DW1O_Size: {c['dw1o_size']:.1f}<br>" +
                                                        f"DW2O_Size: {c['dw2o_size']:.1f}<br>" +
                                                        f"DN1O_Size: {c['dn1o_size']:.1f}<br>" +
                                                        f"DW1O_SNR: {c.get('dw1o_snr', 0):.2f}<br>" +
                                                        f"DW2O_SNR: {c.get('dw2o_snr', 0):.2f}<br>" +
                                                        f"DN1O_SNR: {c.get('dn1o_snr', 0):.2f}"
                                                        for c in correct_coords
                                                    ]
                                                    
                                                    fig_wafer.add_trace(go.Scatter(
                                                        x=x_correct,
                                                        y=y_correct,
                                                        mode='markers',
                                                        name=f'正确检出 (n={len(correct_coords)})',
                                                        marker=dict(size=8, color='green', opacity=0.6),
                                                        text=hover_text_correct,
                                                        hovertemplate='%{text}<extra></extra>'
                                                    ))
                                                
                                                fig_wafer.update_layout(
                                                    title=f'晶圆缺陷分布图 - 尺寸 {size_bin}-{size_bin+2}nm<br>{casi_name} vs {kla_name}',
                                                    xaxis_title='X坐标 (dCenterXCartisian)',
                                                    yaxis_title='Y坐标 (dCenterYCartisian)',
                                                    height=600,
                                                    hovermode='closest',
                                                    showlegend=True,
                                                    xaxis=dict(scaleanchor="y", scaleratio=1),
                                                    yaxis=dict(scaleanchor="x", scaleratio=1)
                                                )
                                                
                                                st.plotly_chart(fig_wafer, use_container_width=True)
                                                
                                                # 显示统计信息
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric("过检数量", len(over_coords))
                                                with col2:
                                                    st.metric("正确检出数量", len(correct_coords))
                                else:
                                    st.info("没有可用的尺寸区间数据")
                                
                                st.write("---")
                        else:
                            st.info("未找到TotalSNR和Size列，无法进行SNR按尺寸分布分析")
                    
                    else:
                        st.warning("未生成匹配结果")
                        
            except Exception as e:
                st.error(f"KLA匹配分析时出错: {str(e)}")
                st.exception(e)
    
    elif kla_match_folder:
        st.error("文件夹路径不存在，请检查路径是否正确")
    
    # 新增：CASI坐标共有率分析（基于匹配结果）
    st.write("---")
    st.markdown('<a name="共有率分析"></a>', unsafe_allow_html=True)
    st.header("🔍 CASI缺陷坐标共有率分析")
    
    st.markdown("""
    ### 功能说明
    基于CASI与KLA匹配分析结果，分析多个子文件夹间CASI缺陷的位置一致性（共有率）。
    
    **重要说明：**
    - **前置条件：** 需要先执行"CASI与KLA匹配分析"
    - **分析对象：** 仅分析CASI数据（有nDefectID），不包括KLA漏检数据
    - **匹配范围：** 200nm（可调整）
    - **匹配类型：** 按过检、正确检出、漏检分别统计
    - **统计方式：** 计算每个位置在多个文件夹中的出现次数
    
    **分析内容：**
    1. 各子文件夹过检/正确检出/漏检的位置重叠情况（CASI数据）
    2. 共有的位置占各子文件夹的百分比
    3. 可视化展示共有位置分布
    4. 导出nDefectID对应关系及完整特征数据
    
    **数据过滤：**
    - ✅ 包含：所有有nDefectID的CASI缺陷数据
    - ❌ 排除：KLA漏检数据（无nDefectID，无法进行特征对比）
    """)
    
    # 检查是否有匹配结果数据
    if 'kla_match_results' not in st.session_state or not st.session_state.kla_match_results:
        st.warning("⚠️ 请先执行上方的 '🔍 CASI与KLA匹配分析' 以生成匹配数据")
    else:
        # 分析参数
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            cohesion_threshold = st.number_input("匹配距离阈值（nm）", value=200.0, min_value=10.0, max_value=1000.0,
                                                help="两个缺陷之间的最大距离，小于此距离视为同一位置")
        with col_param2:
            min_occurrence = st.number_input("最小出现次数", value=2, min_value=2, max_value=10,
                                            help="至少在N个文件夹中出现才统计为共有位置")
        
        if st.button("🔍 开始共有率分析", type="primary", key="cohesion_analysis_btn"):
            try:
                # 从 session_state 获取匹配结果
                all_match_results = st.session_state.kla_match_results
                st.info(f"基于 {len(all_match_results)} 个匹配结果进行分析")
                
                # 从all_match_results中提取数据
                folder_defects = {}
                
                for result in all_match_results:
                    casi_folder = result['CASI文件夹']
                    
                    # 检查是否有coord_data
                    if 'coord_data' in result:
                        coord_data = result['coord_data']
                        
                        if casi_folder not in folder_defects:
                            folder_defects[casi_folder] = {
                                '过检': [],
                                '正确检出': [],
                                '漏检': []
                            }
                        
                        # 合并坐标数据
                        for defect_type in ['过检', '正确检出', '漏检']:
                            folder_defects[casi_folder][defect_type].extend(coord_data.get(defect_type, []))
                
                if len(folder_defects) < 2:
                    st.error("至少需要2个CASI文件夹进行共有率分析")
                else:
                    st.success(f"成功读取 {len(folder_defects)} 个文件夹的数据")
                    
                    # 定义颜色列表（用于可视化）
                    colors = [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AED6F1',
                        '#A9DFBF', '#F9E79F', '#D7BDE2', '#A2D9CE', '#FAD7A0'
                    ]
                    
                    # 对每种类型进行共有率分析
                    defect_types = ['过检', '正确检出', '漏检']
                    
                    # 新增：收集所有文件夹的所有缺陷类型数据（用于不共有缺陷分析）
                    all_points_by_defect_type = {}
                    for dt in defect_types:
                        all_points_by_defect_type[dt] = {}
                        for folder, defects in folder_defects.items():
                            if dt in defects and len(defects[dt]) > 0:
                                all_points_by_defect_type[dt][folder] = defects[dt]
                    
                    for defect_type in defect_types:
                        st.write("---")
                        st.subheader(f"📊 {defect_type} 共有率分析")
                        
                        # 收集所有点
                        all_points_by_folder = {}
                        for folder, defects in folder_defects.items():
                            if defect_type in defects and len(defects[defect_type]) > 0:
                                all_points_by_folder[folder] = np.array(defects[defect_type])
                        
                        if len(all_points_by_folder) < 2:
                            st.info(f"{defect_type}：数据不足，至少需要2个文件夹有此类型缺陷")
                            continue
                        
                        # 执行坐标匹配
                        from scipy.spatial import KDTree
                        
                        # 合并所有点并记录来源
                        all_points = []
                        point_sources = []
                        point_defect_ids = []  # 记录nDefectID
                        point_features = []    # 新增：记录特征数据
                        
                        # 统计信息
                        total_points_before_filter = 0
                        filtered_kla_points = 0
                        
                        for folder, points in all_points_by_folder.items():
                            for point_data in points:
                                total_points_before_filter += 1
                                x, y = point_data[0], point_data[1]
                                defect_id = point_data[2] if len(point_data) > 2 else None
                                features = point_data[3] if len(point_data) > 3 else {}
                                
                                # 过滤掉没有 nDefectID 的数据（来自KLA的漏检数据）
                                if defect_id is None:
                                    filtered_kla_points += 1
                                    continue
                                
                                all_points.append([x, y])
                                point_sources.append(folder)
                                point_defect_ids.append(defect_id)
                                point_features.append(features)
                        
                        # 显示过滤信息
                        if filtered_kla_points > 0:
                            st.info(f"已过滤 {filtered_kla_points} 个来自KLA的{defect_type}数据（无nDefectID），保留 {len(all_points)} 个CASI数据用于共有率分析")
                        
                        if len(all_points) == 0:
                            st.warning(f"{defect_type}：过滤后无有效数据，跳过分析")
                            continue
                        
                        all_points = np.array(all_points)
                        
                        # 使用KDTree查找邻近点
                        tree = KDTree(all_points)
                        matched_groups = []
                        processed = set()
                        
                        for i, point in enumerate(all_points):
                            if i in processed:
                                continue
                            
                            # 查找阈值范围内的所有点
                            indices = tree.query_ball_point(point, cohesion_threshold)
                            
                            # 记录来源文件夹、nDefectID和特征数据
                            group_folders = [point_sources[idx] for idx in indices]
                            unique_folders = list(set(group_folders))
                            
                            if len(unique_folders) >= min_occurrence:
                                # 计算组的中心点
                                group_points = all_points[indices]
                                center = np.mean(group_points, axis=0)
                                
                                # 收集每个文件夹的nDefectID和特征数据
                                folder_defect_ids = {}
                                folder_features = {}  # 新增：保存每个文件夹的特征数据
                                
                                for idx in indices:
                                    folder = point_sources[idx]
                                    defect_id = point_defect_ids[idx]
                                    features = point_features[idx]
                                    
                                    if folder not in folder_defect_ids:
                                        folder_defect_ids[folder] = []
                                        folder_features[folder] = []
                                    
                                    if defect_id is not None:
                                        folder_defect_ids[folder].append(defect_id)
                                        folder_features[folder].append(features)
                                
                                matched_groups.append({
                                    'center': center,
                                    'folders': unique_folders,
                                    'count': len(unique_folders),
                                    'points': group_points,
                                    'folder_defect_ids': folder_defect_ids,
                                    'folder_features': folder_features  # 新增：保存特征数据
                                })
                            
                            # 标记为已处理
                            processed.update(indices)
                        
                        # 显示统计结果
                        if matched_groups:
                            st.write(f"### 📈 {defect_type} 共有位置统计")
                            
                            # 统计每个文件夹的数据
                            folder_stats = []
                            for folder in sorted(all_points_by_folder.keys()):
                                total_count = len(all_points_by_folder[folder])
                                
                                # 计算该文件夹在共有位置中的点数
                                shared_count = sum(1 for group in matched_groups if folder in group['folders'])
                                shared_ratio = (shared_count / total_count * 100) if total_count > 0 else 0
                                
                                folder_stats.append({
                                    '文件夹': folder,
                                    f'{defect_type}总数': total_count,
                                    '共有位置数': shared_count,
                                    '共有率': f"{shared_ratio:.2f}%"
                                })
                            
                            stats_df = pd.DataFrame(folder_stats)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # 共有位置统计
                            st.write(f"**共有位置总数：** {len(matched_groups)}")
                            
                            # 按出现次数统计
                            occurrence_counts = {}
                            for group in matched_groups:
                                count = group['count']
                                occurrence_counts[count] = occurrence_counts.get(count, 0) + 1
                            
                            st.write("**按出现次数分布：**")
                            for count in sorted(occurrence_counts.keys(), reverse=True):
                                st.write(f"  - 出现在 {count} 个文件夹：{occurrence_counts[count]} 个位置")
                            
                            # 新增：显示nDefectID对应关系表格
                            st.write("---")
                            st.write(f"### 📋 {defect_type} nDefectID对应关系及特征数据表")
                            
                            # 准备对应关系数据
                            correspondence_data = []
                            sorted_folders = sorted(all_points_by_folder.keys())
                            
                            for i, group in enumerate(matched_groups, 1):
                                row_data = {
                                    '共有位置ID': i,
                                    'X坐标': f"{group['center'][0]:.2f}",
                                    'Y坐标': f"{group['center'][1]:.2f}",
                                    '出现次数': group['count']
                                }
                                
                                # 首先添加所有文件夹的nDefectID
                                for folder in sorted_folders:
                                    if folder in group['folder_defect_ids']:
                                        defect_ids = group['folder_defect_ids'][folder]
                                        unique_ids = sorted(list(set(defect_ids)))
                                        row_data[f'nDefectID_{folder}'] = ', '.join(map(str, unique_ids))
                                    else:
                                        row_data[f'nDefectID_{folder}'] = ''
                                
                                # 定义要显示的特征（按通道和特征组织）
                                feature_names = [
                                    'DW1O_MaxOrg', 'DW1O_BGMean', 'DW1O_BGDev', 'DW1O_Size', 'DW1O_TotalSNR', 'DW1O_MapSNR',
                                    'DW2O_MaxOrg', 'DW2O_BGMean', 'DW2O_BGDev', 'DW2O_Size', 'DW2O_TotalSNR', 'DW2O_MapSNR',
                                    'DN1O_MaxOrg', 'DN1O_BGMean', 'DN1O_BGDev', 'DN1O_Size', 'DN1O_TotalSNR', 'DN1O_MapSNR'
                                ]
                                
                                # 按特征名称循环，每个特征对应所有文件夹
                                for feat_name in feature_names:
                                    for folder in sorted_folders:
                                        # 特征数据列 - 如果有多个缺陷，取平均值
                                        if folder in group['folder_features'] and len(group['folder_features'][folder]) > 0:
                                            features_list = group['folder_features'][folder]
                                            
                                            values = []
                                            for feat_dict in features_list:
                                                if feat_name in feat_dict and feat_dict[feat_name] is not None:
                                                    try:
                                                        val = float(feat_dict[feat_name])
                                                        if not np.isnan(val):
                                                            values.append(val)
                                                    except (ValueError, TypeError):
                                                        pass
                                            
                                            if values:
                                                # 如果有多个值，显示平均值
                                                if len(values) > 1:
                                                    row_data[f'{feat_name}_{folder}'] = f"{np.mean(values):.2f} (avg)"
                                                else:
                                                    row_data[f'{feat_name}_{folder}'] = f"{values[0]:.2f}"
                                            else:
                                                row_data[f'{feat_name}_{folder}'] = ''
                                        else:
                                            # 该文件夹在此位置没有数据
                                            row_data[f'{feat_name}_{folder}'] = ''
                                
                                correspondence_data.append(row_data)
                            
                            if correspondence_data:
                                correspondence_df = pd.DataFrame(correspondence_data)
                                
                                # 显示表格（由于列数较多，使用可滚动视图）
                                st.write(f"**数据说明：** 表格包含 {len(correspondence_df)} 个共有位置")
                                st.write(f"**列结构：** 基础信息 → nDefectID(所有文件夹) → 特征(每个特征对应所有文件夹)")
                                st.write(f"**特征顺序：** DW1O(MaxOrg→BGMean→BGDev→Size→TotalSNR→MapSNR) → DW2O(...) → DN1O(...)")
                                st.dataframe(correspondence_df, use_container_width=True, height=400)
                                
                                # 导出nDefectID对应关系及特征数据表
                                csv_correspondence = correspondence_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label=f"📥 导出{defect_type} 共有缺陷完整特征数据表(CSV)",
                                    data=csv_correspondence,
                                    file_name=f"{defect_type}_共有缺陷_nDefectID及特征数据_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key=f"download_defectid_{defect_type}_csv",
                                    help=f"下载包含{len(correspondence_df)}行数据和{len(correspondence_df.columns)}列的完整特征数据表"
                                )
                            else:
                                st.info("没有可显示的nDefectID对应关系数据")
                            
                            # 新增：不共有缺陷分析
                            st.write("---")
                            st.write(f"### 📋 {defect_type} 不共有缺陷分析")
                            st.write("""
                            **说明：** 仅在部分文件夹中出现的缺陷（未达到最小出现次数要求），同时查找其他文件夹中相同位置的缺陷特征
                            
                            **匹配说明：**
                            - ✅ **找到匹配**：在其他文件夹的相同位置（匹配距离阈值内）找到缺陷，显示其nDefectID、缺陷类型、nDefectType和特征值
                            - ❌ **空白**：在该位置没有找到任何缺陷（包括已被nDefectType=1000/10001筛掉的）
                            - 🔍 **nDefectType标识**：
                              - 1000/10001：表示该缺陷被筛选规则过滤掉
                              - 其他值：正常缺陷
                            - 💡 **无nDefectID的情况**：可能是KLA漏检数据（KLA源数据没有nDefectID字段）
                            """)
                            
                            # 收集所有文件夹的所有点数据（不限于当前defect_type）
                            all_folders_all_points = {}
                            for dt in defect_types:
                                if dt in all_points_by_defect_type:
                                    for folder, points in all_points_by_defect_type[dt].items():
                                        if folder not in all_folders_all_points:
                                            all_folders_all_points[folder] = []
                                        all_folders_all_points[folder].extend(points)
                            
                            # 找出所有未被匹配成共有位置的点
                            non_shared_data = []
                            
                            for folder in sorted_folders:
                                if folder not in all_points_by_folder:
                                    continue
                                
                                folder_points = all_points_by_folder[folder]
                                
                                # 检查每个点是否在共有位置中
                                for point_data in folder_points:
                                    x, y = point_data[0], point_data[1]
                                    defect_id = point_data[2] if len(point_data) > 2 else None
                                    features = point_data[3] if len(point_data) > 3 else {}
                                    src_defect_type_value = point_data[4] if len(point_data) > 4 else None
                                    
                                    # 检查这个点是否在任何共有组中
                                    is_shared = False
                                    for group in matched_groups:
                                        if folder in group['folders']:
                                            # 计算到组中心的距离
                                            dist = np.sqrt((x - group['center'][0])**2 + (y - group['center'][1])**2)
                                            if dist <= cohesion_threshold:
                                                is_shared = True
                                                break
                                    
                                    # 如果不在共有组中，添加到不共有列表
                                    if not is_shared and defect_id is not None:
                                        row_data = {
                                            '源文件夹': folder,
                                            '源nDefectID': defect_id,
                                            '源nDefectType': src_defect_type_value if src_defect_type_value is not None else '',
                                            '源X坐标': f"{x:.2f}",
                                            '源Y坐标': f"{y:.2f}",
                                            '源缺陷类型': defect_type
                                        }
                                        
                                        # 添加源文件夹的特征数据
                                        feature_names = [
                                            'DW1O_MaxOrg', 'DW1O_BGMean', 'DW1O_BGDev', 'DW1O_Size', 'DW1O_TotalSNR', 'DW1O_MapSNR',
                                            'DW2O_MaxOrg', 'DW2O_BGMean', 'DW2O_BGDev', 'DW2O_Size', 'DW2O_TotalSNR', 'DW2O_MapSNR',
                                            'DN1O_MaxOrg', 'DN1O_BGMean', 'DN1O_BGDev', 'DN1O_Size', 'DN1O_TotalSNR', 'DN1O_MapSNR'
                                        ]
                                        
                                        for feat_name in feature_names:
                                            if feat_name in features and features[feat_name] is not None:
                                                try:
                                                    val = float(features[feat_name])
                                                    if not np.isnan(val):
                                                        row_data[f'源_{feat_name}'] = f"{val:.2f}"
                                                    else:
                                                        row_data[f'源_{feat_name}'] = ''
                                                except (ValueError, TypeError):
                                                    row_data[f'源_{feat_name}'] = ''
                                            else:
                                                row_data[f'源_{feat_name}'] = ''
                                        
                                        # 在其他文件夹中查找相同位置的缺陷
                                        for other_folder in sorted_folders:
                                            if other_folder == folder:
                                                continue
                                            
                                            if other_folder not in all_folders_all_points:
                                                # 如果该文件夹没有数据，设置为空
                                                row_data[f'{other_folder}_nDefectID'] = ''
                                                row_data[f'{other_folder}_nDefectType'] = ''
                                                row_data[f'{other_folder}_缺陷类型'] = ''
                                                row_data[f'{other_folder}_距离'] = ''
                                                for feat_name in feature_names:
                                                    row_data[f'{other_folder}_{feat_name}'] = ''
                                                continue
                                            
                                            # 查找最近的匹配点
                                            min_dist = float('inf')
                                            matched_point = None
                                            
                                            for other_point_data in all_folders_all_points[other_folder]:
                                                other_x, other_y = other_point_data[0], other_point_data[1]
                                                dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                                                
                                                if dist < min_dist and dist <= cohesion_threshold:
                                                    min_dist = dist
                                                    matched_point = other_point_data
                                            
                                            # 如果找到匹配点，提取其信息
                                            if matched_point is not None:
                                                other_defect_id = matched_point[2] if len(matched_point) > 2 else None
                                                other_features = matched_point[3] if len(matched_point) > 3 else {}
                                                other_defect_type_value = matched_point[4] if len(matched_point) > 4 else None
                                                
                                                # 确定该点在other_folder中的缺陷类型
                                                other_defect_type = '未知'
                                                for dt in defect_types:
                                                    if dt in all_points_by_defect_type:
                                                        if other_folder in all_points_by_defect_type[dt]:
                                                            for pt in all_points_by_defect_type[dt][other_folder]:
                                                                if pt[2] == other_defect_id:
                                                                    other_defect_type = dt
                                                                    break
                                                
                                                row_data[f'{other_folder}_nDefectID'] = other_defect_id if other_defect_id else ''
                                                row_data[f'{other_folder}_nDefectType'] = other_defect_type_value if other_defect_type_value is not None else ''
                                                row_data[f'{other_folder}_缺陷类型'] = other_defect_type
                                                row_data[f'{other_folder}_距离'] = f"{min_dist:.2f}"
                                                
                                                # 添加其他文件夹的特征数据
                                                for feat_name in feature_names:
                                                    if feat_name in other_features and other_features[feat_name] is not None:
                                                        try:
                                                            val = float(other_features[feat_name])
                                                            if not np.isnan(val):
                                                                row_data[f'{other_folder}_{feat_name}'] = f"{val:.2f}"
                                                            else:
                                                                row_data[f'{other_folder}_{feat_name}'] = ''
                                                        except (ValueError, TypeError):
                                                            row_data[f'{other_folder}_{feat_name}'] = ''
                                                    else:
                                                        row_data[f'{other_folder}_{feat_name}'] = ''
                                            else:
                                                # 未找到匹配点
                                                row_data[f'{other_folder}_nDefectID'] = ''
                                                row_data[f'{other_folder}_nDefectType'] = ''
                                                row_data[f'{other_folder}_缺陷类型'] = ''
                                                row_data[f'{other_folder}_距离'] = ''
                                                for feat_name in feature_names:
                                                    row_data[f'{other_folder}_{feat_name}'] = ''
                                        
                                        non_shared_data.append(row_data)
                            
                            if non_shared_data:
                                non_shared_df = pd.DataFrame(non_shared_data)
                                
                                # 按源文件夹分组统计
                                st.write(f"**不共有缺陷统计：**")
                                non_shared_counts = non_shared_df['源文件夹'].value_counts()
                                
                                cols_stat = st.columns(min(len(non_shared_counts), 4))
                                for idx, (folder, count) in enumerate(non_shared_counts.items()):
                                    with cols_stat[idx % 4]:
                                        st.metric(f"{folder}", count)
                                
                                st.write(f"**总计：** {len(non_shared_df)} 个不共有{defect_type}缺陷")
                                
                                # 重组列顺序：基础信息 → 源文件夹特征 → 其他文件夹信息和特征
                                base_cols = ['源文件夹', '源nDefectID', '源nDefectType', '源X坐标', '源Y坐标', '源缺陷类型']
                                
                                # 源文件夹的18个特征列
                                source_feature_cols = [f'源_{feat}' for feat in feature_names]
                                
                                # 其他文件夹的列（按文件夹组织）
                                other_folder_cols = []
                                for other_folder in sorted_folders:
                                    if f'{other_folder}_nDefectID' in non_shared_df.columns:
                                        # 每个文件夹的基本信息
                                        other_folder_cols.extend([
                                            f'{other_folder}_nDefectID',
                                            f'{other_folder}_nDefectType',
                                            f'{other_folder}_缺陷类型',
                                            f'{other_folder}_距离'
                                        ])
                                        # 每个文件夹的18个特征
                                        other_folder_cols.extend([f'{other_folder}_{feat}' for feat in feature_names])
                                
                                # 重新排序DataFrame列
                                ordered_cols = base_cols + source_feature_cols + other_folder_cols
                                ordered_cols = [col for col in ordered_cols if col in non_shared_df.columns]
                                non_shared_df = non_shared_df[ordered_cols]
                                
                                # 显示表格（带提示信息）
                                st.info("""
                                💡 **表格说明：** 
                                - **每行**：一个不共有缺陷及其在其他文件夹中对应位置的特征值
                                - **源nDefectType**：源缺陷的类型标识（1000/10001表示被筛选规则过滤）
                                - **其他文件夹列**：
                                  - ✅ **有数据**：在该位置找到缺陷（显示nDefectID、nDefectType、缺陷类型、距离和特征）
                                  - ❌ **空白**：在该位置没有找到任何缺陷
                                - **nDefectType解读**：
                                  - **1000/10001**：被nDefectType筛选规则过滤掉的缺陷
                                  - **其他数值**：正常检出的缺陷
                                  - **空白**：可能是KLA数据（无nDefectType字段）或该位置无缺陷
                                """)
                                st.dataframe(non_shared_df, use_container_width=True, height=400)
                                
                                # 导出不共有缺陷数据
                                csv_non_shared = non_shared_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label=f"📥 导出{defect_type} 不共有缺陷跨文件夹对比数据(CSV)",
                                    data=csv_non_shared,
                                    file_name=f"{defect_type}_不共有缺陷_跨文件夹对比_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key=f"download_non_shared_{defect_type}_csv",
                                    help=f"下载包含{len(non_shared_df)}个不共有缺陷及其在其他文件夹中对应位置的完整特征数据"
                                )
                                
                                # 统计信息：在其他文件夹中找到对应位置的比例
                                st.write("---")
                                st.write(f"### 📊 跨文件夹匹配统计")
                                
                                match_stats = []
                                for other_folder in sorted_folders:
                                    if f'{other_folder}_nDefectID' in non_shared_df.columns:
                                        # 排除源文件夹自身
                                        non_self_df = non_shared_df[non_shared_df['源文件夹'] != other_folder]
                                        if len(non_self_df) > 0:
                                            matched_count = non_self_df[f'{other_folder}_nDefectID'].notna().sum()
                                            match_ratio = (matched_count / len(non_self_df) * 100) if len(non_self_df) > 0 else 0
                                            match_stats.append({
                                                '目标文件夹': other_folder,
                                                '找到对应位置': matched_count,
                                                '总不共有数': len(non_self_df),
                                                '匹配率': f"{match_ratio:.2f}%"
                                            })
                                
                                if match_stats:
                                    match_stats_df = pd.DataFrame(match_stats)
                                    st.dataframe(match_stats_df, use_container_width=True)
                                
                                # 可视化不共有缺陷分布
                                st.write(f"### 🗺️ {defect_type} 不共有缺陷分布图")
                                
                                fig_non_shared = go.Figure()
                                
                                # 为每个文件夹的不共有缺陷使用不同颜色
                                folder_colors = {
                                    folder: colors[idx % len(colors)] 
                                    for idx, folder in enumerate(sorted_folders)
                                }
                                
                                for folder in sorted_folders:
                                    folder_data = non_shared_df[non_shared_df['源文件夹'] == folder]
                                    if len(folder_data) > 0:
                                        x_coords = [float(x) for x in folder_data['源X坐标']]
                                        y_coords = [float(y) for y in folder_data['源Y坐标']]
                                        
                                        hover_texts = [
                                            f"源文件夹: {row['源文件夹']}<br>源nDefectID: {row['源nDefectID']}<br>X: {row['源X坐标']}<br>Y: {row['源Y坐标']}"
                                            for _, row in folder_data.iterrows()
                                        ]
                                        
                                        fig_non_shared.add_trace(go.Scatter(
                                            x=x_coords,
                                            y=y_coords,
                                            mode='markers',
                                            name=f'{folder} ({len(folder_data)})',
                                            marker=dict(
                                                size=8,
                                                color=folder_colors[folder],
                                                line=dict(width=1, color='white')
                                            ),
                                            hovertext=hover_texts,
                                            hoverinfo='text'
                                        ))
                                
                                # 添加晶圆边界
                                theta = np.linspace(0, 2*np.pi, 100)
                                circle_x = 150000 + 150000 * np.cos(theta)
                                circle_y = 150000 + 150000 * np.sin(theta)
                                
                                fig_non_shared.add_trace(go.Scatter(
                                    x=circle_x,
                                    y=circle_y,
                                    mode='lines',
                                    name='晶圆边界',
                                    line=dict(color='gray', width=2, dash='dash'),
                                    showlegend=True,
                                    hoverinfo='skip'
                                ))
                                
                                fig_non_shared.update_layout(
                                    title=f'{defect_type} 不共有缺陷分布',
                                    xaxis=dict(
                                        title='X坐标',
                                        range=[0, 300000],
                                        scaleanchor="y",
                                        scaleratio=1
                                    ),
                                    yaxis=dict(
                                        title='Y坐标',
                                        range=[0, 300000]
                                    ),
                                    width=800,
                                    height=800,
                                    hovermode='closest'
                                )
                                
                                st.plotly_chart(fig_non_shared, use_container_width=True)
                            else:
                                st.info(f"{defect_type}：所有缺陷都是共有的，没有不共有缺陷")
                            
                            # 可视化共有位置
                            st.write(f"### 🗺️ {defect_type} 共有位置分布图")
                            
                            fig_cohesion = go.Figure()
                            
                            # 创建颜色映射（按出现次数）
                            max_count = max(group['count'] for group in matched_groups)
                            
                            # 按出现次数分组显示
                            for occurrence in sorted(set(group['count'] for group in matched_groups), reverse=True):
                                groups_with_occurrence = [g for g in matched_groups if g['count'] == occurrence]
                                
                                x_coords = [g['center'][0] for g in groups_with_occurrence]
                                y_coords = [g['center'][1] for g in groups_with_occurrence]
                                
                                hover_texts = [
                                    f"出现次数: {g['count']}<br>X: {g['center'][0]:.2f}<br>Y: {g['center'][1]:.2f}<br>文件夹: {', '.join(g['folders'])}"
                                    for g in groups_with_occurrence
                                ]
                                
                                # 颜色渐变：次数越多颜色越深
                                intensity = occurrence / max_count
                                color = f'rgba(255, {int(100 * (1-intensity))}, {int(100 * (1-intensity))}, 0.7)'
                                
                                fig_cohesion.add_trace(go.Scatter(
                                    x=x_coords,
                                    y=y_coords,
                                    mode='markers',
                                    name=f'出现{occurrence}次 ({len(groups_with_occurrence)})',
                                    marker=dict(
                                        size=8 + 4 * (occurrence / max_count),
                                        color=color,
                                        line=dict(width=1, color='white')
                                    ),
                                    hovertext=hover_texts,
                                    hoverinfo='text'
                                ))
                            
                            # 添加晶圆边界
                            theta = np.linspace(0, 2*np.pi, 100)
                            circle_x = 150000 + 150000 * np.cos(theta)
                            circle_y = 150000 + 150000 * np.sin(theta)
                            
                            fig_cohesion.add_trace(go.Scatter(
                                x=circle_x,
                                y=circle_y,
                                mode='lines',
                                name='晶圆边界',
                                line=dict(color='gray', width=2, dash='dash'),
                                showlegend=True,
                                hoverinfo='skip'
                            ))
                            
                            fig_cohesion.update_layout(
                                title=f'{defect_type} 共有位置分布',
                                xaxis=dict(
                                    title='X坐标',
                                    range=[0, 300000],
                                    scaleanchor="y",
                                    scaleratio=1
                                ),
                                yaxis=dict(
                                    title='Y坐标',
                                    range=[0, 300000]
                                ),
                                width=800,
                                height=800,
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig_cohesion, use_container_width=True)
                            
                            # 导出共有位置基础数据（坐标和出现次数）
                            export_data = []
                            for i, group in enumerate(matched_groups, 1):
                                export_data.append({
                                    '位置ID': i,
                                    'X坐标': group['center'][0],
                                    'Y坐标': group['center'][1],
                                    '出现次数': group['count'],
                                    '文件夹列表': ', '.join(group['folders'])
                                })
                            
                            export_df = pd.DataFrame(export_data)
                            csv_export = export_df.to_csv(index=False, encoding='utf-8-sig')
                            
                            st.download_button(
                                label=f"📥 导出{defect_type}共有位置基础数据(CSV)",
                                data=csv_export,
                                file_name=f"{defect_type}_共有位置基础数据_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=f"download_{defect_type}_csv",
                                help=f"下载包含位置坐标和出现次数的基础数据（不含详细特征）"
                            )
                        else:
                            st.info(f"{defect_type}：未找到满足条件的共有位置（至少出现{min_occurrence}次）")
            
            except Exception as e:
                st.error(f"共有率分析时出错: {str(e)}")
                st.exception(e)


# Tab3: 区域过滤
with tab3:
    st.markdown('<a name="区域过滤"></a>', unsafe_allow_html=True)
    st.header("✂️ 区域过滤 - 删除指定区域内的缺陷点")
    
    st.markdown("""
    ### 功能说明：
    1. 选择包含子文件夹的父文件夹
    2. 自动读取每个子文件夹内的 `BlobFeatures*.csv` 文件
    3. 支持**多边形框选**和矩形框选两种方式
    4. 生成去除区域内点的新CSV文件
    5. **可下载被删除的点**的CSV文件（新增）
    
    ### 操作步骤：
    **多边形框选（推荐）：**
    1. 输入文件夹路径，选择子文件夹
    2. 依次输入多个顶点的X、Y坐标，点击"➕ 添加顶点"
    3. 至少添加3个顶点形成多边形（绿色显示预览）
    4. 点击"🗑️ 删除多边形内"应用过滤
    5. 可以继续添加新的多边形区域
    
    **矩形框选（快捷方式）：**
    1. 直接输入X/Y的最小值和最大值
    2. 点击"🗑️ 删除矩形区域内"
    
    **导出功能：**
    - 💾 下载保留的点（过滤后的数据）
    - 💾 下载被删除的点（方便检查删除的数据）
    
    **图例说明：**
    - 🔵 蓝色点：当前保留的缺陷点
    - ❌ 红色×：已删除的缺陷点
    - 🟢 绿色：正在绘制的多边形（虚线为预览）
    - 🔴 红色虚线：已应用的删除区域
    """)
    
    # 文件夹选择
    filter_folder = st.text_input("📁 输入父文件夹路径", 
                                  value=r"D:\waferdata",
                                  key="filter_folder_input")
    
    if filter_folder and os.path.exists(filter_folder):
        # 导入配置文件功能
        st.write("---")
        with st.expander("📥 导入已保存的删除区域配置", expanded=False):
            st.info("选择之前导出的JSON配置文件，快速恢复删除区域设置")
            
            uploaded_config = st.file_uploader("上传配置文件 (JSON)", type=['json'], key="config_uploader")
            
            if uploaded_config is not None:
                try:
                    import json
                    config_data = json.load(uploaded_config)
                    
                    st.write(f"📄 配置文件包含 {len(config_data)} 个子文件夹的区域设置")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ 仅导入配置", key="apply_config"):
                            # 初始化必要的session state
                            if 'selection_boxes' not in st.session_state:
                                st.session_state.selection_boxes = {}
                            
                            applied_count = 0
                            for subfolder_name, regions in config_data.items():
                                region_list = []
                                for region in regions:
                                    if region['type'] == 'polygon':
                                        region_list.append({
                                            'type': 'polygon',
                                            'points': [tuple(p) for p in region['vertices']],
                                            'removed': 0  # 重置为0，表示未应用
                                        })
                                    else:
                                        bounds = region.get('bounds', region)  # 兼容两种格式
                                        region_list.append({
                                            'type': 'rectangle',
                                            'x_min': bounds.get('x_min'),
                                            'x_max': bounds.get('x_max'),
                                            'y_min': bounds.get('y_min'),
                                            'y_max': bounds.get('y_max'),
                                            'removed': 0  # 重置为0，表示未应用
                                        })
                                
                                st.session_state.selection_boxes[subfolder_name] = region_list
                                applied_count += 1
                            
                            st.success(f"✅ 已导入 {applied_count} 个子文件夹的删除区域配置")
                            st.info("⚠️ 注意：配置已导入，但未应用。请使用右侧按钮批量应用，或切换到各文件夹手动应用")
                            st.rerun()
                    
                    with col2:
                        if st.button("🚀 导入并批量应用到所有文件夹", key="apply_config_batch"):
                            # 初始化必要的session state
                            if 'selection_boxes' not in st.session_state:
                                st.session_state.selection_boxes = {}
                            if 'filtered_data' not in st.session_state:
                                st.session_state.filtered_data = {}
                            
                            from matplotlib.path import Path as MplPath
                            
                            # 获取所有子文件夹
                            subfolders = [f for f in os.listdir(filter_folder) 
                                         if os.path.isdir(os.path.join(filter_folder, f))]
                            
                            total_folders_processed = 0
                            total_points_removed = 0
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for folder_idx, subfolder_name in enumerate(subfolders):
                                if subfolder_name not in config_data:
                                    continue
                                
                                status_text.text(f"处理中: {subfolder_name} ({folder_idx + 1}/{len(subfolders)})")
                                
                                # 查找该文件夹的BlobFeatures文件
                                subfolder_path = os.path.join(filter_folder, subfolder_name)
                                blob_files = glob.glob(os.path.join(subfolder_path, "BlobFeatures*.csv"))
                                
                                if not blob_files:
                                    continue
                                
                                # 读取CSV文件
                                df_blob = pd.read_csv(blob_files[0])
                                
                                # 查找坐标列
                                x_col = None
                                y_col = None
                                for col in df_blob.columns:
                                    if 'dCenterXCartisian' in col and 'Move' not in col:
                                        x_col = col
                                    elif 'dCenterYCartisian' in col and 'Move' not in col:
                                        y_col = col
                                
                                if not (x_col and y_col):
                                    continue
                                
                                # 导入区域配置
                                region_list = []
                                for region in config_data[subfolder_name]:
                                    if region['type'] == 'polygon':
                                        region_list.append({
                                            'type': 'polygon',
                                            'points': [tuple(p) for p in region['vertices']],
                                            'removed': 0
                                        })
                                    else:
                                        bounds = region.get('bounds', region)
                                        region_list.append({
                                            'type': 'rectangle',
                                            'x_min': bounds.get('x_min'),
                                            'x_max': bounds.get('x_max'),
                                            'y_min': bounds.get('y_min'),
                                            'y_max': bounds.get('y_max'),
                                            'removed': 0
                                        })
                                
                                st.session_state.selection_boxes[subfolder_name] = region_list
                                
                                # 应用删除区域
                                df_working = df_blob.copy()
                                folder_removed = 0
                                
                                for idx, region in enumerate(st.session_state.selection_boxes[subfolder_name]):
                                    if region.get('type') == 'polygon':
                                        # 应用多边形过滤
                                        polygon_path = MplPath(region['points'])
                                        points = np.column_stack([df_working[x_col], df_working[y_col]])
                                        mask = ~polygon_path.contains_points(points)
                                        removed_count = (~mask).sum()
                                        df_working = df_working[mask].reset_index(drop=True)
                                        
                                        st.session_state.selection_boxes[subfolder_name][idx]['removed'] = removed_count
                                        folder_removed += removed_count
                                    else:
                                        # 应用矩形过滤
                                        mask = ~((df_working[x_col] >= region['x_min']) & 
                                               (df_working[x_col] <= region['x_max']) & 
                                               (df_working[y_col] >= region['y_min']) & 
                                               (df_working[y_col] <= region['y_max']))
                                        removed_count = (~mask).sum()
                                        df_working = df_working[mask].reset_index(drop=True)
                                        
                                        st.session_state.selection_boxes[subfolder_name][idx]['removed'] = removed_count
                                        folder_removed += removed_count
                                
                                # 保存过滤后的数据
                                st.session_state.filtered_data[subfolder_name] = df_working
                                
                                total_folders_processed += 1
                                total_points_removed += folder_removed
                                
                                # 更新进度
                                progress_bar.progress((folder_idx + 1) / len(subfolders))
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ 批量处理完成！")
                            st.info(f"📊 处理了 {total_folders_processed} 个文件夹，总共删除 {total_points_removed} 个点")
                            st.warning("💡 提示：切换到各文件夹查看效果，记得保存结果到CSV文件")
                            st.rerun()
                    
                    # 预览配置内容
                    with st.expander("👁️ 预览配置内容", expanded=False):
                        for subfolder_name, regions in config_data.items():
                            st.write(f"**{subfolder_name}** - {len(regions)} 个区域")
                            for region in regions:
                                if region['type'] == 'polygon':
                                    st.write(f"  • 多边形 {region['region_id']}: {len(region['vertices'])} 个顶点")
                                else:
                                    st.write(f"  • 矩形 {region['region_id']}")
                
                except Exception as e:
                    st.error(f"❌ 读取配置文件失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.write("---")
        
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(filter_folder) 
                     if os.path.isdir(os.path.join(filter_folder, f))]
        
        if subfolders:
            st.success(f"找到 {len(subfolders)} 个子文件夹")
            
            # 初始化session state
            if 'current_subfolder_idx' not in st.session_state:
                st.session_state.current_subfolder_idx = 0
            if 'filtered_data' not in st.session_state:
                st.session_state.filtered_data = {}
            if 'selection_boxes' not in st.session_state:
                st.session_state.selection_boxes = {}
            
            # 选择子文件夹
            current_idx = st.session_state.current_subfolder_idx
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("⬅️ 上一个", disabled=(current_idx == 0)):
                    st.session_state.current_subfolder_idx = max(0, current_idx - 1)
                    st.rerun()
            with col2:
                selected_subfolder = st.selectbox(
                    "选择子文件夹",
                    subfolders,
                    index=current_idx,
                    key="subfolder_select"
                )
                if selected_subfolder != subfolders[current_idx]:
                    st.session_state.current_subfolder_idx = subfolders.index(selected_subfolder)
                    st.rerun()
            with col3:
                if st.button("➡️ 下一个", disabled=(current_idx == len(subfolders) - 1)):
                    st.session_state.current_subfolder_idx = min(len(subfolders) - 1, current_idx + 1)
                    st.rerun()
            
            st.write(f"**当前子文件夹:** {selected_subfolder} ({current_idx + 1}/{len(subfolders)})")
            
            # 查找BlobFeatures文件
            subfolder_path = os.path.join(filter_folder, selected_subfolder)
            blob_files = glob.glob(os.path.join(subfolder_path, "BlobFeatures*.csv"))
            
            if blob_files:
                blob_file = blob_files[0]  # 使用第一个匹配的文件
                st.info(f"📄 找到文件: {os.path.basename(blob_file)}")
                
                try:
                    # 读取CSV文件
                    df_blob = pd.read_csv(blob_file)
                    
                    # 查找坐标列
                    x_col = None
                    y_col = None
                    for col in df_blob.columns:
                        if 'dCenterXCartisian' in col and 'Move' not in col:
                            x_col = col
                        elif 'dCenterYCartisian' in col and 'Move' not in col:
                            y_col = col
                    
                    if x_col and y_col and x_col in df_blob.columns and y_col in df_blob.columns:
                        st.success(f"✅ 找到坐标列: X={x_col}, Y={y_col}")
                        st.write(f"📊 原始数据点数: {len(df_blob)}")
                        
                        # 获取或初始化当前子文件夹的过滤数据
                        if selected_subfolder not in st.session_state.filtered_data:
                            st.session_state.filtered_data[selected_subfolder] = df_blob.copy()
                            # 只在selection_boxes中没有该文件夹时才初始化为空列表
                            # 这样可以保留之前复制过来的区域
                            if selected_subfolder not in st.session_state.selection_boxes:
                                st.session_state.selection_boxes[selected_subfolder] = []
                        
                        # 确保selection_boxes中有该文件夹的键（处理边缘情况）
                        if selected_subfolder not in st.session_state.selection_boxes:
                            st.session_state.selection_boxes[selected_subfolder] = []
                        
                        df_current = st.session_state.filtered_data[selected_subfolder]
                        
                        # 创建交互式散点图
                        st.subheader("📍 缺陷点分布图")
                        
                        col_plot, col_control = st.columns([3, 1])
                        
                        with col_control:
                            st.write("### 过滤控制")
                            st.write(f"**当前点数:** {len(df_current)}")
                            st.write(f"**已删除:** {len(df_blob) - len(df_current)}")
                            
                            # 多边形框选功能
                            st.write("---")
                            st.write("**多边形框选删除**")
                            st.info("💡 在下方输入多个顶点坐标，依次点击形成多边形区域\n\n"
                                   "**获取坐标方法：**\n"
                                   "1. 将鼠标悬停在图表上的点附近\n"
                                   "2. 查看弹出的坐标信息\n"
                                   "3. 输入到下方表单中")
                            
                            # 初始化多边形顶点
                            if f'polygon_points_{selected_subfolder}' not in st.session_state:
                                st.session_state[f'polygon_points_{selected_subfolder}'] = []
                            
                            # 添加顶点
                            with st.form(key=f"add_vertex_{selected_subfolder}"):
                                col_x, col_y = st.columns(2)
                                with col_x:
                                    vertex_x = st.number_input("X 坐标", value=0.0, step=1000.0, key=f"vx_{selected_subfolder}")
                                with col_y:
                                    vertex_y = st.number_input("Y 坐标", value=0.0, step=1000.0, key=f"vy_{selected_subfolder}")
                                
                                submit_vertex = st.form_submit_button("➕ 添加顶点")
                                
                                if submit_vertex:
                                    st.session_state[f'polygon_points_{selected_subfolder}'].append((vertex_x, vertex_y))
                                    st.success(f"✅ 已添加顶点 ({vertex_x:.0f}, {vertex_y:.0f})")
                                    st.rerun()
                            
                            # 显示当前多边形顶点
                            polygon_points = st.session_state[f'polygon_points_{selected_subfolder}']
                            if polygon_points:
                                st.write(f"**当前顶点数:** {len(polygon_points)}")
                                for idx, (px, py) in enumerate(polygon_points):
                                    col_idx, col_coords, col_del = st.columns([1, 3, 1])
                                    with col_idx:
                                        st.write(f"{idx + 1}.")
                                    with col_coords:
                                        st.write(f"({px:.0f}, {py:.0f})")
                                    with col_del:
                                        if st.button("❌", key=f"del_vertex_{selected_subfolder}_{idx}"):
                                            st.session_state[f'polygon_points_{selected_subfolder}'].pop(idx)
                                            st.rerun()
                                
                                # 应用多边形过滤
                                col_apply, col_clear = st.columns(2)
                                with col_apply:
                                    if st.button("🗑️ 删除多边形内", key=f"apply_poly_{selected_subfolder}", 
                                               disabled=(len(polygon_points) < 3)):
                                        # 使用多边形判断点是否在内部
                                        from matplotlib.path import Path
                                        
                                        polygon_path = Path(polygon_points)
                                        points = np.column_stack([df_current[x_col], df_current[y_col]])
                                        mask = ~polygon_path.contains_points(points)
                                        
                                        removed_count = (~mask).sum()
                                        df_current = df_current[mask].reset_index(drop=True)
                                        st.session_state.filtered_data[selected_subfolder] = df_current
                                        
                                        # 记录删除区域
                                        st.session_state.selection_boxes[selected_subfolder].append({
                                            'type': 'polygon',
                                            'points': polygon_points.copy(),
                                            'removed': removed_count
                                        })
                                        
                                        # 清空当前多边形
                                        st.session_state[f'polygon_points_{selected_subfolder}'] = []
                                        
                                        st.success(f"✅ 已删除 {removed_count} 个点")
                                        st.rerun()
                                
                                with col_clear:
                                    if st.button("🔄 清空顶点", key=f"clear_poly_{selected_subfolder}"):
                                        st.session_state[f'polygon_points_{selected_subfolder}'] = []
                                        st.rerun()
                                
                                if len(polygon_points) < 3:
                                    st.warning("⚠️ 至少需要3个顶点才能形成多边形")
                            
                            # 矩形框选功能（保留作为备选）
                            st.write("---")
                            st.write("**矩形框选删除（快捷方式）**")
                            
                            with st.form(key=f"manual_filter_{selected_subfolder}"):
                                x_min = st.number_input("X 最小值", value=0.0, step=1000.0)
                                x_max = st.number_input("X 最大值", value=300000.0, step=1000.0)
                                y_min = st.number_input("Y 最小值", value=0.0, step=1000.0)
                                y_max = st.number_input("Y 最大值", value=300000.0, step=1000.0)
                                
                                submit_filter = st.form_submit_button("🗑️ 删除矩形区域内")
                                
                                if submit_filter:
                                    # 过滤数据
                                    mask = ~((df_current[x_col] >= x_min) & 
                                           (df_current[x_col] <= x_max) & 
                                           (df_current[y_col] >= y_min) & 
                                           (df_current[y_col] <= y_max))
                                    
                                    removed_count = (~mask).sum()
                                    df_current = df_current[mask].reset_index(drop=True)
                                    st.session_state.filtered_data[selected_subfolder] = df_current
                                    
                                    # 记录删除区域
                                    st.session_state.selection_boxes[selected_subfolder].append({
                                        'type': 'rectangle',
                                        'x_min': x_min, 'x_max': x_max,
                                        'y_min': y_min, 'y_max': y_max,
                                        'removed': removed_count
                                    })
                                    
                                    st.success(f"✅ 已删除 {removed_count} 个点")
                                    st.rerun()
                            
                            # 显示已删除的区域
                            if st.session_state.selection_boxes[selected_subfolder]:
                                st.write("---")
                                st.write("**已删除区域:**")
                                
                                # 检查是否有未应用的区域（removed为0的区域）
                                unapplied_regions = [box for box in st.session_state.selection_boxes[selected_subfolder] 
                                                    if box.get('removed', 0) == 0]
                                
                                if unapplied_regions:
                                    st.warning(f"⚠️ 有 {len(unapplied_regions)} 个区域未应用（可能是从其他文件夹复制的）")
                                    
                                    if st.button("🔄 应用所有未应用的删除区域", key=f"apply_unapplied_{selected_subfolder}"):
                                        from matplotlib.path import Path
                                        
                                        total_removed = 0
                                        df_working = df_current.copy()
                                        
                                        for idx, region in enumerate(st.session_state.selection_boxes[selected_subfolder]):
                                            if region.get('removed', 0) == 0:  # 只处理未应用的
                                                if region.get('type') == 'polygon':
                                                    # 应用多边形过滤
                                                    polygon_path = Path(region['points'])
                                                    points = np.column_stack([df_working[x_col], df_working[y_col]])
                                                    mask = ~polygon_path.contains_points(points)
                                                    removed_count = (~mask).sum()
                                                    df_working = df_working[mask].reset_index(drop=True)
                                                    
                                                    # 更新删除数量
                                                    st.session_state.selection_boxes[selected_subfolder][idx]['removed'] = removed_count
                                                    total_removed += removed_count
                                                else:
                                                    # 应用矩形过滤
                                                    mask = ~((df_working[x_col] >= region['x_min']) & 
                                                           (df_working[x_col] <= region['x_max']) & 
                                                           (df_working[y_col] >= region['y_min']) & 
                                                           (df_working[y_col] <= region['y_max']))
                                                    removed_count = (~mask).sum()
                                                    df_working = df_working[mask].reset_index(drop=True)
                                                    
                                                    # 更新删除数量
                                                    st.session_state.selection_boxes[selected_subfolder][idx]['removed'] = removed_count
                                                    total_removed += removed_count
                                        
                                        # 更新数据
                                        st.session_state.filtered_data[selected_subfolder] = df_working
                                        
                                        st.success(f"✅ 已应用 {len(unapplied_regions)} 个删除区域，总共删除 {total_removed} 个点")
                                        st.rerun()
                                
                                # 显示所有区域详情
                                for idx, box in enumerate(st.session_state.selection_boxes[selected_subfolder]):
                                    status_text = f"(删除 {box['removed']} 点)" if box.get('removed', 0) > 0 else "(未应用)"
                                    status_color = "🟢" if box.get('removed', 0) > 0 else "🔴"
                                    
                                    if box.get('type') == 'polygon':
                                        with st.expander(f"{status_color} 多边形区域 {idx + 1} {status_text}"):
                                            st.write(f"顶点数: {len(box['points'])}")
                                            for i, (px, py) in enumerate(box['points']):
                                                st.write(f"  {i+1}. ({px:.0f}, {py:.0f})")
                                            
                                            # 单独应用此区域的按钮
                                            if box.get('removed', 0) == 0:
                                                if st.button(f"应用此区域", key=f"apply_region_{selected_subfolder}_{idx}"):
                                                    from matplotlib.path import Path
                                                    polygon_path = Path(box['points'])
                                                    points = np.column_stack([df_current[x_col], df_current[y_col]])
                                                    mask = ~polygon_path.contains_points(points)
                                                    removed_count = (~mask).sum()
                                                    df_current = df_current[mask].reset_index(drop=True)
                                                    
                                                    st.session_state.filtered_data[selected_subfolder] = df_current
                                                    st.session_state.selection_boxes[selected_subfolder][idx]['removed'] = removed_count
                                                    
                                                    st.success(f"✅ 已删除 {removed_count} 个点")
                                                    st.rerun()
                                    else:
                                        with st.expander(f"{status_color} 矩形区域 {idx + 1} {status_text}"):
                                            st.write(f"X: [{box['x_min']:.0f}, {box['x_max']:.0f}]")
                                            st.write(f"Y: [{box['y_min']:.0f}, {box['y_max']:.0f}]")
                                            
                                            # 单独应用此区域的按钮
                                            if box.get('removed', 0) == 0:
                                                if st.button(f"应用此区域", key=f"apply_region_{selected_subfolder}_{idx}"):
                                                    mask = ~((df_current[x_col] >= box['x_min']) & 
                                                           (df_current[x_col] <= box['x_max']) & 
                                                           (df_current[y_col] >= box['y_min']) & 
                                                           (df_current[y_col] <= box['y_max']))
                                                    removed_count = (~mask).sum()
                                                    df_current = df_current[mask].reset_index(drop=True)
                                                    
                                                    st.session_state.filtered_data[selected_subfolder] = df_current
                                                    st.session_state.selection_boxes[selected_subfolder][idx]['removed'] = removed_count
                                                    
                                                    st.success(f"✅ 已删除 {removed_count} 个点")
                                                    st.rerun()
                            
                            # 重置按钮
                            st.write("---")
                            st.write("**🔄 重置选项**")
                            col_reset1, col_reset2 = st.columns(2)
                            
                            with col_reset1:
                                if st.button("重置数据", key=f"reset_data_{selected_subfolder}", 
                                           help="恢复原始数据，保留删除区域定义"):
                                    st.session_state.filtered_data[selected_subfolder] = df_blob.copy()
                                    # 将所有区域标记为未应用
                                    for region in st.session_state.selection_boxes[selected_subfolder]:
                                        region['removed'] = 0
                                    st.success("✅ 数据已重置，区域定义已保留")
                                    st.rerun()
                            
                            with col_reset2:
                                if st.button("完全重置", key=f"reset_all_{selected_subfolder}",
                                           help="恢复原始数据并清除所有删除区域"):
                                    st.session_state.filtered_data[selected_subfolder] = df_blob.copy()
                                    st.session_state.selection_boxes[selected_subfolder] = []
                                    if f'polygon_points_{selected_subfolder}' in st.session_state:
                                        st.session_state[f'polygon_points_{selected_subfolder}'] = []
                                    st.success("✅ 已完全重置")
                                    st.rerun()
                            
                            # 导出按钮
                            st.write("---")
                            if len(df_current) < len(df_blob):
                                st.write("**💾 导出数据**")
                                col_export1, col_export2 = st.columns(2)
                                
                                with col_export1:
                                    # 下载过滤后的CSV（保留的点）
                                    output_filename = f"{selected_subfolder}_filtered.csv"
                                    csv_data = df_current.to_csv(index=False, encoding='utf-8-sig')
                                    
                                    st.download_button(
                                        label="💾 下载过滤后的CSV（保留的点）",
                                        data=csv_data,
                                        file_name=output_filename,
                                        mime="text/csv",
                                        key=f"download_{selected_subfolder}"
                                    )
                                
                                with col_export2:
                                    # 下载被删除的点的CSV（新增功能）
                                    # 找出被删除的点：原始数据中不在当前数据中的点
                                    if len(df_current) > 0:
                                        # 使用索引来找出被删除的行
                                        # 假设有唯一标识列，或者通过坐标来匹配
                                        # 这里我们通过比较索引（如果可用）或者创建一个标记
                                        df_blob_with_index = df_blob.copy()
                                        df_current_with_index = df_current.copy()
                                        
                                        # 添加临时索引用于比较
                                        df_blob_with_index['_temp_index'] = range(len(df_blob))
                                        
                                        # 通过坐标匹配找出保留的点
                                        # 创建一个合并键来标识每个点
                                        df_blob_with_index['_merge_key'] = (
                                            df_blob_with_index[x_col].round(2).astype(str) + '_' + 
                                            df_blob_with_index[y_col].round(2).astype(str)
                                        )
                                        df_current_with_index['_merge_key'] = (
                                            df_current_with_index[x_col].round(2).astype(str) + '_' + 
                                            df_current_with_index[y_col].round(2).astype(str)
                                        )
                                        
                                        # 找出被删除的点
                                        deleted_keys = set(df_blob_with_index['_merge_key']) - set(df_current_with_index['_merge_key'])
                                        df_deleted = df_blob_with_index[df_blob_with_index['_merge_key'].isin(deleted_keys)].copy()
                                        
                                        # 删除临时列
                                        df_deleted = df_deleted.drop(['_temp_index', '_merge_key'], axis=1)
                                    else:
                                        # 如果所有点都被删除，返回原始数据
                                        df_deleted = df_blob.copy()
                                    
                                    deleted_filename = f"{selected_subfolder}_deleted.csv"
                                    csv_deleted = df_deleted.to_csv(index=False, encoding='utf-8-sig')
                                    
                                    st.download_button(
                                        label=f"💾 下载被删除的点（{len(df_deleted)}个）",
                                        data=csv_deleted,
                                        file_name=deleted_filename,
                                        mime="text/csv",
                                        key=f"download_deleted_{selected_subfolder}"
                                    )
                                
                                # 保存到原文件夹选项
                                st.write("")
                                col_save1, col_save2 = st.columns(2)
                                with col_save1:
                                    if st.button("💾 保存过滤后到原文件夹", key=f"save_{selected_subfolder}"):
                                        output_path = os.path.join(subfolder_path, output_filename)
                                        df_current.to_csv(output_path, index=False, encoding='utf-8-sig')
                                        st.success(f"✅ 已保存到: {output_path}")
                                
                                with col_save2:
                                    if st.button("💾 保存被删除点到原文件夹", key=f"save_deleted_{selected_subfolder}"):
                                        deleted_path = os.path.join(subfolder_path, deleted_filename)
                                        df_deleted.to_csv(deleted_path, index=False, encoding='utf-8-sig')
                                        st.success(f"✅ 已保存到: {deleted_path}")
                            else:
                                st.info("未进行任何过滤")
                            
                            # 复制删除区域到其他子文件夹
                            if st.session_state.selection_boxes.get(selected_subfolder):
                                st.write("---")
                                st.write("**📋 复制删除区域**")
                                st.info("将当前子文件夹的删除区域应用到其他子文件夹")
                                
                                # 选择目标子文件夹
                                other_subfolders = [f for f in subfolders if f != selected_subfolder]
                                if other_subfolders:
                                    target_folders = st.multiselect(
                                        "选择目标子文件夹",
                                        other_subfolders,
                                        key=f"copy_target_{selected_subfolder}"
                                    )
                                    
                                    if target_folders:
                                        if st.button("📋 复制区域到选中的子文件夹", key=f"copy_regions_{selected_subfolder}"):
                                            copied_regions = st.session_state.selection_boxes[selected_subfolder]
                                            
                                            # 复制区域到每个目标文件夹
                                            for target in target_folders:
                                                # 确保目标文件夹有selection_boxes键
                                                if target not in st.session_state.selection_boxes:
                                                    st.session_state.selection_boxes[target] = []
                                                
                                                # 深拷贝区域信息（追加而非覆盖）
                                                for region in copied_regions:
                                                    if region.get('type') == 'polygon':
                                                        # 深拷贝多边形顶点
                                                        new_region = {
                                                            'type': 'polygon',
                                                            'points': copy.deepcopy(region['points']),
                                                            'removed': 0  # 重置删除数量，因为还没应用
                                                        }
                                                        st.session_state.selection_boxes[target].append(new_region)
                                                    else:
                                                        # 复制矩形区域
                                                        new_region = {
                                                            'type': 'rectangle',
                                                            'x_min': region['x_min'],
                                                            'x_max': region['x_max'],
                                                            'y_min': region['y_min'],
                                                            'y_max': region['y_max'],
                                                            'removed': 0
                                                        }
                                                        st.session_state.selection_boxes[target].append(new_region)
                                            
                                            st.success(f"✅ 已复制 {len(copied_regions)} 个删除区域到 {len(target_folders)} 个子文件夹")
                                            st.info("💡 切换到目标子文件夹查看区域，需要重新处理数据才会实际删除点")
                                            
                                            # 触发页面重新加载以更新状态
                                            st.rerun()
                                else:
                                    st.info("没有其他子文件夹可以复制")
                        
                        with col_plot:
                            # 创建Plotly散点图
                            fig = go.Figure()
                            
                            # 添加当前点
                            fig.add_trace(go.Scatter(
                                x=df_current[x_col],
                                y=df_current[y_col],
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color='blue',
                                    opacity=0.6
                                ),
                                name='当前点',
                                text=[f"Index: {i}<br>X: {x:.2f}<br>Y: {y:.2f}" 
                                      for i, (x, y) in enumerate(zip(df_current[x_col], df_current[y_col]))],
                                hovertemplate='%{text}<extra></extra>'
                            ))
                            
                            # 如果有删除的点，显示为红色
                            if len(df_current) < len(df_blob):
                                # 找出被删除的点
                                df_removed = df_blob[~df_blob.index.isin(df_current.index)]
                                if len(df_removed) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=df_removed[x_col],
                                        y=df_removed[y_col],
                                        mode='markers',
                                        marker=dict(
                                            size=5,
                                            color='red',
                                            opacity=0.3,
                                            symbol='x'
                                        ),
                                        name='已删除点',
                                        text=[f"X: {x:.2f}<br>Y: {y:.2f}" 
                                              for x, y in zip(df_removed[x_col], df_removed[y_col])],
                                        hovertemplate='%{text}<extra></extra>'
                                    ))
                            
                            # 添加已保存的删除区域
                            for idx, box in enumerate(st.session_state.selection_boxes[selected_subfolder]):
                                if box.get('type') == 'polygon':
                                    # 绘制多边形
                                    points = box['points']
                                    # 闭合多边形
                                    x_coords = [p[0] for p in points] + [points[0][0]]
                                    y_coords = [p[1] for p in points] + [points[0][1]]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_coords,
                                        y=y_coords,
                                        mode='lines',
                                        line=dict(color='red', width=2, dash='dash'),
                                        fill='toself',
                                        fillcolor='rgba(255, 0, 0, 0.1)',
                                        name=f'已删除多边形 {idx + 1}',
                                        showlegend=True,
                                        hoverinfo='skip'
                                    ))
                                else:
                                    # 绘制矩形
                                    fig.add_shape(
                                        type="rect",
                                        x0=box['x_min'], x1=box['x_max'],
                                        y0=box['y_min'], y1=box['y_max'],
                                        line=dict(color="red", width=2, dash="dash"),
                                        fillcolor="red",
                                        opacity=0.1,
                                        name=f"已删除矩形 {idx + 1}"
                                    )
                            
                            # 显示当前正在绘制的多边形
                            polygon_points = st.session_state[f'polygon_points_{selected_subfolder}']
                            if polygon_points:
                                if len(polygon_points) >= 2:
                                    # 绘制已连接的边
                                    x_coords = [p[0] for p in polygon_points]
                                    y_coords = [p[1] for p in polygon_points]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_coords,
                                        y=y_coords,
                                        mode='lines+markers',
                                        line=dict(color='green', width=2),
                                        marker=dict(size=10, color='green'),
                                        name='当前多边形（绘制中）',
                                        showlegend=True,
                                        hoverinfo='skip'
                                    ))
                                    
                                    # 如果有3个或以上顶点，显示闭合预览
                                    if len(polygon_points) >= 3:
                                        x_coords_closed = x_coords + [x_coords[0]]
                                        y_coords_closed = y_coords + [y_coords[0]]
                                        
                                        fig.add_trace(go.Scatter(
                                            x=x_coords_closed,
                                            y=y_coords_closed,
                                            mode='lines',
                                            line=dict(color='lightgreen', width=1, dash='dot'),
                                            fill='toself',
                                            fillcolor='rgba(0, 255, 0, 0.1)',
                                            name='多边形预览',
                                            showlegend=True,
                                            hoverinfo='skip'
                                        ))
                                else:
                                    # 只有一个顶点，显示为点
                                    fig.add_trace(go.Scatter(
                                        x=[polygon_points[0][0]],
                                        y=[polygon_points[0][1]],
                                        mode='markers',
                                        marker=dict(size=10, color='green'),
                                        name='当前顶点',
                                        showlegend=True,
                                        hoverinfo='skip'
                                    ))
                            
                            # 设置坐标轴
                            fig.update_xaxes(
                                range=[0, 300000],
                                title='X坐标',
                                scaleanchor="y",
                                scaleratio=1
                            )
                            fig.update_yaxes(
                                range=[0, 300000],
                                title='Y坐标'
                            )
                            
                            fig.update_layout(
                                title=f'{selected_subfolder} - 缺陷点分布',
                                width=800,
                                height=800,
                                hovermode='closest',
                                showlegend=True,
                                dragmode='pan'  # 默认为拖动模式
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("💡 提示：\n"
                                   "- 在左侧输入坐标添加多边形顶点（绿色显示）\n"
                                   "- 至少添加3个顶点可以形成多边形\n"
                                   "- 点击'删除多边形内'应用过滤\n"
                                   "- 已删除区域显示为红色虚线\n"
                                   "- 可以使用图表工具栏缩放和平移查看细节")
                        
                        # 批量导出所有过滤后的文件
                        st.write("---")
                        st.subheader("📦 批量导出")
                        
                        col_export1, col_export2 = st.columns(2)
                        
                        with col_export1:
                            if st.button("💾 导出所有已过滤的子文件夹"):
                                export_count = 0
                                for subfolder_name, filtered_df in st.session_state.filtered_data.items():
                                    # 获取原始数据
                                    subfolder_path_exp = os.path.join(filter_folder, subfolder_name)
                                    blob_files_exp = glob.glob(os.path.join(subfolder_path_exp, "BlobFeatures*.csv"))
                                    
                                    if blob_files_exp:
                                        original_df = pd.read_csv(blob_files_exp[0])
                                        
                                        # 只导出有变化的文件
                                        if len(filtered_df) < len(original_df):
                                            output_path = os.path.join(subfolder_path_exp, f"{subfolder_name}_filtered.csv")
                                            filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                                            export_count += 1
                                
                                if export_count > 0:
                                    st.success(f"✅ 已导出 {export_count} 个过滤后的文件")
                                else:
                                    st.info("没有需要导出的文件（所有文件都未进行过滤）")
                        
                        with col_export2:
                            if st.button("📄 导出删除区域配置"):
                                # 辅助函数：转换numpy类型为Python原生类型
                                def convert_to_native_types(obj):
                                    """递归转换numpy类型为Python原生类型"""
                                    if isinstance(obj, dict):
                                        return {k: convert_to_native_types(v) for k, v in obj.items()}
                                    elif isinstance(obj, list):
                                        return [convert_to_native_types(item) for item in obj]
                                    elif isinstance(obj, tuple):
                                        return tuple(convert_to_native_types(item) for item in obj)
                                    elif isinstance(obj, (np.integer, np.int64, np.int32)):
                                        return int(obj)
                                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                                        return float(obj)
                                    elif isinstance(obj, np.ndarray):
                                        return obj.tolist()
                                    else:
                                        return obj
                                
                                # 收集所有子文件夹的删除区域信息
                                all_regions_config = {}
                                
                                for subfolder_name, regions in st.session_state.selection_boxes.items():
                                    if regions:  # 只导出有删除区域的子文件夹
                                        all_regions_config[subfolder_name] = []
                                        for idx, region in enumerate(regions):
                                            region_info = {
                                                'region_id': int(idx + 1),
                                                'type': str(region.get('type', 'rectangle')),
                                                'removed_count': int(region.get('removed', 0))
                                            }
                                            
                                            if region.get('type') == 'polygon':
                                                # 转换顶点坐标
                                                vertices = region.get('points', [])
                                                region_info['vertices'] = [[float(x), float(y)] for x, y in vertices]
                                                region_info['vertex_count'] = int(len(vertices))
                                            else:
                                                region_info['x_min'] = float(region.get('x_min', 0))
                                                region_info['x_max'] = float(region.get('x_max', 0))
                                                region_info['y_min'] = float(region.get('y_min', 0))
                                                region_info['y_max'] = float(region.get('y_max', 0))
                                            
                                            all_regions_config[subfolder_name].append(region_info)
                                
                                if all_regions_config:
                                    import json
                                    # 确保所有数据都是原生Python类型
                                    all_regions_config = convert_to_native_types(all_regions_config)
                                    json_data = json.dumps(all_regions_config, indent=2, ensure_ascii=False)
                                    
                                    st.download_button(
                                        label="💾 下载JSON配置",
                                        data=json_data,
                                        file_name=f"filter_regions_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        key="download_regions_json"
                                    )
                                    
                                    # 同时生成CSV格式（展开的多边形顶点）
                                    regions_list = []
                                    for subfolder_name, regions in all_regions_config.items():
                                        for region in regions:
                                            if region['type'] == 'polygon':
                                                for vertex_idx, (x, y) in enumerate(region['vertices']):
                                                    regions_list.append({
                                                        '子文件夹': subfolder_name,
                                                        '区域ID': region['region_id'],
                                                        '类型': '多边形',
                                                        '顶点序号': vertex_idx + 1,
                                                        'X坐标': x,
                                                        'Y坐标': y,
                                                        '删除点数': region['removed_count'] if vertex_idx == 0 else ''
                                                    })
                                            else:
                                                regions_list.append({
                                                    '子文件夹': subfolder_name,
                                                    '区域ID': region['region_id'],
                                                    '类型': '矩形',
                                                    '顶点序号': '',
                                                    'X坐标': f"{region['x_min']} ~ {region['x_max']}",
                                                    'Y坐标': f"{region['y_min']} ~ {region['y_max']}",
                                                    '删除点数': region['removed_count']
                                                })
                                    
                                    if regions_list:
                                        regions_df = pd.DataFrame(regions_list)
                                        csv_data = regions_df.to_csv(index=False, encoding='utf-8-sig')
                                        
                                        st.download_button(
                                            label="📊 下载CSV配置",
                                            data=csv_data,
                                            file_name=f"filter_regions_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv",
                                            key="download_regions_csv"
                                        )
                                    
                                    st.success("✅ 配置文件已准备好下载")
                                else:
                                    st.info("没有删除区域需要导出")
                        
                        # 删除区域配置预览和管理
                        st.write("---")
                        st.subheader("📋 删除区域配置管理")
                        
                        # 显示所有子文件夹的删除区域汇总
                        if st.session_state.selection_boxes:
                            total_regions = sum(len(regions) for regions in st.session_state.selection_boxes.values())
                            total_polygon = sum(
                                sum(1 for r in regions if r.get('type') == 'polygon')
                                for regions in st.session_state.selection_boxes.values()
                            )
                            total_rectangle = total_regions - total_polygon
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("总删除区域数", total_regions)
                            with col_stat2:
                                st.metric("多边形区域", total_polygon)
                            with col_stat3:
                                st.metric("矩形区域", total_rectangle)
                            
                            # 详细列表
                            with st.expander("📝 查看所有删除区域详情", expanded=False):
                                for subfolder_name, regions in st.session_state.selection_boxes.items():
                                    if regions:
                                        st.write(f"### {subfolder_name}")
                                        for idx, region in enumerate(regions):
                                            if region.get('type') == 'polygon':
                                                st.write(f"**区域 {idx + 1} - 多边形** (删除 {region['removed']} 点)")
                                                vertices_text = "\n".join([
                                                    f"  顶点{i+1}: ({x:.2f}, {y:.2f})" 
                                                    for i, (x, y) in enumerate(region['points'])
                                                ])
                                                st.text(vertices_text)
                                            else:
                                                st.write(f"**区域 {idx + 1} - 矩形** (删除 {region['removed']} 点)")
                                                st.text(f"  X: {region['x_min']:.2f} ~ {region['x_max']:.2f}\n"
                                                       f"  Y: {region['y_min']:.2f} ~ {region['y_max']:.2f}")
                                        st.write("---")
                            
                            # 保存配置到本地文件
                            if st.button("💾 保存配置到父文件夹", key="save_config_local"):
                                import json
                                
                                # 辅助函数：转换numpy类型为Python原生类型
                                def convert_to_native_types(obj):
                                    """递归转换numpy类型为Python原生类型"""
                                    if isinstance(obj, dict):
                                        return {k: convert_to_native_types(v) for k, v in obj.items()}
                                    elif isinstance(obj, list):
                                        return [convert_to_native_types(item) for item in obj]
                                    elif isinstance(obj, tuple):
                                        return tuple(convert_to_native_types(item) for item in obj)
                                    elif isinstance(obj, (np.integer, np.int64, np.int32)):
                                        return int(obj)
                                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                                        return float(obj)
                                    elif isinstance(obj, np.ndarray):
                                        return obj.tolist()
                                    else:
                                        return obj
                                
                                config_data = {}
                                for subfolder_name, regions in st.session_state.selection_boxes.items():
                                    if regions:
                                        config_data[subfolder_name] = []
                                        for idx, region in enumerate(regions):
                                            region_info = {
                                                'region_id': int(idx + 1),
                                                'type': str(region.get('type', 'rectangle')),
                                                'removed_count': int(region.get('removed', 0))
                                            }
                                            
                                            if region.get('type') == 'polygon':
                                                # 转换顶点坐标
                                                vertices = region.get('points', [])
                                                region_info['vertices'] = [[float(x), float(y)] for x, y in vertices]
                                            else:
                                                region_info['bounds'] = {
                                                    'x_min': float(region.get('x_min', 0)),
                                                    'x_max': float(region.get('x_max', 0)),
                                                    'y_min': float(region.get('y_min', 0)),
                                                    'y_max': float(region.get('y_max', 0))
                                                }
                                            
                                            config_data[subfolder_name].append(region_info)
                                
                                if config_data:
                                    # 确保所有数据都是原生Python类型
                                    config_data = convert_to_native_types(config_data)
                                    config_path = os.path.join(filter_folder, f"filter_regions_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
                                    with open(config_path, 'w', encoding='utf-8') as f:
                                        json.dump(config_data, f, indent=2, ensure_ascii=False)
                                    st.success(f"✅ 配置已保存到: {config_path}")
                                else:
                                    st.info("没有删除区域需要保存")
                        else:
                            st.info("当前没有任何删除区域")
                        
                    else:
                        st.error(f"❌ 未找到坐标列 dCenterXCartisian 和 dCenterYCartisian")
                        st.write("可用的列：", df_blob.columns.tolist())
                        
                except Exception as e:
                    st.error(f"读取文件时出错: {str(e)}")
                    st.exception(e)
            else:
                st.warning(f"⚠️ 在子文件夹 {selected_subfolder} 中未找到 BlobFeatures*.csv 文件")
        else:
            st.warning("⚠️ 未找到子文件夹")
    elif filter_folder:
        st.error("❌ 文件夹路径不存在，请检查路径")


    st.markdown('<a name="饱和像素分析"></a>', unsafe_allow_html=True)
    st.header("🔬 饱和像素分析")
    st.write("分析mask图像中缺陷区域的饱和像素分布")
    
    # 导入必要的库
    import cv2
    from pathlib import Path
    from scipy.signal import find_peaks
    from scipy.interpolate import interp1d
    import io
    
    # 辅助函数
    @st.cache_data
    def find_peaks_in_row(row_values, height_threshold=None):
        """找到一行像素值中的波峰个数"""
        if len(row_values) < 3:
            return 0
        
        if height_threshold is None:
            std_val = np.std(row_values)
            prominence = std_val * 0.5
        else:
            prominence = height_threshold
        
        peaks, _ = find_peaks(row_values, prominence=prominence)
        return len(peaks)
    
    @st.cache_data
    def count_saturated_pixels(row_values, saturated_value=65532):
        """计算一行中等于饱和值的像素个数"""
        return np.sum(row_values == saturated_value)
    
    def create_row_plot(row_values, row_index):
        """为一行像素值创建拟合曲线图"""
        fig, ax = plt.subplots(figsize=(6, 3))
        x = np.arange(len(row_values))
        
        # 如果数据点太多，进行采样
        if len(row_values) > 500:
            step = len(row_values) // 300
            x_sampled = x[::step]
            y_sampled = row_values[::step]
        else:
            x_sampled = x
            y_sampled = row_values
        
        # 绘制原始数据
        ax.plot(x_sampled, y_sampled, 'b-', alpha=0.6, linewidth=0.8, label='原始')
        
        # 拟合曲线
        if len(row_values) > 3:
            try:
                f = interp1d(x, row_values, kind='cubic', fill_value='extrapolate')
                x_smooth = np.linspace(0, len(row_values)-1, min(len(row_values)*3, 500))
                y_smooth = f(x_smooth)
                ax.plot(x_smooth, y_smooth, 'r-', linewidth=1.2, label='拟合', alpha=0.8)
            except:
                pass
        
        ax.set_xlabel('像素位置', fontsize=9)
        ax.set_ylabel('像素值', fontsize=9)
        ax.set_title(f'第{row_index+1}行像素值分布', fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        
        return fig
    
    @st.cache_data
    def analyze_defect_from_mask(mask_img, original_img):
        """根据mask图分析原图中的缺陷"""
        # 确保mask是二值图
        if len(mask_img.shape) == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        
        # 二值化mask
        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        
        # 找到mask中的连通区域
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # 获取最大的连通区域
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # 提取缺陷区域
        defect_region = original_img[y:y+h, x:x+w]
        defect_mask = binary_mask[y:y+h, x:x+w]
        
        # 分析每一行
        row_results = []
        for row_idx in range(h):
            row_mask = defect_mask[row_idx, :]
            row_pixels = defect_region[row_idx, :]
            
            # 只分析mask中标记为缺陷的像素
            valid_pixels = row_pixels[row_mask > 0]
            
            if len(valid_pixels) > 0:
                peak_count = find_peaks_in_row(valid_pixels)
                saturated_count = count_saturated_pixels(valid_pixels)
                total_pixels = len(valid_pixels)
                saturated_ratio = (saturated_count / total_pixels * 100) if total_pixels > 0 else 0
                row_results.append({
                    'row_index': row_idx,
                    'pixels': valid_pixels,
                    'peak_count': peak_count,
                    'saturated_count': saturated_count,
                    'total_pixels': total_pixels,
                    'saturated_ratio': saturated_ratio
                })
        
        # 找到像素个数最多的行作为主行
        if len(row_results) > 0:
            main_row_idx = max(range(len(row_results)), key=lambda i: row_results[i]['total_pixels'])
            
            # 标记主行、次行1和次行2
            for i, row in enumerate(row_results):
                if i == main_row_idx:
                    row['row_type'] = '主行'
                elif i == main_row_idx - 1:
                    row['row_type'] = '次行1'
                elif i == main_row_idx + 1:
                    row['row_type'] = '次行2'
                else:
                    row['row_type'] = '其他'
        
        return row_results
    
    @st.cache_data
    def process_folder_tab7(folder_path):
        """处理单个文件夹中的所有缺陷图像"""
        folder_path = Path(folder_path)
        folder_name = folder_path.name
        
        results = []
        
        # 查找所有mask文件
        mask_files = list(folder_path.glob('*-defect-*.bmp'))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, mask_file in enumerate(mask_files):
            # 解析文件名
            filename = mask_file.stem
            parts = filename.split('-')
            
            if len(parts) >= 3:
                defect_id = parts[0]
                channel = parts[2]
                
                # 找到对应的原图
                original_file = folder_path / f"{defect_id}-{channel}.tiff"
                
                if not original_file.exists():
                    continue
                
                # 读取图像
                mask_img = cv2.imread(str(mask_file))
                original_img = cv2.imread(str(original_file), cv2.IMREAD_UNCHANGED)
                
                if mask_img is None or original_img is None:
                    continue
                
                # 分析缺陷
                row_analysis = analyze_defect_from_mask(mask_img, original_img)
                
                if row_analysis and len(row_analysis) > 0:
                    # 提取主行、次行1、次行2的数据
                    main_row = next((r for r in row_analysis if r.get('row_type') == '主行'), None)
                    sub_row1 = next((r for r in row_analysis if r.get('row_type') == '次行1'), None)
                    sub_row2 = next((r for r in row_analysis if r.get('row_type') == '次行2'), None)
                    
                    result = {
                        '缺陷ID': defect_id,
                        '子文件夹': folder_name,
                        '通道': channel,
                        '缺陷行数': len(row_analysis),
                        '主行像素数': main_row['total_pixels'] if main_row else 0,
                        '主行饱和像素数': main_row['saturated_count'] if main_row else 0,
                        '主行饱和占比(%)': round(main_row['saturated_ratio'], 2) if main_row else 0,
                        '次行1像素数': sub_row1['total_pixels'] if sub_row1 else 0,
                        '次行1饱和像素数': sub_row1['saturated_count'] if sub_row1 else 0,
                        '次行1饱和占比(%)': round(sub_row1['saturated_ratio'], 2) if sub_row1 else 0,
                        '次行2像素数': sub_row2['total_pixels'] if sub_row2 else 0,
                        '次行2饱和像素数': sub_row2['saturated_count'] if sub_row2 else 0,
                        '次行2饱和占比(%)': round(sub_row2['saturated_ratio'], 2) if sub_row2 else 0,
                        'row_data': row_analysis
                    }
                    
                    results.append(result)
            
            # 更新进度
            progress = (idx + 1) / len(mask_files)
            progress_bar.progress(progress)
            status_text.text(f"处理中: {idx + 1}/{len(mask_files)}")
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    # UI界面
    st.subheader("📁 选择数据文件夹")
    
    root_folder = st.text_input("根文件夹路径", value=r"D:\1023kehupian\slot7\slot7-P1\crop")
    
    if st.button("🔍 开始分析", key="start_analysis_tab7"):
        root_path = Path(root_folder)
        
        if not root_path.exists():
            st.error(f"❌ 文件夹不存在: {root_folder}")
        else:
            # 获取所有子文件夹
            subfolders = [f for f in root_path.iterdir() if f.is_dir()]
            
            if not subfolders:
                st.warning("⚠️ 未找到子文件夹")
            else:
                st.success(f"✅ 找到 {len(subfolders)} 个子文件夹")
                
                all_results = []
                
                # 处理每个子文件夹
                for subfolder in subfolders:
                    with st.expander(f"📂 处理文件夹: {subfolder.name}", expanded=False):
                        results = process_folder_tab7(subfolder)
                        all_results.extend(results)
                        st.info(f"✅ 处理完成，找到 {len(results)} 个缺陷")
                
                if all_results:
                    st.success(f"🎉 总共处理了 {len(all_results)} 个缺陷")
                    
                    # 保存结果到session state
                    st.session_state['tab7_results'] = all_results
                else:
                    st.warning("⚠️ 未找到任何缺陷数据")
    
    # 显示结果
    if 'tab7_results' in st.session_state and st.session_state['tab7_results']:
        all_results = st.session_state['tab7_results']
        
        st.divider()
        st.subheader("📊 分析结果")
        
        # 创建数据表（不包含row_data）
        df_results = []
        for result in all_results:
            df_result = {k: v for k, v in result.items() if k != 'row_data'}
            df_results.append(df_result)
        
        df = pd.DataFrame(df_results)
        
        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总缺陷数", len(all_results))
        with col2:
            st.metric("平均主行饱和占比", f"{df['主行饱和占比(%)'].mean():.2f}%")
        with col3:
            st.metric("最大主行饱和占比", f"{df['主行饱和占比(%)'].max():.2f}%")
        with col4:
            st.metric("文件夹数", df['子文件夹'].nunique())
        
        # 显示数据表
        st.dataframe(df, use_container_width=True, height=400)
        
        # 下载按钮
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载CSV",
            data=csv,
            file_name="饱和像素分析结果.csv",
            mime="text/csv",
        )
        
        st.divider()
        st.subheader("📈 数据可视化")
        
        # 1. 箱线图对比
        st.write("#### 1️⃣ 饱和像素占比箱线图对比")
        
        # 按子文件夹分组
        folder_data = {}
        for result in all_results:
            folder = result['子文件夹']
            if folder not in folder_data:
                folder_data[folder] = {
                    '主行': [],
                    '次行1': [],
                    '次行2': []
                }
            
            folder_data[folder]['主行'].append(result['主行饱和占比(%)'])
            if result['次行1饱和占比(%)'] > 0 or result['次行1像素数'] > 0:
                folder_data[folder]['次行1'].append(result['次行1饱和占比(%)'])
            if result['次行2饱和占比(%)'] > 0 or result['次行2像素数'] > 0:
                folder_data[folder]['次行2'].append(result['次行2饱和占比(%)'])
        
        # 创建箱线图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        row_types = ['主行', '次行1', '次行2']
        
        for idx, row_type in enumerate(row_types):
            ax = axes[idx]
            data_to_plot = []
            labels = []
            
            for folder in sorted(folder_data.keys()):
                if folder_data[folder][row_type]:
                    data_to_plot.append(folder_data[folder][row_type])
                    labels.append(folder)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax.set_xlabel('文件夹', fontsize=10)
                ax.set_ylabel('饱和像素占比 (%)', fontsize=10)
                ax.set_title(f'{row_type}饱和像素占比分布', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # 2. 分文件夹的三行对比柱状图
        st.write("#### 2️⃣ 各文件夹三行对比")
        
        selected_folder = st.selectbox("选择文件夹", sorted(folder_data.keys()))
        
        if selected_folder:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_dict = {
                '主行': folder_data[selected_folder]['主行'],
                '次行1': folder_data[selected_folder]['次行1'],
                '次行2': folder_data[selected_folder]['次行2']
            }
            
            # 计算统计量
            stats_data = []
            for row_type in ['主行', '次行1', '次行2']:
                data = data_dict[row_type]
                if data:
                    stats_data.append({
                        'type': row_type,
                        'mean': np.mean(data),
                        'median': np.median(data),
                        'std': np.std(data),
                        'min': np.min(data),
                        'max': np.max(data)
                    })
            
            if stats_data:
                x = np.arange(len(stats_data))
                width = 0.35
                
                means = [s['mean'] for s in stats_data]
                medians = [s['median'] for s in stats_data]
                stds = [s['std'] for s in stats_data]
                labels_plot = [s['type'] for s in stats_data]
                
                ax.bar(x - width/2, means, width, label='平均值', alpha=0.8, yerr=stds, capsize=5)
                ax.bar(x + width/2, medians, width, label='中位数', alpha=0.8)
                
                ax.set_xlabel('行类型', fontsize=11)
                ax.set_ylabel('饱和像素占比 (%)', fontsize=11)
                ax.set_title(f'{selected_folder} - 饱和像素占比统计', fontsize=13, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(labels_plot)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # 3. 热图
        st.write("#### 3️⃣ 饱和占比热图")
        
        folders = sorted(folder_data.keys())
        row_types = ['主行', '次行1', '次行2']
        
        heatmap_data = np.zeros((len(row_types), len(folders)))
        
        for i, row_type in enumerate(row_types):
            for j, folder in enumerate(folders):
                data = folder_data[folder][row_type]
                if data:
                    heatmap_data[i, j] = np.mean(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(folders)))
        ax.set_yticks(np.arange(len(row_types)))
        ax.set_xticklabels(folders)
        ax.set_yticklabels(row_types)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(row_types)):
            for j in range(len(folders)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('不同文件夹各行类型平均饱和像素占比热图', fontsize=13, fontweight='bold')
        fig.colorbar(im, ax=ax, label='平均饱和占比 (%)')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # 4. 按通道分组的折线图
        st.write("#### 4️⃣ 按通道分组的文件夹对比")
        
        # 按通道和文件夹分组
        channel_data = {}
        for result in all_results:
            channel = result['通道']
            folder = result['子文件夹']
            
            if channel not in channel_data:
                channel_data[channel] = {}
            
            if folder not in channel_data[channel]:
                channel_data[channel][folder] = {
                    '主行': [],
                    '次行1': [],
                    '次行2': []
                }
            
            channel_data[channel][folder]['主行'].append(result['主行饱和占比(%)'])
            if result['次行1饱和占比(%)'] > 0 or result['次行1像素数'] > 0:
                channel_data[channel][folder]['次行1'].append(result['次行1饱和占比(%)'])
            if result['次行2饱和占比(%)'] > 0 or result['次行2像素数'] > 0:
                channel_data[channel][folder]['次行2'].append(result['次行2饱和占比(%)'])
        
        # 选择通道
        selected_channel = st.selectbox("选择通道", sorted(channel_data.keys()))
        
        if selected_channel:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            row_types = ['主行', '次行1', '次行2']
            
            folders = sorted(channel_data[selected_channel].keys())
            
            for idx, row_type in enumerate(row_types):
                ax = axes[idx]
                
                # 收集每个文件夹的统计数据
                means = []
                medians = []
                stds = []
                mins = []
                maxs = []
                
                for folder in folders:
                    data = channel_data[selected_channel][folder][row_type]
                    if data:
                        means.append(np.mean(data))
                        medians.append(np.median(data))
                        stds.append(np.std(data))
                        mins.append(np.min(data))
                        maxs.append(np.max(data))
                    else:
                        means.append(0)
                        medians.append(0)
                        stds.append(0)
                        mins.append(0)
                        maxs.append(0)
                
                x = np.arange(len(folders))
                
                # 绘制折线图
                ax.plot(x, means, 'o-', label='平均值', linewidth=2, markersize=8, color='#2E86AB')
                ax.plot(x, medians, 's-', label='中位数', linewidth=2, markersize=7, color='#A23B72')
                ax.fill_between(x, np.array(means) - np.array(stds), 
                               np.array(means) + np.array(stds), 
                               alpha=0.2, color='#2E86AB', label='±1标准差')
                
                # 添加最大最小值范围
                ax.fill_between(x, mins, maxs, alpha=0.1, color='gray', label='最大-最小范围')
                
                ax.set_xlabel('文件夹', fontsize=11)
                ax.set_ylabel('饱和像素占比 (%)', fontsize=11)
                ax.set_title(f'{selected_channel} 通道 - {row_type}饱和占比分布', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(folders, rotation=45, ha='right')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # 在点上标注数值
                for i, (mean_val, median_val) in enumerate(zip(means, medians)):
                    if mean_val > 0:
                        ax.text(i, mean_val, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f'{selected_channel} 通道 - 不同文件夹三行饱和占比对比', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 5. 像素值分布拟合图
        st.write("#### 5️⃣ 像素值分布拟合图")
        
        # 选择要查看的缺陷
        defect_options = [f"{r['缺陷ID']} - {r['子文件夹']} - {r['通道']}" for r in all_results]
        selected_defect_idx = st.selectbox("选择缺陷", range(len(defect_options)), 
                                          format_func=lambda x: defect_options[x])
        
        if selected_defect_idx is not None:
            selected_result = all_results[selected_defect_idx]
            row_analysis = selected_result['row_data']
            
            st.write(f"**缺陷信息:** {selected_result['缺陷ID']} - {selected_result['子文件夹']} - {selected_result['通道']}")
            
            # 找到主行、次行1、次行2
            main_row = next((r for r in row_analysis if r.get('row_type') == '主行'), None)
            sub_row1 = next((r for r in row_analysis if r.get('row_type') == '次行1'), None)
            sub_row2 = next((r for r in row_analysis if r.get('row_type') == '次行2'), None)
            
            rows_to_plot = []
            if main_row:
                rows_to_plot.append(('主行', main_row))
            if sub_row1:
                rows_to_plot.append(('次行1', sub_row1))
            if sub_row2:
                rows_to_plot.append(('次行2', sub_row2))
            
            # 创建多列布局
            cols = st.columns(len(rows_to_plot))
            
            for idx, (row_name, row_info) in enumerate(rows_to_plot):
                with cols[idx]:
                    st.write(f"**{row_name}**")
                    fig = create_row_plot(row_info['pixels'], row_info['row_index'])
                    st.pyplot(fig)
                    plt.close()
                    
                    st.write(f"像素数: {row_info['total_pixels']}")
                    st.write(f"饱和像素数: {row_info['saturated_count']}")
                    st.write(f"饱和占比: {row_info['saturated_ratio']:.2f}%")
        
        # 6. 饱和占比0-100%分布直方图
        st.write("#### 6️⃣ 饱和占比分布直方图 (0-100%)")
        
        # 提供不同的分组选项
        distribution_type = st.radio(
            "选择分布类型",
            ["总体分布", "按通道+行类型分组", "按文件夹分组", "按通道分组", "按行类型分组"],
            horizontal=True
        )
        
        if distribution_type == "总体分布":
            # 总体分布 - 主行、次行1、次行2分别显示
            st.write("**所有缺陷的饱和占比分布**")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            row_types = ['主行', '次行1', '次行2']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for idx, (row_type, color) in enumerate(zip(row_types, colors)):
                ax = axes[idx]
                
                # 收集该行类型的所有饱和占比数据
                if row_type == '主行':
                    data = [r['主行饱和占比(%)'] for r in all_results]
                elif row_type == '次行1':
                    data = [r['次行1饱和占比(%)'] for r in all_results if r['次行1像素数'] > 0]
                else:  # 次行2
                    data = [r['次行2饱和占比(%)'] for r in all_results if r['次行2像素数'] > 0]
                
                if data:
                    # 绘制直方图
                    n, bins, patches = ax.hist(data, bins=20, range=(0, 100), 
                                              alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                    
                    # 添加统计信息
                    mean_val = np.mean(data)
                    median_val = np.median(data)
                    std_val = np.std(data)
                    
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_val:.1f}%')
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'中位数: {median_val:.1f}%')
                    
                    ax.set_xlabel('饱和像素占比 (%)', fontsize=11)
                    ax.set_ylabel('缺陷数量', fontsize=11)
                    ax.set_title(f'{row_type} - 饱和占比分布 (n={len(data)})', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 添加统计文本框
                    textstr = f'平均: {mean_val:.1f}%\n中位数: {median_val:.1f}%\n标准差: {std_val:.1f}%'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.65, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        elif distribution_type == "按通道+行类型分组":
            # 按通道+行类型组合，显示在不同文件夹的分布
            st.write("**特定通道+行类型在不同文件夹的饱和占比分布**")
            
            # 选择通道和行类型
            col1, col2 = st.columns(2)
            with col1:
                channels = sorted(channel_data.keys())
                selected_channel_dist = st.selectbox("选择通道", channels, key='dist_channel_select')
            with col2:
                selected_row_type_dist = st.selectbox("选择行类型", ['主行', '次行1', '次行2'], key='dist_row_select')
            
            st.write(f"**{selected_channel_dist} 通道 - {selected_row_type_dist} 在各文件夹的分布**")
            
            # 收集该通道+行类型在各文件夹的数据
            folders = sorted(channel_data[selected_channel_dist].keys())
            n_folders = len(folders)
            
            # 动态计算子图布局
            n_cols = min(3, n_folders)
            n_rows = (n_folders + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_folders == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_folders > 1 else [axes]
            
            # 用于存储所有文件夹的统计信息
            all_folder_stats = []
            
            for idx, folder in enumerate(folders):
                ax = axes[idx]
                
                # 获取该文件夹该通道该行类型的数据
                data = channel_data[selected_channel_dist][folder][selected_row_type_dist]
                
                if data:
                    # 绘制直方图
                    n, bins, patches = ax.hist(data, bins=15, range=(0, 100), 
                                              alpha=0.75, color='steelblue', edgecolor='black', linewidth=0.5)
                    
                    # 统计信息
                    mean_val = np.mean(data)
                    median_val = np.median(data)
                    std_val = np.std(data)
                    
                    # 保存统计信息
                    all_folder_stats.append({
                        'folder': folder,
                        'mean': mean_val,
                        'median': median_val,
                        'std': std_val,
                        'count': len(data)
                    })
                    
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                              label=f'均值: {mean_val:.1f}%')
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, 
                              label=f'中位: {median_val:.1f}%')
                    
                    ax.set_xlabel('饱和占比 (%)', fontsize=10)
                    ax.set_ylabel('缺陷数量', fontsize=10)
                    ax.set_title(f'{folder}\n(n={len(data)})', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8, loc='upper right')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 添加统计文本框
                    textstr = f'μ={mean_val:.1f}%\nσ={std_val:.1f}%'
                    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', bbox=props)
                else:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{folder}', fontsize=11)
            
            # 隐藏多余的子图
            for idx in range(n_folders, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'{selected_channel_dist} 通道 - {selected_row_type_dist} - 不同文件夹分布对比', 
                        fontsize=14, fontweight='bold', y=1.0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 显示统计汇总表
            if all_folder_stats:
                st.write("**📊 统计汇总**")
                stats_df = pd.DataFrame(all_folder_stats)
                stats_df.columns = ['文件夹', '平均值(%)', '中位数(%)', '标准差(%)', '样本数']
                stats_df['平均值(%)'] = stats_df['平均值(%)'].round(2)
                stats_df['中位数(%)'] = stats_df['中位数(%)'].round(2)
                stats_df['标准差(%)'] = stats_df['标准差(%)'].round(2)
                st.dataframe(stats_df, use_container_width=True)
            
            # 添加对比柱状图
            if all_folder_stats and len(all_folder_stats) > 1:
                st.write("**📊 文件夹间对比柱状图**")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                folders_list = [s['folder'] for s in all_folder_stats]
                means = [s['mean'] for s in all_folder_stats]
                medians = [s['median'] for s in all_folder_stats]
                stds = [s['std'] for s in all_folder_stats]
                counts = [s['count'] for s in all_folder_stats]
                
                x = np.arange(len(folders_list))
                width = 0.35
                
                # 左图：平均值和中位数对比
                bars1 = ax1.bar(x - width/2, means, width, label='平均值', 
                               alpha=0.8, color='steelblue', yerr=stds, capsize=5)
                bars2 = ax1.bar(x + width/2, medians, width, label='中位数', 
                               alpha=0.8, color='coral')
                
                # 在柱子上添加数值
                for i, (m, med) in enumerate(zip(means, medians)):
                    ax1.text(i - width/2, m + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=9)
                    ax1.text(i + width/2, med + 1, f'{med:.1f}', ha='center', va='bottom', fontsize=9)
                
                ax1.set_xlabel('文件夹', fontsize=11)
                ax1.set_ylabel('饱和占比 (%)', fontsize=11)
                ax1.set_title(f'{selected_channel_dist}-{selected_row_type_dist} 平均值与中位数对比', 
                            fontsize=12, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(folders_list, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
                
                # 右图：样本数量
                bars3 = ax2.bar(x, counts, alpha=0.8, color='mediumseagreen')
                
                for i, c in enumerate(counts):
                    ax2.text(i, c + 1, str(c), ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax2.set_xlabel('文件夹', fontsize=11)
                ax2.set_ylabel('缺陷数量', fontsize=11)
                ax2.set_title(f'{selected_channel_dist}-{selected_row_type_dist} 样本数量', 
                            fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(folders_list, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # 添加箱线图对比
            if all_folder_stats and len(all_folder_stats) > 1:
                st.write("**📦 箱线图对比**")
                
                fig, ax = plt.subplots(figsize=(max(10, len(folders)*2), 6))
                
                # 收集所有数据用于箱线图
                box_data = []
                box_labels = []
                for folder in folders:
                    data = channel_data[selected_channel_dist][folder][selected_row_type_dist]
                    if data:
                        box_data.append(data)
                        box_labels.append(folder)
                
                if box_data:
                    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                                   showmeans=True, meanline=True)
                    
                    # 美化箱线图
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.7)
                    
                    for median in bp['medians']:
                        median.set(color='red', linewidth=2)
                    
                    for mean in bp['means']:
                        mean.set(color='green', linewidth=2, linestyle='--')
                    
                    ax.set_xlabel('文件夹', fontsize=12)
                    ax.set_ylabel('饱和像素占比 (%)', fontsize=12)
                    ax.set_title(f'{selected_channel_dist} 通道 - {selected_row_type_dist} - 箱线图对比', 
                               fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # 添加图例
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', linewidth=2, label='中位数'),
                        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='平均值')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
        elif distribution_type == "按文件夹分组":
            # 按文件夹分组显示
            st.write("**不同文件夹的饱和占比分布对比**")
            
            # 选择行类型
            selected_row_type = st.selectbox("选择行类型", ['主行', '次行1', '次行2'], key='dist_row_type')
            
            folders = sorted(folder_data.keys())
            n_folders = len(folders)
            
            # 动态计算子图布局
            n_cols = min(3, n_folders)
            n_rows = (n_folders + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_folders == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_folders > 1 else [axes]
            
            for idx, folder in enumerate(folders):
                ax = axes[idx]
                
                # 获取数据
                data = folder_data[folder][selected_row_type]
                
                if data:
                    # 绘制直方图
                    n, bins, patches = ax.hist(data, bins=15, range=(0, 100), 
                                              alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
                    
                    # 统计信息
                    mean_val = np.mean(data)
                    median_val = np.median(data)
                    
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值: {mean_val:.1f}%')
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, label=f'中位: {median_val:.1f}%')
                    
                    ax.set_xlabel('饱和占比 (%)', fontsize=10)
                    ax.set_ylabel('数量', fontsize=10)
                    ax.set_title(f'{folder}\n{selected_row_type} (n={len(data)})', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
            
            # 隐藏多余的子图
            for idx in range(n_folders, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        elif distribution_type == "按通道分组":
            # 按通道分组显示
            st.write("**不同通道的饱和占比分布对比**")
            
            # 选择行类型
            selected_row_type = st.selectbox("选择行类型", ['主行', '次行1', '次行2'], key='dist_row_type_channel')
            
            channels = sorted(channel_data.keys())
            
            fig, axes = plt.subplots(1, len(channels), figsize=(6*len(channels), 5))
            if len(channels) == 1:
                axes = [axes]
            
            colors_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            for idx, channel in enumerate(channels):
                ax = axes[idx]
                
                # 收集该通道所有文件夹的数据
                channel_all_data = []
                for folder in channel_data[channel].keys():
                    channel_all_data.extend(channel_data[channel][folder][selected_row_type])
                
                if channel_all_data:
                    # 绘制直方图
                    n, bins, patches = ax.hist(channel_all_data, bins=20, range=(0, 100), 
                                              alpha=0.7, color=colors_palette[idx % len(colors_palette)], 
                                              edgecolor='black', linewidth=0.5)
                    
                    # 统计信息
                    mean_val = np.mean(channel_all_data)
                    median_val = np.median(channel_all_data)
                    std_val = np.std(channel_all_data)
                    
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_val:.1f}%')
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'中位: {median_val:.1f}%')
                    
                    ax.set_xlabel('饱和占比 (%)', fontsize=11)
                    ax.set_ylabel('缺陷数量', fontsize=11)
                    ax.set_title(f'{channel} 通道 - {selected_row_type}\n(n={len(channel_all_data)})', 
                               fontsize=12, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 添加统计文本框
                    textstr = f'均值: {mean_val:.1f}%\n中位: {median_val:.1f}%\n标准差: {std_val:.1f}%'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.65, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:  # 按行类型分组
            # 按行类型分组，对比不同情况
            st.write("**不同行类型的饱和占比分布对比**")
            
            # 选择对比维度
            compare_by = st.radio("对比维度", ["文件夹", "通道"], horizontal=True, key='compare_dim')
            
            if compare_by == "文件夹":
                selected_folder = st.selectbox("选择文件夹", sorted(folder_data.keys()), key='dist_folder_comp')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                row_types = ['主行', '次行1', '次行2']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                for row_type, color in zip(row_types, colors):
                    data = folder_data[selected_folder][row_type]
                    if data:
                        ax.hist(data, bins=20, range=(0, 100), alpha=0.5, 
                               label=f'{row_type} (n={len(data)})', 
                               color=color, edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('饱和像素占比 (%)', fontsize=12)
                ax.set_ylabel('缺陷数量', fontsize=12)
                ax.set_title(f'{selected_folder} - 三行类型饱和占比分布对比', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            else:  # 通道
                selected_channel_comp = st.selectbox("选择通道", sorted(channel_data.keys()), key='dist_channel_comp')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                row_types = ['主行', '次行1', '次行2']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                for row_type, color in zip(row_types, colors):
                    # 收集该通道所有文件夹的数据
                    all_data = []
                    for folder in channel_data[selected_channel_comp].keys():
                        all_data.extend(channel_data[selected_channel_comp][folder][row_type])
                    
                    if all_data:
                        ax.hist(all_data, bins=20, range=(0, 100), alpha=0.5, 
                               label=f'{row_type} (n={len(all_data)})', 
                               color=color, edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('饱和像素占比 (%)', fontsize=12)
                ax.set_ylabel('缺陷数量', fontsize=12)
                ax.set_title(f'{selected_channel_comp} 通道 - 三行类型饱和占比分布对比', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # 7. 累积分布函数 (CDF)
        st.write("#### 7️⃣ 饱和占比累积分布图 (CDF)")
        
        cdf_type = st.radio("CDF显示类型", ["总体CDF", "按通道+行类型分组", "按文件夹对比", "按通道对比"], horizontal=True, key='cdf_type')
        
        if cdf_type == "总体CDF":
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            row_types = ['主行', '次行1', '次行2']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for idx, (row_type, color) in enumerate(zip(row_types, colors)):
                ax = axes[idx]
                
                # 收集数据
                if row_type == '主行':
                    data = [r['主行饱和占比(%)'] for r in all_results]
                elif row_type == '次行1':
                    data = [r['次行1饱和占比(%)'] for r in all_results if r['次行1像素数'] > 0]
                else:
                    data = [r['次行2饱和占比(%)'] for r in all_results if r['次行2像素数'] > 0]
                
                if data:
                    # 排序并计算CDF
                    sorted_data = np.sort(data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
                    
                    ax.plot(sorted_data, y, linewidth=2, color=color, label=row_type)
                    ax.fill_between(sorted_data, y, alpha=0.3, color=color)
                    
                    # 添加参考线
                    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50%分位')
                    ax.axhline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90%分位')
                    
                    # 标注关键分位点
                    percentiles = [50, 90, 95]
                    for p in percentiles:
                        val = np.percentile(data, p)
                        ax.plot(val, p, 'ro', markersize=8)
                        ax.text(val, p, f'  {p}%: {val:.1f}%', fontsize=9, va='center')
                    
                    ax.set_xlabel('饱和像素占比 (%)', fontsize=11)
                    ax.set_ylabel('累积百分比 (%)', fontsize=11)
                    ax.set_title(f'{row_type} - 累积分布', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        elif cdf_type == "按通道+行类型分组":
            # 特定通道+行类型在不同文件夹的CDF对比
            st.write("**特定通道+行类型在不同文件夹的累积分布对比**")
            
            # 选择通道和行类型
            col1, col2 = st.columns(2)
            with col1:
                channels_cdf = sorted(channel_data.keys())
                selected_channel_cdf = st.selectbox("选择通道", channels_cdf, key='cdf_channel_select')
            with col2:
                selected_row_type_cdf = st.selectbox("选择行类型", ['主行', '次行1', '次行2'], key='cdf_row_select')
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            folders_cdf = sorted(channel_data[selected_channel_cdf].keys())
            colors_cdf = plt.cm.tab10(np.linspace(0, 1, len(folders_cdf)))
            
            # 存储统计信息
            cdf_stats = []
            
            for folder, color in zip(folders_cdf, colors_cdf):
                data = channel_data[selected_channel_cdf][folder][selected_row_type_cdf]
                if data:
                    sorted_data = np.sort(data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
                    ax.plot(sorted_data, y, linewidth=2.5, label=f'{folder} (n={len(data)})', 
                           color=color, marker='o', markersize=3, alpha=0.8)
                    
                    # 计算关键分位数
                    p50 = np.percentile(data, 50)
                    p90 = np.percentile(data, 90)
                    cdf_stats.append({
                        'folder': folder,
                        'p50': p50,
                        'p90': p90,
                        'mean': np.mean(data),
                        'count': len(data)
                    })
            
            # 添加参考线
            ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50%分位线')
            ax.axhline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90%分位线')
            
            ax.set_xlabel('饱和像素占比 (%)', fontsize=12)
            ax.set_ylabel('累积百分比 (%)', fontsize=12)
            ax.set_title(f'{selected_channel_cdf} 通道 - {selected_row_type_cdf} - 不同文件夹累积分布对比', 
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right', ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 显示关键分位数统计表
            if cdf_stats:
                st.write("**📊 关键分位数统计**")
                cdf_stats_df = pd.DataFrame(cdf_stats)
                cdf_stats_df.columns = ['文件夹', '中位数(50%)', '90%分位数', '平均值', '样本数']
                cdf_stats_df['中位数(50%)'] = cdf_stats_df['中位数(50%)'].round(2)
                cdf_stats_df['90%分位数'] = cdf_stats_df['90%分位数'].round(2)
                cdf_stats_df['平均值'] = cdf_stats_df['平均值'].round(2)
                
                # 使用颜色突出显示
                st.dataframe(
                    cdf_stats_df.style.background_gradient(subset=['中位数(50%)', '90%分位数', '平均值'], 
                                                          cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # 添加分位数对比柱状图
                st.write("**📊 分位数对比**")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                folders_list = cdf_stats_df['文件夹'].tolist()
                p50_vals = cdf_stats_df['中位数(50%)'].tolist()
                p90_vals = cdf_stats_df['90%分位数'].tolist()
                
                x = np.arange(len(folders_list))
                width = 0.35
                
                # 左图：中位数对比
                bars1 = ax1.bar(x, p50_vals, alpha=0.8, color='steelblue')
                for i, v in enumerate(p50_vals):
                    ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
                
                ax1.set_xlabel('文件夹', fontsize=11)
                ax1.set_ylabel('中位数 (%)', fontsize=11)
                ax1.set_title(f'{selected_channel_cdf}-{selected_row_type_cdf} 中位数(50%分位)', 
                            fontsize=12, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(folders_list, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # 右图：90%分位数对比
                bars2 = ax2.bar(x, p90_vals, alpha=0.8, color='coral')
                for i, v in enumerate(p90_vals):
                    ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
                
                ax2.set_xlabel('文件夹', fontsize=11)
                ax2.set_ylabel('90%分位数 (%)', fontsize=11)
                ax2.set_title(f'{selected_channel_cdf}-{selected_row_type_cdf} 90%分位数', 
                            fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(folders_list, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
        elif cdf_type == "按文件夹对比":
            selected_row_type_cdf = st.selectbox("选择行类型", ['主行', '次行1', '次行2'], key='cdf_row_type')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            folders = sorted(folder_data.keys())
            colors_cdf = plt.cm.tab10(np.linspace(0, 1, len(folders)))
            
            for folder, color in zip(folders, colors_cdf):
                data = folder_data[folder][selected_row_type_cdf]
                if data:
                    sorted_data = np.sort(data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
                    ax.plot(sorted_data, y, linewidth=2, label=f'{folder} (n={len(data)})', color=color)
            
            ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(90, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('饱和像素占比 (%)', fontsize=12)
            ax.set_ylabel('累积百分比 (%)', fontsize=12)
            ax.set_title(f'{selected_row_type_cdf} - 不同文件夹累积分布对比', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:  # 按通道对比
            selected_row_type_cdf2 = st.selectbox("选择行类型", ['主行', '次行1', '次行2'], key='cdf_row_type2')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            channels = sorted(channel_data.keys())
            colors_cdf = plt.cm.Set2(np.linspace(0, 1, len(channels)))
            
            for channel, color in zip(channels, colors_cdf):
                # 收集该通道所有数据
                all_data = []
                for folder in channel_data[channel].keys():
                    all_data.extend(channel_data[channel][folder][selected_row_type_cdf2])
                
                if all_data:
                    sorted_data = np.sort(all_data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
                    ax.plot(sorted_data, y, linewidth=2.5, label=f'{channel} (n={len(all_data)})', color=color)
            
            ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(90, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('饱和像素占比 (%)', fontsize=12)
            ax.set_ylabel('累积百分比 (%)', fontsize=12)
            ax.set_title(f'{selected_row_type_cdf2} - 不同通道累积分布对比', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with tab4:
    st.markdown('<a name="规则编辑器"></a>', unsafe_allow_html=True)
    st.header("⚙️ 分类规则编辑器")
    
    import json
    import rule_engine
    
    # 规则文件路径
    default_rules_path = "classification_rules.json"
    
    st.subheader("📂 规则文件管理")
    
    # 添加加载方式选择
    load_method = st.radio(
        "选择加载方式",
        ["📁 从文件路径加载", "📤 上传JSON文件"],
        horizontal=True
    )
    
    rules_file_path = None
    load_button = False
    uploaded_rules = None
    
    if load_method == "📁 从文件路径加载":
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            rules_file_path = st.text_input("规则文件路径", value=default_rules_path, key="rules_path_input")
        with col2:
            st.write("")
            st.write("")
            load_button = st.button("🔄 加载规则", key="load_from_path")
        with col3:
            st.write("")
            st.write("")
            # 文件浏览器按钮提示
            if st.button("💡 提示", key="path_help"):
                st.info("💡 在文本框中输入完整的文件路径，例如：\n\n`D:/streamlit/classification_rules.json`\n\n或使用相对路径：\n\n`classification_rules.json`")
    
    else:  # 上传JSON文件
        uploaded_file = st.file_uploader(
            "选择JSON规则文件",
            type=['json'],
            help="上传classification_rules.json文件",
            key="json_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # 读取上传的文件内容
                uploaded_rules = json.load(uploaded_file)
                st.success(f"✅ 文件 '{uploaded_file.name}' 上传成功")
                
                # 显示预览
                with st.expander("📄 文件预览"):
                    st.json(uploaded_rules)
                
                # 加载按钮
                if st.button("✔️ 确认加载此文件", type="primary", key="load_uploaded"):
                    load_button = True
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON文件格式错误: {str(e)}")
            except Exception as e:
                st.error(f"❌ 读取文件失败: {str(e)}")
    
    # 初始化session state
    if 'rules_config' not in st.session_state:
        # 首次加载，尝试加载默认文件
        try:
            rules_config = rule_engine.load_rules_from_json(default_rules_path)
            if rules_config:
                st.session_state.rules_config = rules_config
                st.session_state.current_rules_source = default_rules_path
                st.info(f"ℹ️ 已自动加载默认规则文件：{default_rules_path}")
            else:
                st.warning("⚠️ 未找到默认规则文件，请加载或上传规则文件")
                st.stop()
        except:
            st.warning("⚠️ 未找到默认规则文件，请加载或上传规则文件")
            st.stop()
    
    # 处理加载操作
    if load_button:
        if load_method == "📁 从文件路径加载" and rules_file_path:
            rules_config = rule_engine.load_rules_from_json(rules_file_path)
            if rules_config:
                st.session_state.rules_config = rules_config
                st.session_state.current_rules_source = rules_file_path
                st.success(f"✅ 成功加载规则文件：{rules_file_path}")
                st.rerun()
            else:
                st.error(f"❌ 加载规则文件失败：{rules_file_path}")
                st.error("请检查文件路径是否正确，文件是否存在")
                st.stop()
        elif load_method == "📤 上传JSON文件" and uploaded_rules:
            st.session_state.rules_config = uploaded_rules
            st.session_state.current_rules_source = uploaded_file.name
            st.success(f"✅ 成功加载上传的规则文件：{uploaded_file.name}")
            st.rerun()
    
    rules_config = st.session_state.rules_config
    
    # 显示当前加载的规则来源
    current_source = st.session_state.get('current_rules_source', '未知')
    st.caption(f"📌 当前规则来源: `{current_source}`")
    
    # 显示规则文件信息
    st.subheader("📋 规则配置信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("规则版本", rules_config.get('version', 'N/A'))
    with col2:
        st.metric("规则数量", len(rules_config.get('rules', [])))
    with col3:
        enabled_count = sum(1 for r in rules_config.get('rules', []) if r.get('enabled', True))
        st.metric("已启用规则", enabled_count)
    
    st.info(f"📝 描述: {rules_config.get('description', '无描述')}")
    
    # 阈值参数设置
    st.subheader("🎛️ 全局阈值参数")
    thresholds = rules_config.get('thresholds', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        snr_adj = st.number_input("SNR调整值", 
                                   value=float(thresholds.get('snr_adjustment', 0)),
                                   step=0.5,
                                   format="%.1f")
        thresholds['snr_adjustment'] = snr_adj
    with col2:
        dw1o_adj = st.number_input("DW1O峰值调整", 
                                     value=float(thresholds.get('dw1o_peak_adjustment', 0)),
                                     step=100.0,
                                     format="%.0f")
        thresholds['dw1o_peak_adjustment'] = dw1o_adj
    with col3:
        dw2o_adj = st.number_input("DW2O峰值调整", 
                                     value=float(thresholds.get('dw2o_peak_adjustment', 0)),
                                     step=100.0,
                                     format="%.0f")
        thresholds['dw2o_peak_adjustment'] = dw2o_adj
    
    rules_config['thresholds'] = thresholds
    
    # 默认返回值设置
    st.subheader("🔢 默认返回值")
    default_return = st.number_input("当没有规则匹配时的返回值", 
                                     value=int(rules_config.get('default_return', 10002)),
                                     step=1)
    rules_config['default_return'] = default_return
    
    st.markdown("---")
    
    # 规则列表编辑
    st.subheader("📜 分类规则列表")
    
    # 添加新规则按钮
    if st.button("➕ 添加新规则"):
        new_rule = {
            "rule_id": max([r.get('rule_id', 0) for r in rules_config['rules']], default=0) + 1,
            "name": "新规则",
            "conditions": [],
            "logic": "AND",
            "return_value": 0,
            "enabled": True
        }
        rules_config['rules'].append(new_rule)
        st.success("✅ 已添加新规则")
        st.rerun()
    
    # 可用特征列表
    available_features = rules_config.get('available_features', [])
    operators = ['>', '>=', '<', '<=', '==', '!=']
    
    # 通道组合映射（内部值 -> 显示名称）
    channel_combinations_map = {
        '': '无限制',
        'D_only': 'DW1O通道单独',
        'J_only': 'DW2O通道单独',
        'P_only': 'DN1O通道单独',
        'D_and_J': 'DW1O+DW2O组合',
        'D_and_P': 'DW1O+DN1O组合',
        'J_and_P': 'DW2O+DN1O组合',
        'D_and_J_and_P': 'DW1O+DW2O+DN1O全通道'
    }
    channel_combinations = list(channel_combinations_map.keys())
    
    # 显示每条规则
    rules_to_delete = []
    for idx, rule in enumerate(rules_config['rules']):
        with st.expander(f"🔖 规则 {rule.get('rule_id', idx+1)}: {rule.get('name', '未命名')} {'✅' if rule.get('enabled', True) else '❌'}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                rule['name'] = st.text_input("规则名称", value=rule.get('name', ''), key=f"name_{idx}")
            
            with col2:
                rule['enabled'] = st.checkbox("启用", value=rule.get('enabled', True), key=f"enabled_{idx}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rule['rule_id'] = st.number_input("规则ID", value=int(rule.get('rule_id', idx+1)), 
                                                  step=1, key=f"id_{idx}")
            with col2:
                rule['return_value'] = st.number_input("返回值", value=int(rule.get('return_value', 0)), 
                                                       step=1, key=f"return_{idx}")
            with col3:
                # 选择逻辑模式
                use_complex_logic = st.checkbox("使用复杂逻辑表达式", 
                                               value='logic_expression' in rule,
                                               key=f"complex_{idx}",
                                               help="启用后可以使用 &&、||、! 和括号组合条件")
            
            # 通道组合（可选）
            current_combination = rule.get('channel_combination', '')
            combination_index = channel_combinations.index(current_combination) if current_combination in channel_combinations else 0
            
            # 使用中文显示名称
            selected_display = st.selectbox(
                "通道组合限制（可选）", 
                options=channel_combinations,
                format_func=lambda x: channel_combinations_map.get(x, x),
                index=combination_index,
                key=f"channel_{idx}",
                help="限制规则仅在特定通道组合下生效"
            )
            
            if selected_display:
                rule['channel_combination'] = selected_display
            elif 'channel_combination' in rule:
                del rule['channel_combination']
            
            # 逻辑设置
            if use_complex_logic:
                # 使用复杂逻辑表达式
                st.info("💡 复杂逻辑表达式说明：使用条件ID组合，支持 && (AND)、|| (OR)、! (NOT) 和括号")
                st.markdown("""
                **示例**：
                - `1 && 2` : 条件1 AND 条件2
                - `1 || 2 || 3` : 条件1 OR 条件2 OR 条件3
                - `1 && (2 || 3)` : 条件1 AND (条件2 OR 条件3)
                - `(1 || 2) && !3` : (条件1 OR 条件2) AND NOT 条件3
                - `1 && (2 || 3 || 4) && (!5)` : 条件1 AND (条件2 OR 条件3 OR 条件4) AND (NOT 条件5)
                """)
                
                current_expression = rule.get('logic_expression', '')
                rule['logic_expression'] = st.text_input(
                    "逻辑表达式", 
                    value=current_expression,
                    key=f"logic_expr_{idx}",
                    placeholder="例如: 1 && (2 || 3) && (!4)"
                )
                
                # 删除简单逻辑字段
                if 'logic' in rule:
                    del rule['logic']
            else:
                # 使用简单逻辑
                rule['logic'] = st.selectbox("逻辑关系", ['AND', 'OR'], 
                                            index=0 if rule.get('logic', 'AND') == 'AND' else 1,
                                            key=f"logic_{idx}",
                                            help="AND: 所有条件都满足, OR: 任一条件满足")
                
                # 删除复杂逻辑字段
                if 'logic_expression' in rule:
                    del rule['logic_expression']
            
            # 条件列表
            st.write("**条件列表:**")
            
            conditions = rule.get('conditions', [])
            conditions_to_delete = []
            use_complex = 'logic_expression' in rule
            
            for cond_idx, condition in enumerate(conditions):
                # 如果使用复杂逻辑，显示条件ID
                if use_complex:
                    col0, col1, col2, col3, col4, col5 = st.columns([0.5, 2.5, 1, 2, 1, 1])
                    with col0:
                        # 确保有condition_id
                        if 'condition_id' not in condition:
                            condition['condition_id'] = cond_idx + 1
                        condition['condition_id'] = st.number_input("ID", 
                                                                    value=int(condition.get('condition_id', cond_idx+1)),
                                                                    min_value=1,
                                                                    step=1,
                                                                    key=f"cond_id_{idx}_{cond_idx}")
                else:
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 1, 1])
                    # 移除condition_id（简单逻辑不需要）
                    if 'condition_id' in condition:
                        del condition['condition_id']
                
                with col1:
                    feature_index = available_features.index(condition['feature']) if condition['feature'] in available_features else 0
                    condition['feature'] = st.selectbox("特征", available_features, 
                                                       index=feature_index,
                                                       key=f"feat_{idx}_{cond_idx}")
                
                with col2:
                    op_index = operators.index(condition['operator']) if condition['operator'] in operators else 0
                    condition['operator'] = st.selectbox("操作符", operators, 
                                                        index=op_index,
                                                        key=f"op_{idx}_{cond_idx}")
                
                with col3:
                    condition['value'] = st.number_input("值", value=float(condition['value']), 
                                                        step=0.1,
                                                        key=f"val_{idx}_{cond_idx}")
                
                with col4:
                    condition['use_threshold'] = st.checkbox("使用阈值", 
                                                            value=condition.get('use_threshold', False),
                                                            key=f"thresh_{idx}_{cond_idx}")
                
                with col5:
                    if st.button("🗑️", key=f"del_cond_{idx}_{cond_idx}"):
                        conditions_to_delete.append(cond_idx)
            
            # 删除标记的条件
            for cond_idx in sorted(conditions_to_delete, reverse=True):
                conditions.pop(cond_idx)
            
            rule['conditions'] = conditions
            
            # 添加新条件按钮
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("➕ 添加条件", key=f"add_cond_{idx}"):
                    new_condition = {
                        "feature": available_features[0] if available_features else "",
                        "operator": ">",
                        "value": 0,
                        "use_threshold": False
                    }
                    # 如果使用复杂逻辑，添加condition_id
                    if 'logic_expression' in rule:
                        # 找到最大的condition_id
                        max_id = max([c.get('condition_id', 0) for c in conditions], default=0)
                        new_condition['condition_id'] = max_id + 1
                    conditions.append(new_condition)
                    st.rerun()
            
            with col2:
                if st.button("❌ 删除此规则", key=f"del_rule_{idx}"):
                    rules_to_delete.append(idx)
                    st.rerun()
    
    # 删除标记的规则
    for rule_idx in sorted(rules_to_delete, reverse=True):
        rules_config['rules'].pop(rule_idx)
    
    st.markdown("---")
    
    # 保存按钮
    st.subheader("💾 保存规则")
    
    # 保存方式选择
    save_method = st.radio(
        "选择保存方式",
        ["💾 保存到文件路径", "⬇️ 下载JSON文件"],
        horizontal=True,
        key="save_method"
    )
    
    if save_method == "💾 保存到文件路径":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 获取默认保存路径
            default_save_path = st.session_state.get('current_rules_source', default_rules_path)
            if not default_save_path.endswith('.json'):
                default_save_path = default_rules_path
            
            save_path = st.text_input(
                "保存文件路径", 
                value=default_save_path, 
                key="save_path",
                help="输入完整的文件路径，例如：D:/streamlit/my_rules.json"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 保存", type="primary", key="save_to_file"):
                st.session_state.rules_config = rules_config
                if rule_engine.save_rules_to_json(rules_config, save_path):
                    st.success(f"✅ 规则已成功保存到:\n`{save_path}`")
                    st.session_state.current_rules_source = save_path
                    st.balloons()
                else:
                    st.error("❌ 保存失败，请检查文件路径是否正确")
        
        st.info("💡 **提示**: 保存后，您可以在`离线过漏检.py`中使用此规则文件")
    
    else:  # 下载JSON文件
        st.write("点击下方按钮下载规则文件到本地：")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            download_filename = st.text_input(
                "文件名", 
                value="classification_rules_export.json",
                key="download_filename",
                help="设置下载的文件名"
            )
        
        with col2:
            st.write("")
            st.write("")
            # 生成JSON字符串
            json_str = json.dumps(rules_config, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ 下载JSON",
                data=json_str,
                file_name=download_filename,
                mime="application/json",
                type="primary",
                key="download_json"
            )
        
        st.info("💡 **提示**: 下载后，您可以将文件放置到工作目录，然后在规则编辑器中重新加载")
    
    st.markdown("---")
    
    # 规则预览
    st.subheader("👁️ 规则JSON预览")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        show_full_json = st.checkbox("显示完整JSON", value=False, key="show_full_json")
    with col2:
        if st.button("📋 复制JSON到剪贴板", key="copy_json_btn"):
            st.code(json.dumps(rules_config, ensure_ascii=False, indent=2), language="json")
            st.info("💡 请选中上方代码框的内容，然后按 Ctrl+C 复制")
    
    if show_full_json:
        st.json(rules_config)
    else:
        with st.expander("点击展开查看完整JSON"):
            st.json(rules_config)