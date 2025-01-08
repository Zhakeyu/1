import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_checker import DataChecker
from data_analysis import DataAnalysis
from sentiment_analysis import SentimentAnalysis

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 全局页面配置
st.set_page_config(
    page_title="数据分析程序",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # 页面顶部标题与背景
    st.markdown(
        """
        <style>
            .main-header {
                background-color: #1f77b4;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                font-size: 1.5em;
            }
            .card {
                background-color: #f4f4f4;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        <div class="main-header">
            数据分析程序
        </div>
        """, unsafe_allow_html=True
    )

    # 侧边栏上传数据集
    st.sidebar.title("菜单")
    uploaded_file = st.sidebar.file_uploader("上传数据集 (CSV格式)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # 数据集信息卡片
        st.markdown(
            f"""
            <div class="card">
                <b>数据集名称：</b> {uploaded_file.name} <br>
                <b>数据集大小：</b> {df.shape[0]} 行, {df.shape[1]} 列
            </div>
            """, unsafe_allow_html=True
        )

        # 初始化工具类
        data_checker = DataChecker(df)
        data_analysis = DataAnalysis(df)
        sentiment_analysis = SentimentAnalysis(df)

        # 侧边栏菜单选项
        analysis_option = st.sidebar.selectbox(
            "选择分析方式",
            ["基础展示", "绘制分布图", "方差分析", "t-检验", "回归分析", "卡方检验", "情绪分析", "退出"]
        )

        # 基础展示
        if analysis_option == "基础展示":
            st.markdown("<h3>基础统计信息</h3>", unsafe_allow_html=True)

            # 数据统计信息整理为 DataFrame
            stats_data = []
            for col in df.columns:
                col_data = {
                    "列名": col,
                    "数据类型": str(df[col].dtype),
                    "均值": round(df[col].mean(), 2) if pd.api.types.is_numeric_dtype(df[col]) else "-",
                    "中位数": round(df[col].median(), 2) if pd.api.types.is_numeric_dtype(df[col]) else "-",
                    "众数": df[col].mode()[0] if not df[col].mode().empty else "-",
                    "标准差": round(df[col].std(), 2) if pd.api.types.is_numeric_dtype(df[col]) else "-",
                    "偏度": round(df[col].skew(), 2) if pd.api.types.is_numeric_dtype(df[col]) else "-",
                    "峰度": round(df[col].kurt(), 2) if pd.api.types.is_numeric_dtype(df[col]) else "-",
                }
                stats_data.append(col_data)

            stats_df = pd.DataFrame(stats_data)

            # 展示前5行数据
            st.write("数据预览（前5行）：")
            st.dataframe(df.head(), use_container_width=True)

            # 以表格形式展示统计信息
            st.write("每列的统计信息：")
            st.dataframe(stats_df, use_container_width=True)

        # 绘制分布图
        elif analysis_option == "绘制分布图":
            st.markdown("<h3>绘制分布图</h3>", unsafe_allow_html=True)
            column = st.selectbox("选择要绘制的列", df.columns)

            # 创建缩小图表
            fig, ax = plt.subplots(figsize=(6, 4))
            data_checker.plot_distribution(column)
            st.pyplot(fig)

            # 提供放大功能
            with st.expander("点击查看大图"):
                fig_full, ax_full = plt.subplots(figsize=(12, 8))  # 放大图表
                data_checker.plot_distribution(column)
                st.pyplot(fig_full)

        # 方差分析
        elif analysis_option == "方差分析":
            st.markdown("<h3>方差分析</h3>", unsafe_allow_html=True)
            category_col = st.selectbox("选择分类变量", df.select_dtypes(include=['object']).columns)
            numeric_col = st.selectbox("选择连续变量", df.select_dtypes(include=['float64', 'int64']).columns)
            data_analysis.variance_analysis(category_col, numeric_col)

        # t-检验
        elif analysis_option == "t-检验":
            st.markdown("<h3>t-检验</h3>", unsafe_allow_html=True)
            category_col = st.selectbox("选择分类变量", df.select_dtypes(include=['object']).columns)
            numeric_col = st.selectbox("选择连续变量", df.select_dtypes(include=['float64', 'int64']).columns)
            data_analysis.t_test(category_col, numeric_col)

        # 回归分析
        elif analysis_option == "回归分析":
            st.markdown("<h3>回归分析</h3>", unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            independent_var = st.selectbox("选择自变量", numeric_cols)
            dependent_var = st.selectbox("选择因变量", numeric_cols)
            data_analysis.regression_analysis(independent_var, dependent_var)

        # 卡方检验
        elif analysis_option == "卡方检验":
            st.markdown("<h3>卡方检验</h3>", unsafe_allow_html=True)
            category_cols = df.select_dtypes(include=['object']).columns
            if len(category_cols) < 2:
                st.write("数据中没有足够的分类变量来进行卡方检验。")
            else:
                cat1 = st.selectbox("选择第一个分类变量", category_cols)
                cat2 = st.selectbox("选择第二个分类变量", category_cols)
                data_analysis.chi_square_test(cat1, cat2)

        # 情绪分析
        elif analysis_option == "情绪分析":
            st.markdown("<h3>情绪分析</h3>", unsafe_allow_html=True)
            sentiment_analysis.analyze_sentiment()

        # 退出
        elif analysis_option == "退出":
            st.stop()

if __name__ == "__main__":
    main()
