import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import streamlit as st

class DataChecker:
    def __init__(self, df):
        self.df = df

    def load_csv(self, file_path):
        """加载CSV数据集"""
        self.df = pd.read_csv(file_path)
        return self.df

    def column_statistics(self):
        """计算并显示每一列的统计信息"""
        for col in self.df.columns:
            st.write(f"列: {col}")
            st.write(f"数据类型: {self.df[col].dtype}")
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                st.write(f"均值: {self.df[col].mean()}")
                st.write(f"中位数: {self.df[col].median()}")
                st.write(f"众数: {self.df[col].mode()[0]}")
                st.write(f"标准差: {self.df[col].std()}")
                st.write(f"偏度: {skew(self.df[col], nan_policy='omit')}")
                st.write(f"峰度: {kurtosis(self.df[col], nan_policy='omit')}")
            elif pd.api.types.is_object_dtype(self.df[col]):
                st.write(f"众数: {self.df[col].mode()[0]}")
            st.write("\n")

    def plot_variables(self):
        """显示所有可用于可视化的列"""
        return self.df.columns

    def plot_distribution(self, column):
        """根据数据类型绘制分布图"""
        data = self.df[column].dropna()
        
        if pd.api.types.is_numeric_dtype(self.df[column]):
            sns.histplot(data, kde=True)
            plt.title(f"直方图：{column}")
            st.pyplot()
        elif pd.api.types.is_object_dtype(self.df[column]):
            sns.countplot(x=data)
            plt.title(f"条形图：{column}")
            st.pyplot()
