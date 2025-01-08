import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu, kruskal
from statsmodels.graphics.gofplots import qqplot
import streamlit as st
import pandas as pd
import numpy as np

class DataAnalysis:
    def __init__(self, df):
        self.df = df

    def chi_square_test(self, cat1, cat2):
        """卡方检验"""
        try:
            # 检查输入变量是否存在
            if cat1 not in self.df.columns or cat2 not in self.df.columns:
                st.error("所选变量不在数据集中，请检查输入。")
                return

            # 获取非空数据
            data = self.df[[cat1, cat2]].dropna()

            # 检查分类变量类别数
            if data[cat1].nunique() < 2 or data[cat2].nunique() < 2:
                st.error("每个分类变量至少需要包含两个类别。")
                return

            # 执行卡方检验
            cross_tab = pd.crosstab(data[cat1], data[cat2])
            chi2_stat, p_val, _, _ = stats.chi2_contingency(cross_tab)

            # 显示结果
            st.markdown(f"**卡方检验结果:**")
            st.markdown(f"**χ²统计量:** {chi2_stat:.4f}")
            st.markdown(f"**p值:** {p_val:.4f}")
            if p_val < 0.05:
                st.markdown("**两个分类变量之间具有显著相关性（拒绝零假设）。**")
            else:
                st.markdown("**两个分类变量之间没有显著相关性（未能拒绝零假设）。**")

        except Exception as e:
            st.error("卡方检验过程中发生错误，请检查数据。")
            st.write(f"错误详细信息: {e}")


    def variance_analysis(self, category_col, numeric_col):
        """进行方差分析"""
        try:
            data = self.df[[category_col, numeric_col]].dropna()
            numeric_data = data[numeric_col]

            # 正态性检验
            if len(numeric_data) > 2000:
                test_name = "Anderson-Darling检验"
                result = stats.anderson(numeric_data)
                stat, p_value = result.statistic, result.significance_level[0] / 100
            else:
                test_name = "Shapiro-Wilk检验"
                stat, p_value = stats.shapiro(numeric_data)

            st.markdown(f"**{test_name}结果:** 统计量 = **{stat:.4f}**, p值 = **{p_value:.4f}**")
            if p_value > 0.05:
                st.markdown("**数据呈正态分布，可以进行方差分析。**")
                # 方差分析
                groups = [data[numeric_col][data[category_col] == group] for group in data[category_col].unique()]
                f_stat, p_val = stats.f_oneway(*groups)
                st.markdown(f"**方差分析结果:** F统计量 = **{f_stat:.4f}**, p值 = **{p_val:.4f}**")
                if p_val < 0.05:
                    st.markdown("**组间均值存在显著差异（拒绝零假设）。**")
                else:
                    st.markdown("**组间均值不存在显著差异（未能拒绝零假设）。**")
            else:
                st.markdown("**数据不呈正态分布，使用非参数检验（Kruskal-Wallis检验）。**")
                groups = [data[numeric_col][data[category_col] == group] for group in data[category_col].unique()]
                h_stat, p_val = stats.kruskal(*groups)
                st.markdown(f"**Kruskal-Wallis检验结果:** H统计量 = **{h_stat:.4f}**, p值 = **{p_val:.4f}**")
                if p_val < 0.05:
                    st.markdown("**组间均值存在显著差异（拒绝零假设）。**")
                else:
                    st.markdown("**组间均值不存在显著差异（未能拒绝零假设）。**")

            # 绘制箱线图
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=category_col, y=numeric_col, data=data)
            plt.title(f"{category_col} 对 {numeric_col} 的箱线图")
            st.pyplot(plt)

            # 绘制Q-Q图
            qqplot(numeric_data, line="s")
            plt.title("Q-Q图")
            st.pyplot(plt)
        except Exception as e:
            st.error("在方差分析过程中发生错误，请检查输入的数据是否正确。")
            st.write(f"错误详细信息: {e}")

    def t_test(self, category_col, numeric_col):
        """进行t检验"""
        try:
            data = self.df[[category_col, numeric_col]].dropna()
            numeric_data = data[numeric_col]

            # 正态性检验
            if len(numeric_data) > 2000:
                test_name = "Anderson-Darling检验"
                result = stats.anderson(numeric_data)
                stat, p_value = result.statistic, result.significance_level[0] / 100
            else:
                test_name = "Shapiro-Wilk检验"
                stat, p_value = stats.shapiro(numeric_data)

            st.markdown(f"**{test_name}结果:** 统计量 = **{stat:.4f}**, p值 = **{p_value:.4f}**")
            if p_value > 0.05:
                st.markdown("**数据呈正态分布，可以进行t检验。**")
                # 执行t检验
                groups = [data[numeric_col][data[category_col] == group] for group in data[category_col].unique()]
                t_stat, p_val = stats.ttest_ind(*groups)
                st.markdown(f"**t检验结果:** t统计量 = **{t_stat:.4f}**, p值 = **{p_val:.4f}**")
                if p_val < 0.05:
                    st.markdown("**组间均值存在显著差异（拒绝零假设）。**")
                else:
                    st.markdown("**组间均值不存在显著差异（未能拒绝零假设）。**")
            else:
                st.markdown("**数据不呈正态分布，建议使用非参数检验（Mann-Whitney U检验）。**")
                groups = [data[numeric_col][data[category_col] == group] for group in data[category_col].unique()]
                u_stat, p_val = stats.mannwhitneyu(*groups)
                st.markdown(f"**Mann-Whitney U检验结果:** U统计量 = **{u_stat:.4f}**, p值 = **{p_val:.4f}**")
                if p_val < 0.05:
                    st.markdown("**组间均值存在显著差异（拒绝零假设）。**")
                else:
                    st.markdown("**组间均值不存在显著差异（未能拒绝零假设）。**")

            # 绘制箱线图
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=category_col, y=numeric_col, data=data)
            plt.title(f"{category_col} 对 {numeric_col} 的箱线图")
            st.pyplot(plt)

            # 绘制Q-Q图
            qqplot(numeric_data, line="s")
            plt.title("Q-Q图")
            st.pyplot(plt)
        except Exception as e:
            st.error("在t检验过程中发生错误，请检查输入的数据是否正确。")
            st.write(f"错误详细信息: {e}")

    def regression_analysis(self, independent_var, dependent_var):
        """进行回归分析"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            data = self.df[[independent_var, dependent_var]].dropna()
            X = data[[independent_var]]
            y = data[dependent_var]

            # 创建线性回归模型
            model = LinearRegression()
            model.fit(X, y)

            # 回归结果
            intercept = model.intercept_
            coef = model.coef_[0]
            r2 = r2_score(y, model.predict(X))
            st.markdown(f"**回归方程:** y = **{intercept:.4f} + {coef:.4f} * x**")
            st.markdown(f"**R²值:** {r2:.4f}（模型解释了目标变量的 {r2*100:.2f}% 变异）")

            # 绘制散点图与回归线
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=independent_var, y=dependent_var, data=data, label="实际值")
            plt.plot(data[independent_var], model.predict(X), color="red", label="拟合线")
            plt.title(f"{independent_var} 对 {dependent_var} 的回归分析")
            plt.xlabel(independent_var)
            plt.ylabel(dependent_var)
            plt.legend()
            st.pyplot(plt)

            # 绘制Q-Q图
            residuals = y - model.predict(X)
            qqplot(residuals, line="s")
            plt.title("残差的Q-Q图")
            st.pyplot(plt)

        except Exception as e:
            st.error("在回归分析过程中发生错误，请检查输入的数据是否正确。")
            st.write(f"错误详细信息: {e}")
