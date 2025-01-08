from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob
import streamlit as st
import time  # 用于模拟进度更新

class SentimentAnalysis:
    def __init__(self, df):
        """
        初始化情绪分析类
        """
        self.df = df
        self.vader_analyzer = SentimentIntensityAnalyzer()  # VADER 初始化

        # 尝试初始化 DistilBERT，并捕获潜在的错误
        try:
            self.distilbert_analyzer = pipeline("sentiment-analysis", framework="pt")  # 强制使用 PyTorch 框架
        except Exception as e:
            st.warning("DistilBERT 初始化失败，可能是环境问题导致。请确保安装了兼容的 PyTorch 或 TensorFlow。")
            self.distilbert_analyzer = None

    def analyze_sentiment(self):
        """
        执行情绪分析
        """
        try:
            # 检查是否有文本列
            text_cols = self.df.select_dtypes(include=['object']).columns
            if len(text_cols) == 0:
                st.warning("数据集中没有文本列，无法进行情绪分析。")
                return
            
            text_col = st.selectbox("选择文本列进行分析", text_cols)
            text_data = self.df[text_col].dropna()

            # 用户选择情绪分析方法
            sentiment_option = st.selectbox("选择情绪分析方法", ["VADER", "TextBlob", "DistilBERT"])

            st.write(f"正在对列 `{text_col}` 使用 {sentiment_option} 方法进行情绪分析...")

            # 创建进度条
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # 按行分析情绪
            results = []
            total = len(text_data)
            for i, text in enumerate(text_data):
                progress_text.text(f"正在处理文本 ({i + 1}/{total})...")
                if sentiment_option == "VADER":
                    results.append(self._vader_analysis(text))
                elif sentiment_option == "TextBlob":
                    results.append(self._textblob_analysis(text))
                elif sentiment_option == "DistilBERT":
                    if self.distilbert_analyzer is not None:
                        results.append(self._distilbert_analysis(text))
                    else:
                        st.error("DistilBERT 无法初始化，无法进行分析。")
                        return
                
                # 更新进度条
                progress_bar.progress((i + 1) / total)
                time.sleep(0.1)  # 模拟处理延迟

            # 完成进度条
            progress_text.text("情绪分析完成！")
            progress_bar.empty()

            # 显示分析结果
            st.markdown("### 分析结果")
            st.write(f"共分析了 {len(results)} 条文本数据。")
            st.dataframe(results)

        except Exception as e:
            st.error("情绪分析过程中发生错误，请检查输入数据是否正确。")
            st.write(f"错误详细信息: {e}")

    def _vader_analysis(self, text):
        """
        使用 VADER 进行情绪分析
        """
        sentiment = self.vader_analyzer.polarity_scores(text)
        if sentiment['compound'] >= 0.05:
            label = "积极"
        elif sentiment['compound'] <= -0.05:
            label = "消极"
        else:
            label = "中性"
        return {"文本": text, "情绪得分": sentiment, "情绪类型": label}

    def _textblob_analysis(self, text):
        """
        使用 TextBlob 进行情绪分析
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            label = "积极"
        elif polarity < 0:
            label = "消极"
        else:
            label = "中性"
        return {"文本": text, "情绪得分": {"polarity": polarity}, "情绪类型": label}

    def _distilbert_analysis(self, text):
        """
        使用 DistilBERT 进行情绪分析
        """
        result = self.distilbert_analyzer(text)[0]
        label = result['label']
        score = result['score']
        if label == "POSITIVE":
            label = "积极"
        elif label == "NEGATIVE":
            label = "消极"
        else:
            label = "中性"
        return {"文本": text, "情绪得分": {"score": score, "label": label}, "情绪类型": label}
