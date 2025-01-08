数据分析程序

基础统计信息展示

* 数据分布可视化
* 方差分析
* t-检验
* 回归分析
* 卡方检验
* 情绪分析

## 特性

1. **高端界面**：使用 Streamlit 开发，界面简洁美观。
2. **灵活分析**：支持多种分析方法，适用于不同数据集。
3. **交互图表**：图表支持缩小显示，点击可以放大查看。
4. **多样化情绪分析**：支持 VADER、TextBlob 和 DistilBERT 等情绪分析方法。

---

## 安装与运行

### 1. 克隆项目

```
git clone https://github.com/your-repository/data-analysis-program.git
cd data-analysis-program
```

### 2. 安装依赖

建议使用虚拟环境来管理依赖，确保项目运行不受其他环境影响。

#### 创建虚拟环境

```
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate     # Windows
```

#### 安装依赖

```
pip install -r requirements.txt
```

### 3. 运行程序

```
streamlit run app.py
```

打开浏览器，访问 `<span>http://localhost:8501</span>`，即可看到项目界面。

---

## 功能介绍

### 1. 基础统计展示

* 查看数据集前 5 行
* 展示每列的基本统计信息（如均值、中位数、众数、标准差、偏度、峰度等）

### 2. 数据分布可视化

* 支持直方图、箱线图、条形图等多种可视化方式
* 图表支持缩小显示，点击可放大查看

### 3. 方差分析

* 分析分类变量与连续变量之间的关系
* 自动判断正态性，选择适当的分析方法（方差分析或 Kruskal-Wallis 检验）

### 4. t-检验

* 对分类变量与连续变量进行独立样本 t 检验
* 自动检测正态性，提供非正态分布的替代检验方法（如 Mann-Whitney U 检验）

### 5. 回归分析

* 选择自变量和因变量，生成回归模型
* 支持回归模型的可视化（散点图与拟合线）

### 6. 卡方检验

* 分析两个分类变量之间的相关性
* 提供检验统计量和显著性水平

### 7. 情绪分析

* 自动检测文本列，支持以下情绪分析方法：
  * **VADER**：基于规则的情绪分析
  * **TextBlob**：基于极性分析
  * **DistilBERT**：基于深度学习的情绪分析

---

## 文件结构

```
data-analysis-program/
├── app.py                   # 主程序入口
├── data_checker.py          # 数据检查与预处理模块
├── data_analysis.py         # 数据分析模块
├── sentiment_analysis.py    # 情绪分析模块
├── requirements.txt         # 项目依赖
├── README.md                # 项目说明文档
└── assets/
    └── sample_data.csv      # 示例数据文件
```

---

## 依赖项

* `<span>streamlit</span>`：快速构建交互式 Web 应用
* `<span>pandas</span>`：数据处理和分析
* `<span>matplotlib</span>`：数据可视化
* `<span>seaborn</span>`：高级数据可视化
* `<span>vaderSentiment</span>`：基于规则的情绪分析
* `<span>textblob</span>`：基于极性分析的情绪分析
* `<span>transformers</span>`：基于深度学习的自然语言处理库
* `<span>torch</span>`：PyTorch 深度学习框架
