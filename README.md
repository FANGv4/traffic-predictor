# 客流预测小工具（线性回归）

一个看起来“很专业”的小预测器：输入去年的客流/销售数据，点击按钮，即可给出下周的预测值（使用 `scikit-learn` 的线性回归）。

现在已升级为多页面：

- `app.py`：官网式首页（功能介绍/入口）
- `pages/1_预测器.py`：预测器主功能
- `pages/2_方法与指标.py`：方法与指标解释

## 运行

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Cloud（最快上线）

1. 把本项目推送到 GitHub（需要包含：`app.py`、`requirements.txt`、`.streamlit/config.toml`、`runtime.txt`）
2. 打开 `https://share.streamlit.io/` → New app
3. 选择仓库与分支，Main file path 填 `app.py`
4. 点击 Deploy，等待构建完成即可获得公开访问链接

## 使用方式

- **方式 1：手动粘贴数据**：把历史数据按行或用逗号分隔粘贴进输入框（例如 365 个“每天”的数值，或 52 个“每周”的数值）。
- **方式 2：上传 CSV**：两列即可（日期 + 数值）。
  - 推荐列名：`date,value`（大小写不敏感）
  - 也支持“前两列分别是日期和数值”的 CSV

## 预测逻辑（简化版）

- 使用线性回归拟合趋势：`y = a * t + b`
- **按天数据**：预测未来 7 天并求和，得到“下周总量”
- **按周数据**：直接预测“下一周”值
- 额外展示：拟合趋势图、简要指标（R²、MAE）、基于残差的粗略区间（用于展示“可信范围”，不是严格置信区间）
