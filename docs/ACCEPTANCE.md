# 第一版验收（由用户运行）

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m compileall -q scripts app.py
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py
.\.venv\Scripts\python.exe scripts\verify_refactor.py
.\.venv\Scripts\python.exe scripts\predict.py --image data/mvtec_ad/bottle/test/good/006.png
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. verify_refactor 输出 PASS，与归档基线 83 张图的预测及指标一致。
2. CLI 的 006.png 输出约 0.474999，anomaly。
3. 打开 http://127.0.0.1:8501，上传 006.png 点击检测；页面分数与 CLI 一致。
4. 页面三张图正常显示，PNG、JSON 可下载。
5. 上传 014.png 应显示约 0.228893、正常；再上传 broken_large/000.png 检查缺陷展示。
6. 损坏 PNG 应给出可读错误；无上传时不自动加载模型。
7. 保存正常、缺陷页面截图为 docs/images/demo_normal.png、demo_defect.png。
8. Ctrl+C 停止页面；反馈验收结果后上传 GitHub。

新增 Streamlit 的实际版本在安装后保存，不覆盖原核心依赖清单：

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements-demo-lock.txt
```

含 MVTec 原图的截图遵守数据许可并保留来源说明。
