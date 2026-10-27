"""Local bottle anomaly detection demo. Launch with streamlit run app.py."""
import io
import json
from pathlib import Path
import sys
import threading

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from core import AnomalyDetector, render_panels, render_comparison

st.set_page_config(page_title='Bottle Defect Inspector', page_icon='🔍', layout='wide')
st.title('瓶口外观异常检测')
st.caption('正常局部特征库 · ResNet18 · 最近邻距离')
st.info('适用于与 MVTec AD bottle 类似的瓶口俯视图。分数表示外观偏离程度，不是缺陷概率。')


@st.cache_resource
def load_detector(bank_version, threshold_version):
    return AnomalyDetector(), threading.Lock()


uploaded = st.file_uploader('上传待检查图片', type=['png', 'jpg', 'jpeg', 'bmp'])
if uploaded is None:
    st.caption('选择一张图片，再点击“开始检测”。图片在本机服务内处理。')
    st.stop()

try:
    with Image.open(uploaded) as source:
        if source.width * source.height > 20_000_000:
            st.error('图片过大，请缩小到 2000 万像素以内。')
            st.stop()
        image = source.convert('RGB')
except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
    st.error('无法读取该图片，请使用有效的 PNG/JPEG/BMP 文件。')
    st.stop()

st.image(image, caption=uploaded.name, width=320)
if st.button('开始检测', type='primary'):
    try:
        artifacts = ROOT / 'outputs' / 'baseline_v1'
        with st.spinner('正在提取特征并搜索正常参照…'):
            detector, lock = load_detector(
                (artifacts / 'memory_bank.pt').stat().st_mtime_ns,
                (artifacts / 'threshold.json').stat().st_mtime_ns,
            )
            with lock:
                prediction = detector.predict(image)
        report = prediction.report()
        a, b, c = st.columns(3)
        a.metric('判断结果', '疑似异常' if report['prediction'] == 'anomaly' else '正常')
        b.metric('异常分数', f'{prediction.score:.6f}')
        c.metric('固定阈值', f'{prediction.threshold:.6f}')
        for column, panel, caption in zip(st.columns(3), render_panels(image, prediction),
                ['原图', '距离热力图：蓝=0，红=2', '叠加图：黄圈为最大距离网格中心']):
            column.image(panel, caption=caption, use_container_width=True)
        st.caption('热力图来自 14×14 特征网格插值，用于粗略定位；没有验证像素级分割精度。')
        buffer = io.BytesIO()
        render_comparison(image, prediction).save(buffer, format='PNG')
        st.download_button('下载结果图', buffer.getvalue(), 'prediction.png', 'image/png')
        st.download_button('下载预测 JSON', json.dumps(report, indent=2),
                           'prediction.json', 'application/json')
    except FileNotFoundError:
        st.error('缺少特征库或阈值文件，请先按 README 完成建库和阈值校准。')
    except (RuntimeError, ValueError, OSError) as error:
        st.error(f'检测失败：{error}')
