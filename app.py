import streamlit as st
from PIL import Image

from config import APP_CONFIG
from inference import ClickbaitPredictor


st.set_page_config(
    page_title="Vietnamese Clickbait Detector",
    page_icon="📰",
    layout="centered",
)


def read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_predictor() -> ClickbaitPredictor:
    return ClickbaitPredictor(
        hf_token=read_secret("HF_TOKEN")
    )


st.title("Vietnamese Clickbait Detector")
st.caption(
    "PhoBERT + ResNet50 + metadata-guided sparse attention"
)

if APP_CONFIG.hf_repo_id.startswith("YOUR_HF_USERNAME/"):
    st.error(
        "Bạn chưa sửa HF_REPO_ID trong config.py."
    )
    st.stop()

try:
    predictor = load_predictor()
except Exception as error:
    st.error("Không thể tải mô hình từ Hugging Face.")
    st.exception(error)
    st.stop()

with st.form("prediction_form"):
    title = st.text_input(
        "Tiêu đề bài báo *",
        placeholder="Nhập tiêu đề cần kiểm tra...",
    )

    lead = st.text_area(
        "Đoạn mở đầu",
        placeholder="Nhập lead paragraph...",
        height=140,
    )

    uploaded_image = st.file_uploader(
        "Ảnh thumbnail",
        type=["png", "jpg", "jpeg", "webp"],
    )

    source_options = ["Không rõ / UNK"] + predictor.get_sources()
    category_options = (
        ["Không rõ / UNK"] + predictor.get_categories()
    )

    source = st.selectbox("Nguồn báo", source_options)
    category = st.selectbox(
        "Chuyên mục",
        category_options,
    )

    submitted = st.form_submit_button(
        "Phân tích",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not title.strip():
        st.warning("Bạn cần nhập tiêu đề.")
        st.stop()

    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

    normalized_source = (
        None if source == "Không rõ / UNK" else source
    )
    normalized_category = (
        None
        if category == "Không rõ / UNK"
        else category
    )

    try:
        with st.spinner("Đang dự đoán..."):
            result = predictor.predict(
                title=title,
                lead=lead,
                image=image,
                source=normalized_source,
                category=normalized_category,
            )
    except Exception as error:
        st.error("Dự đoán thất bại.")
        st.exception(error)
        st.stop()

    probability = result["clickbait_probability"]

    st.subheader("Kết quả")

    if result["label_id"] == 1:
        st.error("Có dấu hiệu clickbait")
    else:
        st.success("Không có dấu hiệu clickbait")

    col1, col2 = st.columns(2)
    col1.metric(
        "Xác suất clickbait",
        f"{probability:.2%}",
    )
    col2.metric(
        "Xác suất non-clickbait",
        f"{result['non_clickbait_probability']:.2%}",
    )

    st.progress(probability)

    if image is not None:
        st.image(
            image,
            caption="Ảnh thumbnail được phân tích",
            use_container_width=True,
        )

    with st.expander("Thông tin kỹ thuật"):
        st.json(result)
