import streamlit as st
from PIL import Image
from pathlib import Path

from config import APP_CONFIG
from inference import ClickbaitPredictor
import textwrap

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vietnamese Clickbait Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PATH CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "logo.png"
HERO_IMAGE_PATH = ASSETS_DIR / "clickbait_banner.png"
DECORATION_IMAGE_PATH = ASSETS_DIR / "news_analysis.png"


# =========================================================
# CUSTOM CSS
# =========================================================
def load_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* =========================
           GLOBAL
        ========================= */
        .stApp {
            background:
                radial-gradient(
                    circle at top left,
                    rgba(99, 102, 241, 0.16),
                    transparent 32%
                ),
                radial-gradient(
                    circle at top right,
                    rgba(14, 165, 233, 0.14),
                    transparent 28%
                ),
                linear-gradient(
                    135deg,
                    #f8fafc 0%,
                    #eef2ff 48%,
                    #f0f9ff 100%
                );
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            font-family: Inter, Arial, sans-serif;
        }

        p, label, div {
            font-family: Inter, Arial, sans-serif;
        }

        /* =========================
           HIDE STREAMLIT ELEMENTS
        ========================= */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            background: transparent !important;
        }

        /* =========================
           HERO
        ========================= */
        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 2.7rem 3rem;
            margin-bottom: 2rem;
            border-radius: 28px;
            color: white;
            background:
                linear-gradient(
                    135deg,
                    rgba(30, 41, 59, 0.98),
                    rgba(67, 56, 202, 0.95) 55%,
                    rgba(14, 165, 233, 0.90)
                );
            box-shadow:
                0 25px 60px rgba(30, 41, 59, 0.20);
        }

        .hero-card::before {
            content: "";
            position: absolute;
            width: 300px;
            height: 300px;
            top: -170px;
            right: -70px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.12);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            bottom: -110px;
            left: 40%;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.9rem;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(10px);
        }

        .hero-title {
            position: relative;
            z-index: 2;
            max-width: 760px;
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 3.9rem);
            font-weight: 850;
            line-height: 1.08;
            letter-spacing: -0.04em;
        }

        .hero-description {
            position: relative;
            z-index: 2;
            max-width: 760px;
            margin-top: 1rem;
            margin-bottom: 0;
            color: rgba(255, 255, 255, 0.84);
            font-size: 1.05rem;
            line-height: 1.7;
        }

        .hero-tags {
            position: relative;
            z-index: 2;
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.5rem;
        }

        .hero-tag {
            padding: 0.42rem 0.8rem;
            border-radius: 10px;
            color: #ffffff;
            font-size: 0.79rem;
            font-weight: 650;
            background: rgba(255, 255, 255, 0.12);
        }

        /* =========================
           SECTION HEADERS
        ========================= */
        .section-label {
            margin-bottom: 0.35rem;
            color: #4f46e5;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .section-title {
            margin-top: 0;
            margin-bottom: 0.45rem;
            color: #0f172a;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        .section-description {
            margin-bottom: 1.4rem;
            color: #64748b;
            line-height: 1.65;
        }

        /* =========================
           INPUT CARD
        ========================= */
        div[data-testid="stForm"] {
            padding: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.90);
            box-shadow:
                0 18px 45px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(12px);
        }

        /* =========================
           INPUTS
        ========================= */
        .stTextInput input,
        .stTextArea textarea {
            border: 1px solid #dbeafe !important;
            border-radius: 13px !important;
            background: #f8fafc !important;
            transition: all 0.2s ease;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow:
                0 0 0 3px rgba(99, 102, 241, 0.13) !important;
        }

        div[data-baseweb="select"] > div {
            border-color: #dbeafe !important;
            border-radius: 13px !important;
            background: #f8fafc !important;
        }

        section[data-testid="stFileUploaderDropzone"] {
            min-height: 150px;
            border: 1.5px dashed #a5b4fc;
            border-radius: 16px;
            background:
                linear-gradient(
                    135deg,
                    rgba(238, 242, 255, 0.9),
                    rgba(240, 249, 255, 0.9)
                );
        }

        /* =========================
           BUTTON
        ========================= */
        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 52px;
            border: none !important;
            border-radius: 14px !important;
            color: white !important;
            font-size: 1rem;
            font-weight: 750;
            background:
                linear-gradient(
                    135deg,
                    #4f46e5,
                    #7c3aed 55%,
                    #0ea5e9
                ) !important;
            box-shadow:
                0 12px 25px rgba(79, 70, 229, 0.24);
            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            box-shadow:
                0 16px 35px rgba(79, 70, 229, 0.33);
        }

        /* =========================
           INFORMATION CARD
        ========================= */
        .info-card {
            padding: 1.6rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow:
                0 14px 35px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(10px);
        }

        .info-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            margin-bottom: 1rem;
            border-radius: 14px;
            font-size: 1.45rem;
            background:
                linear-gradient(
                    135deg,
                    #e0e7ff,
                    #dbeafe
                );
        }

        .info-title {
            margin-bottom: 0.45rem;
            color: #0f172a;
            font-size: 1rem;
            font-weight: 800;
        }

        .info-description {
            margin: 0;
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* =========================
           RESULT
        ========================= */
        .result-wrapper {
            padding: 2rem;
            margin-top: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 26px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow:
                0 22px 55px rgba(15, 23, 42, 0.09);
        }

        .result-clickbait {
            padding: 1.3rem 1.5rem;
            margin-bottom: 1.2rem;
            border: 1px solid #fecaca;
            border-radius: 18px;
            background:
                linear-gradient(
                    135deg,
                    #fff1f2,
                    #fef2f2
                );
        }

        .result-safe {
            padding: 1.3rem 1.5rem;
            margin-bottom: 1.2rem;
            border: 1px solid #bbf7d0;
            border-radius: 18px;
            background:
                linear-gradient(
                    135deg,
                    #f0fdf4,
                    #ecfdf5
                );
        }

        .result-status-title {
            margin: 0 0 0.3rem 0;
            color: #0f172a;
            font-size: 1.25rem;
            font-weight: 850;
        }

        .result-status-description {
            margin: 0;
            color: #475569;
            line-height: 1.6;
        }

        /* =========================
           METRICS
        ========================= */
        div[data-testid="stMetric"] {
            padding: 1.25rem;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            background:
                linear-gradient(
                    135deg,
                    #ffffff,
                    #f8fafc
                );
            box-shadow:
                0 10px 25px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 850;
        }

        /* =========================
           PROGRESS
        ========================= */
        div[data-testid="stProgress"] > div > div {
            background:
                linear-gradient(
                    90deg,
                    #4f46e5,
                    #7c3aed,
                    #0ea5e9
                );
        }

        /* =========================
           FOOTER
        ========================= */
        .custom-footer {
            padding-top: 3rem;
            color: #94a3b8;
            font-size: 0.82rem;
            text-align: center;
        }

        /* =========================
           MOBILE
        ========================= */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-card {
                padding: 2rem 1.4rem;
                border-radius: 20px;
            }

            .hero-title {
                font-size: 2.2rem;
            }

            div[data-testid="stForm"] {
                padding: 1.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_custom_css()


# =========================================================
# HELPERS
# =========================================================
def read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


@st.cache_resource(show_spinner="Đang tải mô hình phân tích...")
def load_predictor() -> ClickbaitPredictor:
    return ClickbaitPredictor(
        hf_token=read_secret("HF_TOKEN")
    )



def render_hero() -> None:
    st.markdown(
        """
<div class="hero-card">
<div class="hero-badge">✦ AI-POWERED NEWS ANALYSIS</div>

<h1 class="hero-title">Vietnamese Clickbait Detector</h1>

<p class="hero-description">
Phân tích tiêu đề, nội dung mở đầu, ảnh thumbnail và metadata
của bài báo để nhận diện dấu hiệu clickbait trong tin tức tiếng Việt.
</p>

<div class="hero-tags">
<span class="hero-tag">PhoBERT</span>
<span class="hero-tag">ResNet50</span>
<span class="hero-tag">Multimodal Learning</span>
<span class="hero-tag">Sparse Attention</span>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_information_cards() -> None:
    cards = [
        (
            "Phân tích văn bản",
            "Mô hình đánh giá tiêu đề và đoạn mở đầu "
            "để phát hiện ngôn ngữ gây tò mò hoặc phóng đại.",
        ),
        (
            "Phân tích thumbnail",
            "ResNet50 trích xuất đặc trưng hình ảnh và "
            "kết hợp với thông tin văn bản.",
        ),
        (
            "Kết hợp đa phương thức",
            "Nguồn báo, chuyên mục, văn bản và hình ảnh "
            "được tổng hợp để đưa ra dự đoán.",
        ),
    ]

    for title, description in cards:
        st.markdown(
            f"""
<div class="info-card">
<div class="info-title">{title}</div>
<p class="info-description">{description}</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_result_status(
    label_id: int,
    probability: float,
) -> None:
    if label_id == 1:
        status_class = "result-clickbait"
        icon = "⚠️"
        title = "Phát hiện dấu hiệu clickbait"
        description = (
            f"Mô hình đánh giá bài báo có "
            f"{probability:.2%} khả năng chứa nội dung clickbait. "
            "Bạn nên kiểm tra mức độ tương đồng giữa tiêu đề, "
            "ảnh thumbnail và nội dung thực tế của bài báo."
        )
    else:
        status_class = "result-safe"
        icon = "✅"
        title = "Không phát hiện dấu hiệu clickbait rõ ràng"
        description = (
            f"Mô hình đánh giá bài báo có "
            f"{1 - probability:.2%} khả năng là non-clickbait. "
            "Tiêu đề và các thành phần của bài báo có xu hướng "
            "phù hợp với nội dung được cung cấp."
        )

    st.markdown(
        f"""
        <div class="{status_class}">
            <h3 class="result-status-title">
                {icon} {title}
            </h3>
            <p class="result-status-description">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HERO SECTION
# =========================================================
render_hero()


# =========================================================
# MODEL CONFIG VALIDATION
# =========================================================
if APP_CONFIG.hf_repo_id.startswith("YOUR_HF_USERNAME/"):
    st.error(
        "Bạn chưa cấu hình `HF_REPO_ID` trong file `config.py`."
    )
    st.stop()


try:
    predictor = load_predictor()
except Exception as error:
    st.error(
        "Không thể tải mô hình từ Hugging Face. "
        "Vui lòng kiểm tra kết nối, repository và HF_TOKEN."
    )
    st.exception(error)
    st.stop()


# =========================================================
# MAIN CONTENT
# =========================================================
input_column, information_column = st.columns(
    [1.65, 0.75],
    gap="large",
)


with input_column:
    st.markdown(
        """
        <div class="section-label">News Analyzer</div>
        <h2 class="section-title">Nhập thông tin bài báo</h2>
        <p class="section-description">
            Cung cấp càng đầy đủ thông tin, mô hình càng có nhiều
            dữ liệu để đánh giá chính xác.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        title = st.text_input(
            "Tiêu đề bài báo *",
            placeholder=(
                "Ví dụ: Bạn sẽ không tin điều gì xảy ra "
                "sau khi người đàn ông mở cánh cửa..."
            ),
        )

        lead = st.text_area(
            "Đoạn mở đầu",
            placeholder=(
                "Nhập đoạn giới thiệu hoặc đoạn mở đầu "
                "của bài báo..."
            ),
            height=160,
        )

        uploaded_image = st.file_uploader(
            "Ảnh thumbnail",
            type=["png", "jpg", "jpeg", "webp"],
            help=(
                "Hỗ trợ định dạng PNG, JPG, JPEG và WEBP."
            ),
        )

        metadata_col1, metadata_col2 = st.columns(2)

        source_options = (
            ["Không rõ / UNK"] + predictor.get_sources()
        )
        category_options = (
            ["Không rõ / UNK"] + predictor.get_categories()
        )

        with metadata_col1:
            source = st.selectbox(
                "Nguồn báo",
                source_options,
            )

        with metadata_col2:
            category = st.selectbox(
                "Chuyên mục",
                category_options,
            )

        st.markdown("<div style='height: 8px'></div>",
                    unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🔍 Phân tích bài báo",
            type="primary",
            use_container_width=True,
        )


with information_column:
    st.markdown(
        """
        <div class="section-label">Model Overview</div>
        <h2 class="section-title">Cách hệ thống hoạt động</h2>
        <p class="section-description">
            Hệ thống kết hợp nhiều loại dữ liệu thay vì chỉ
            phân tích riêng tiêu đề.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if DECORATION_IMAGE_PATH.exists():
        st.image(
            str(DECORATION_IMAGE_PATH),
            use_container_width=True,
        )

    render_information_cards()

    st.info(
        "Kết quả dự đoán chỉ mang tính hỗ trợ. "
        "Bạn vẫn nên kiểm tra nội dung bài báo trước khi "
        "đưa ra kết luận cuối cùng."
    )


# =========================================================
# PREDICTION
# =========================================================
if submitted:
    if not title.strip():
        st.warning("Bạn cần nhập tiêu đề bài báo.")
        st.stop()

    image = None

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
        except Exception:
            st.error(
                "Không thể đọc ảnh thumbnail. "
                "Vui lòng chọn một file ảnh hợp lệ."
            )
            st.stop()

    normalized_source = (
        None
        if source == "Không rõ / UNK"
        else source
    )

    normalized_category = (
        None
        if category == "Không rõ / UNK"
        else category
    )

    try:
        with st.spinner(
            "Mô hình đang phân tích văn bản, hình ảnh "
            "và metadata..."
        ):
            result = predictor.predict(
                title=title,
                lead=lead,
                image=image,
                source=normalized_source,
                category=normalized_category,
            )

    except Exception as error:
        st.error(
            "Quá trình dự đoán thất bại. "
            "Vui lòng kiểm tra dữ liệu đầu vào và mô hình."
        )
        st.exception(error)
        st.stop()

    probability = float(
        result["clickbait_probability"]
    )

    non_clickbait_probability = float(
        result["non_clickbait_probability"]
    )

    st.markdown(
        """
        <div class="result-wrapper">
            <div class="section-label">Analysis Result</div>
            <h2 class="section-title">Kết quả phân tích</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_result_status(
        label_id=result["label_id"],
        probability=probability,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Xác suất clickbait",
        f"{probability:.2%}",
    )

    metric_col2.metric(
        "Xác suất non-clickbait",
        f"{non_clickbait_probability:.2%}",
    )

    confidence = max(
        probability,
        non_clickbait_probability,
    )

    metric_col3.metric(
        "Độ tin cậy",
        f"{confidence:.2%}",
    )

    st.markdown("#### Mức độ clickbait")
    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    if probability >= 0.8:
        st.error(
            "Mức cảnh báo: Cao — bài báo có nhiều đặc điểm "
            "tương đồng với nội dung clickbait."
        )
    elif probability >= 0.5:
        st.warning(
            "Mức cảnh báo: Trung bình — nên kiểm tra thêm "
            "tiêu đề và nội dung bài báo."
        )
    else:
        st.success(
            "Mức cảnh báo: Thấp — chưa phát hiện dấu hiệu "
            "clickbait rõ ràng."
        )

    result_left, result_right = st.columns(
        [1.1, 0.9],
        gap="large",
    )

    with result_left:
        st.markdown("#### Nội dung đã phân tích")

        st.markdown("**Tiêu đề**")
        st.write(title)

        if lead.strip():
            st.markdown("**Đoạn mở đầu**")
            st.write(lead)

        st.markdown("**Metadata**")
        st.write(
            {
                "Nguồn báo": normalized_source or "Không xác định",
                "Chuyên mục": (
                    normalized_category or "Không xác định"
                ),
            }
        )

    with result_right:
        if image is not None:
            st.markdown("#### Thumbnail")
            st.image(
                image,
                caption="Ảnh thumbnail được mô hình phân tích",
                use_container_width=True,
            )
        else:
            st.markdown("#### Thumbnail")
            st.info(
                "Không có ảnh thumbnail được cung cấp."
            )

    with st.expander(
        "⚙️ Xem thông tin kỹ thuật của mô hình"
    ):
        st.json(result)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="custom-footer">
        Vietnamese Clickbait Detector ·
        PhoBERT + ResNet50 + Metadata-guided Sparse Attention
        <br>
        Research and educational demonstration system
    </div>
    """,
    unsafe_allow_html=True,
)
