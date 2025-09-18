import streamlit as st
import os
import json
import base64
from io import BytesIO
import google.generativeai as genai
from PIL import Image
from pdf2image import convert_from_bytes
import tempfile

# --- Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="Trích xuất thông tin bảo hiểm",
    page_icon="📋",
    layout="wide"
)

# --- Tiêu đề ứng dụng ---
st.title("🏥 Hệ thống trích xuất thông tin đơn yêu cầu bảo hiểm")
st.markdown("---")


# --- Cấu hình API Key ---
@st.cache_resource
def configure_gemini_api():
    """Cấu hình API key cho Gemini"""
    try:
        # Kiểm tra API key từ Streamlit secrets hoặc biến môi trường
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            st.error("⚠️ Không tìm thấy GOOGLE_API_KEY. Vui lòng cấu hình trong Streamlit secrets.")
            st.info("Hướng dẫn: Thêm GOOGLE_API_KEY vào file .streamlit/secrets.toml")
            st.stop()

        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {str(e)}")
        st.stop()


# Cấu hình API
configure_gemini_api()


# --- Hàm trích xuất thông tin ---
@st.cache_data
def extract_information_from_images_gemini(_image_list):
    """Trích xuất thông tin từ hình ảnh sử dụng Gemini AI"""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt_text = """
        Bạn là một trợ lý AI chuyên nghiệp chuyên xử lý tài liệu bảo hiểm.
        Nhiệm vụ của bạn là trích xuất thông tin quan trọng từ các hình ảnh của một đơn yêu cầu bảo hiểm dưới đây.
        Hãy trả về kết quả dưới dạng một đối tượng JSON.
        Các trường thông tin cần trích xuất là:

        --- Thông tin người được bảo hiểm ---
        insured_name: Tên của người được bảo hiểm (phần 1).
        policyholder_name: Tên của người chủ hợp đồng.
        hkid_number: Số HKID của người được bảo hiểm.
        date_of_birth: Ngày sinh (định dạng DD/MM/YYYY).
        sex: Giới tính ('Male' hoặc 'Female').
        occupation: Nghề nghiệp.
        signature_date: Ngày ký tên của người được bảo hiểm (định dạng DD/MM/YYYY).
        policy_number: Số hợp đồng bảo hiểm hoặc chứng chỉ.
        product_type: Loại sản phẩm ('Individual' hoặc 'Group').
        claim_benefits: Các lợi ích yêu cầu bồi thường (dưới dạng mảng string, ví dụ ['Medical Reimbursement', 'Hospital Income']).
        other_insurance: Có nộp yêu cầu cho công ty bảo hiểm khác không (true hoặc false).
        other_insurance_company: Tên công ty bảo hiểm khác nếu có.
        other_policy_number: Số hợp đồng của công ty khác nếu có.
        doctor_name: Tên bác sĩ điều trị.
        treatment_date: Ngày điều trị (định dạng DD/MM/YYYY).
        accident_date: Ngày xảy ra tai nạn (định dạng DD/MM/YYYY, nếu áp dụng).
        accident_time: Thời gian xảy ra tai nạn (nếu áp dụng).
        accident_place: Nơi xảy ra tai nạn (nếu áp dụng).
        accident_description: Mô tả tai nạn (nếu áp dụng).

        --- Thông tin y tế ---
        patient_name: Tên của bệnh nhân (phần 2, do bác sĩ điền).
        admission_date: Ngày nhập viện (định dạng DD/MM/YYYY).
        discharge_date: Ngày xuất viện (định dạng DD/MM/YYYY).
        symptoms: Các triệu chứng chính được liệt kê.
        final_diagnosis: Chẩn đoán cuối cùng của bác sĩ.
        operation_performed: Tên các phẫu thuật/thủ thuật đã thực hiện.
        is_accident: Việc nhập viện có phải do tai nạn không (true hoặc false).
        hospital_name: Tên bệnh viện.
        first_consultation_date: Ngày tư vấn đầu tiên với bác sĩ cho bệnh này (định dạng DD/MM/YYYY).
        symptoms_duration: Thời gian triệu chứng tồn tại trước khi tư vấn đầu tiên.
        attending_physician_name: Tên bác sĩ điều trị.
        hospital_ward: Loại phòng bệnh viện ('Private', 'Semi-private', 'Ward', v.v.).
        icu_from: Ngày bắt đầu lưu trú ICU (định dạng DD/MM/YYYY, nếu áp dụng).
        icu_to: Ngày kết thúc lưu trú ICU (định dạng DD/MM/YYYY, nếu áp dụng).
        operation_date: Ngày thực hiện phẫu thuật (định dạng DD/MM/YYYY).
        mode_of_anesthesia: Chế độ gây mê (ví dụ: 'GA', 'LA', 'MAC', 'Sedation').
        professional_comment: Nhận xét chuyên môn của bác sĩ.
        is_pre_existing: Bệnh có tồn tại trước khi mua bảo hiểm không (true hoặc false).
        condition_causes: Các nguyên nhân liên quan đến bệnh (dưới dạng mảng string, ví dụ ['Pregnancy', 'Self-inflicted injury', 'Infertility']).

        --- Thông tin tài chính ---
        bank_name: Tên ngân hàng để nhận thanh toán.
        bank_account_number: Số tài khoản ngân hàng.
        account_holder_name: Tên chủ tài khoản ngân hàng.
        currency: Tiền tệ của tài khoản ('HKD', 'USD', v.v.).
        branch_number: Số chi nhánh ngân hàng.
        bank_code: Mã ngân hàng.
        fps_identifier: Số điện thoại, email hoặc ID FPS để nhận thanh toán.

        --- Thông tin khác ---
        reason_outside_hk: Lý do chi phí y tế ngoài Hong Kong/Macau (nếu áp dụng).
        policyowner_hkid: Số HKID của chủ hợp đồng.
        policyowner_signature_date: Ngày ký tên của chủ hợp đồng (định dạng DD/MM/YYYY).

        QUAN TRỌNG: Chỉ trả về nội dung của đối tượng JSON, không bao gồm các ký tự markdown như ```json
        """

    request_contents = [prompt_text, *_image_list]
    response = model.generate_content(request_contents)
    cleaned_json_text = (
        response.text.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    return json.loads(cleaned_json_text)


# --- Hàm xử lý file ---
def process_uploaded_file(uploaded_file):
    """Xử lý file upload và chuyển đổi thành danh sách hình ảnh"""
    file_bytes = uploaded_file.read()
    images = []

    if uploaded_file.type == "application/pdf":
        # Xử lý file PDF
        images = convert_from_bytes(file_bytes)
    else:
        # Xử lý file hình ảnh
        image = Image.open(BytesIO(file_bytes))
        images = [image]

    return images


# --- Giao diện chính ---
def main():
    # Sidebar cho cấu hình
    st.sidebar.header("⚙️ Cấu hình")
    st.sidebar.info("Hỗ trợ các định dạng: PDF, JPG, PNG, JPEG")

    # Upload file
    st.header("📄 Upload tài liệu bảo hiểm")
    uploaded_file = st.file_uploader(
        "Chọn file đơn yêu cầu bảo hiểm",
        type=['pdf', 'jpg', 'jpeg', 'png'],
        help="Hỗ trợ file PDF hoặc hình ảnh (JPG, PNG, JPEG)"
    )

    if uploaded_file is not None:
        # Hiển thị thông tin file
        col1, col2 = st.columns([1, 2])

        with col1:
            st.info(f"**Tên file:** {uploaded_file.name}")
            st.info(f"**Kích thước:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Loại file:** {uploaded_file.type}")

        with col2:
            # Preview hình ảnh nếu không phải PDF
            if uploaded_file.type != "application/pdf":
                image = Image.open(uploaded_file)
                st.image(image, caption="Preview", use_column_width=True)

        # Nút xử lý
        if st.button("🚀 Trích xuất thông tin", type="primary"):
            with st.spinner("Đang xử lý tài liệu..."):
                try:
                    # Xử lý file
                    images = process_uploaded_file(uploaded_file)

                    # Trích xuất thông tin
                    extracted_data = extract_information_from_images_gemini(images)

                    # Hiển thị kết quả
                    st.success("✅ Trích xuất thông tin thành công!")

                    # Tạo tabs cho các loại thông tin khác nhau
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "👤 Thông tin cá nhân",
                        "🏥 Thông tin y tế",
                        "💰 Thông tin tài chính",
                        "📋 Thông tin khác",
                        "📊 JSON Raw"
                    ])

                    with tab1:
                        st.subheader("Thông tin người được bảo hiểm")
                        personal_fields = [
                            'insured_name', 'policyholder_name', 'hkid_number',
                            'date_of_birth', 'sex', 'occupation', 'signature_date'
                        ]
                        for field in personal_fields:
                            if field in extracted_data and extracted_data[field]:
                                st.text(f"{field.replace('_', ' ').title()}: {extracted_data[field]}")

                    with tab2:
                        st.subheader("Thông tin y tế")
                        medical_fields = [
                            'patient_name', 'admission_date', 'discharge_date', 'symptoms',
                            'final_diagnosis', 'operation_performed', 'hospital_name',
                            'attending_physician_name', 'operation_date'
                        ]
                        for field in medical_fields:
                            if field in extracted_data and extracted_data[field]:
                                st.text(f"{field.replace('_', ' ').title()}: {extracted_data[field]}")

                    with tab3:
                        st.subheader("Thông tin tài chính")
                        financial_fields = [
                            'bank_name', 'bank_account_number', 'account_holder_name',
                            'currency', 'branch_number', 'bank_code', 'fps_identifier'
                        ]
                        for field in financial_fields:
                            if field in extracted_data and extracted_data[field]:
                                st.text(f"{field.replace('_', ' ').title()}: {extracted_data[field]}")

                    with tab4:
                        st.subheader("Thông tin bổ sung")
                        other_fields = [
                            'policy_number', 'product_type', 'claim_benefits',
                            'other_insurance', 'doctor_name', 'treatment_date'
                        ]
                        for field in other_fields:
                            if field in extracted_data and extracted_data[field]:
                                st.text(f"{field.replace('_', ' ').title()}: {extracted_data[field]}")

                    with tab5:
                        st.subheader("Dữ liệu JSON đầy đủ")
                        st.json(extracted_data)

                        # Nút download JSON
                        json_string = json.dumps(extracted_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="💾 Tải xuống JSON",
                            data=json_string,
                            file_name=f"extracted_data_{uploaded_file.name}.json",
                            mime="application/json"
                        )

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý: {str(e)}")
                    st.exception(e)


# --- Footer ---
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>🤖 Được hỗ trợ bởi Google Gemini AI | Made with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
    show_footer()