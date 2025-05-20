import streamlit as st

# Set page title
st.set_page_config(page_title="Streamlit Test App", page_icon="🚀")

# Main app
def main():
    st.title("مرحبًا بك في تطبيق اختبار Streamlit! 🎉")
    st.markdown("ادخل اسمك واضغط على الزر لتحية بسيطة.")

    # Text input
    name = st.text_input("ما اسمك؟", placeholder="اكتب اسمك هنا...")

    # Button
    if st.button("قل مرحبًا"):
        if name.strip() == "":
            st.warning("من فضلك، اكتب اسمك أولاً!")
        else:
            st.success(f"مرحبًا، {name}! شكرًا لتجربة التطبيق! 😊")

if __name__ == "__main__":
    main()