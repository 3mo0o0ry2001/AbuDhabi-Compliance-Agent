import streamlit as st
from agent import get_compliance_agent

# 1. إعدادات الصفحة (The Look & Feel)
st.set_page_config(
    page_title="Sovereign Compliance Agent",
    page_icon="🏗️",
    layout="centered"
)

# 2. تصميم الواجهة
st.title("🏗️ The Sovereign Compliance Architect")
st.markdown("""
This system acts as a sovereign AI agent to review engineering compliance based on the **Abu Dhabi International Building Code**.
""")
st.divider()

# 3. تحميل الـ Agent (نستخدم Cache عشان الموديل ميحملش من الصفر مع كل رسالة)
@st.cache_resource
def load_agent():
    return get_compliance_agent()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"حدث خطأ في تحميل قاعدة البيانات. تأكد من مسار الـ Chroma DB. \n {e}")
    st.stop()

# 4. إعداد ذاكرة المحادثة (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, engineer. I am the authorized regulatory agent for the Abu Dhabi Building Code. How may I assist you today?"}
    ]

# 5. عرض المحادثات السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. مربع إدخال المستخدم
if prompt := st.chat_input("Example: What is the maximum permitted height for external fences?"):
    # إضافة وعرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # تشغيل الـ Agent وعرض النتيجة
    with st.chat_message("assistant"):
        with st.spinner("🔍 جاري البحث في اللوائح والقوانين..."):
            try:
                # استدعاء الـ Agent
                response = agent.invoke(prompt)
                st.markdown(response)
                # حفظ الرد في الذاكرة
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"حدث خطأ أثناء الاتصال بالنموذج: {e}")