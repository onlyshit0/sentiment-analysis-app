import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(
    page_title="",
    page_icon="",
    layout="wide"
)

st.title("情感分析")
st.markdown("输入一段英文文本， AI会判断是正面还是负面情绪")

@st.cache_resource
def load_model():
    model_path = './distilbert-imdb-model'
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
st.sidebar.success("模型加载")

user_input = st.text_area("请输入英文文本",height=150)

if st.button("分析情感"):
    if user_input.strip():
        with st.spinner("分析中..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
            confidence = predictions[0][predicted_class].item()

        if predicted_class == 1:
            st.success(f"({confidence*100:.2f}%)")
        else:
            st.error(f"({confidence*100:.2f}%)")

        st.write("**所有类别概率**")
        st.write(f"- 负面情感： {predictions[0][0].item()*100:.2f}%")
        st.write(f"- 正面情感: {predictions[0][1].item()*100:.2f}%")
    else:
        st.warning("请输入文本内容")

st.markdown("---")
st.caption("模型：DistilBERT 微调于 IMDB 数据集 | 支持英文文本情感分析")