import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Page Setup ----
st.set_page_config(page_title="TailorMailer.ai", layout="wide")
st.title("🏢 TailorMailer.ai")
st.markdown("<h4 style='color: gray; margin-top: -10px;'>Craft tailored emails for your Product in seconds! </h4>", unsafe_allow_html=True)

# ---- LLM Setup ----
llm = ChatGroq(
    temperature=0,
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="mistral-saba-24b"
)

# ---- Summarization Prompt ----
summarize_prompt = PromptTemplate.from_template(
    """
    ### WEBSITE CHUNK:
    {page}

    ### INSTRUCTION:
    Summarize this content in 2-3 sentences, focusing only on what the company does and offers.

    ### SUMMARY:
    """
)
summarizer = summarize_prompt | llm

# ---- Email Generation Prompt ----
email_prompt = PromptTemplate.from_template(
    """
    ### CLIENT COMPANY SUMMARY:
    {client_summary}

    ### OUR PRODUCT SUMMARY:
    {product_summary}

    ### INSTRUCTION:
    You are Praveen, a Business Development Executive at IBM.
    Write a personalized cold email to the client introducing IBM and explaining
    how it can help them based on the client company summary & our product summary.
    Be clear, professional, helpful, and end with a soft call to action.

    ### COLD EMAIL:
    """
)
email_chain = email_prompt | llm

# ---- Helper: Chunk + Summarize ----
def summarize_large_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer.invoke({"page": chunk}).content
            summaries.append(summary)
        except Exception as e:
            summaries.append("")  # skip bad chunk if needed
    return "\n".join(summaries)

# ---- UI Inputs ----
col1, col2 = st.columns(2)
with col1:
    client_url = st.text_input("🔗 Enter Client Website URL", placeholder="https://client.com")
with col2:
    product_url = st.text_input("🏢 Enter Your Product Website URL", placeholder="https://yourproduct.com")

# ---- Button Action ----
if st.button("Generate Personalized B2B Email"):
    if not client_url or not product_url:
        st.warning("Please provide both URLs.")
    else:
        with st.spinner("Scraping, summarizing, and generating email..."):
            try:
                # Step 1: Scrape both sites
                client_text = WebBaseLoader(client_url).load().pop().page_content
                product_text = WebBaseLoader(product_url).load().pop().page_content

                # Step 2: Chunk + summarize
                client_summary = summarize_large_text(client_text)
                product_summary = summarize_large_text(product_text)

                # Step 3: Generate email
                email = email_chain.invoke({
                    "client_summary": client_summary,
                    "product_summary": product_summary
                }).content

                # Step 4: Show result
                st.subheader("📬 Personalized Cold Email")
                st.code(email, language="markdown")

            except Exception as e:
                st.error(f"Error: {e}")
