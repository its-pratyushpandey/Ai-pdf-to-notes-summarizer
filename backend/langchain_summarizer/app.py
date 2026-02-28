#app.py            
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# ✅ Streamlit app UI
st.set_page_config(page_title="LangChain Summarizer", page_icon=":guardsman:", layout="wide")
st.title("LangChain Summarizer")
st.markdown("This app uses LangChain to summarize text from various sources including YouTube videos and web pages.")

# ✅ Sidebar for Groq API Key
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", value="", type="password")

# ✅ Input field for URL
generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter a YouTube video URL or a web page URL")

# Set up LLM using Groq
if groq_api_key:
    
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)


    # ✅ Prompt template
    prompt_template = """
    Provide a summary of the following content in 500 words:
    Content: {text}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the content from YT and website"):
        # ✅ Input validations
        if not generic_url.strip():
            st.error("Please provide a URL.")
        elif not validators.url(generic_url):
            st.error("Invalid URL.")
        else:
            try:
                with st.spinner("Loading..."):
                    # ✅ Load content from YouTube or Web
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                        )

                    documents = loader.load()

                    # ✅ Summarization Chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output = chain.invoke(documents)

                    st.success("Summary generated successfully!")
                    st.write(output)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check the URL or your Groq API key.")
else:
    st.warning("Please enter your Groq API key to begin.")




