
import streamlit as st
import base64
from ingest import *
from retriever import *
from generator import *

# Initialize the API key in session state if it doesn't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

st.session_state.vector_store = None
def display_pdf(uploaded_file):

    """
    Display a PDF file that has been uploaded to Streamlit.

    The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded PDF file to display.

    Returns
    -------
    None
    """
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_streamlit_page():

    """
    Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. 
    The right column contains a header and text that greet the user and explain the purpose of the tool.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_file: The uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Input your OpenAI API key")
        st.text_input('OpenAI API key', type='password', key='api_key',
                    label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

    return col1, col2, uploaded_file


# Make a streamlit page
col1, col2, uploaded_file = load_streamlit_page()

# Process the input
if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)
    embedder = get_embedder(st.session_state.api_key)
    # Load in the documents
    loaded_vectorstore = load_vectorstore(uploaded_file.name, embedder)
    
    
    if loaded_vectorstore:
        st.session_state.vector_store = loaded_vectorstore
        st.write("Loaded existing store")
    else:
        documents = load_documents(uploaded_file)
        chunks = chunk_text(documents)
        
    
        st.session_state.vector_store = create_vectorstore(file_name=uploaded_file.name,
                                                           texts=chunks,
                                                           embedding_function=embedder)
        st.write("Input Processed")

# Generate answer
with col1:
    query = st.text_area("Enter your question or topic here:")
    button_container = st.container()
    placeholder = st.empty()

    if st.session_state.vector_store:
        context = retrieve(st.session_state.vector_store, query)

        ask_clicked = button_container.button("Ask me!")
        quiz_clicked = button_container.button("Get Quiz")

        if ask_clicked:
            placeholder.empty()
            with st.spinner("Generating answer"):
                # Load vectorstore:
                
                res = generate_question_answering(context, 
                                                    question=query, 
                                                    api_key=st.session_state.api_key)
                
                answer_box = st.container()
                with placeholder:
                    with answer_box:
                        st.subheader("Answer:")
                        st.text(res.answer)
                        st.subheader("Sources:")
                        st.caption(res.sources)

        if quiz_clicked:
            placeholder.empty()
            with st.spinner("Generating quiz"):
                # Load vectorstore:
                res = generate_flashcard(context, 
                                        num_cards=5, 
                                        api_key=st.session_state.api_key)
                st.write("Enjoy your quiz!")
                st.balloons()
                answer_box = st.container()
                with placeholder:
                    with answer_box:
                        for index, card in enumerate(res.cards):
                            st.subheader(f"Question {index + 1}:", divider="blue")
                            st.text(card.question)
                            st.subheader("Answer:")
                            st.text(card.keypoint)
                            st.caption("Sources:")
                            st.caption(card.sources)
        