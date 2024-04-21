from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import pinecone
import streamlit as st
import os
import tempfile
import time



embeddings = OpenAIEmbeddings()


def process_and_index_pdf(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        start_time = time.time()
        uploaded_file_name = uploaded_file.name
        st.write(f"Processing file: {uploaded_file_name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_file_path = tmp.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        text = text_splitter.split_documents(document)

        for doc in text:
            doc.metadata["title"] = uploaded_file_name

        metalist = []
        for doc in text:
            metadict = {
                "filename_source": doc.metadata["source"],
                "pagenum_doc": doc.metadata["page"],
                "total_pages": doc.metadata["total_pages"],
                "document_title": doc.metadata["title"],
            }
            metalist.append(metadict)

        doc_store = PineconeVectorStore.from_texts(
            [d.page_content for d in text],
            embeddings,
            metadatas=metalist,
            index_name="hq-broc",
        )
        num_lines = len([d.page_content for d in text])
        total_time_minutes = (time.time() - start_time) / 60
        results.append((uploaded_file_name, num_lines, total_time_minutes))

    return results


def main():
    st.set_page_config(page_title="Brochure Upload Data", page_icon=":moon:")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Brochure Upload: - Upload Data Module :moon:")
    st.caption("Upload Data.")

    uploaded_files = st.file_uploader(
        "Choose PDF files to be uploaded on the vector database",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files is not None:
        if st.button("Upload"):
            with st.spinner("Uploading the selected documents to Pinecone..."):
                processing_results = process_and_index_pdf(uploaded_files)
                for result in processing_results:
                    st.write(
                        f"Processed '{result[0]}': {result[1]} lines in {result[2]:.2f} minutes."
                    )

    st.write("\n\n\n")
    st.header("Optional")
    st.write(
        "If you want to load and just confirm if you already uploaded the document"
    )
    query = st.text_input(
        "Enter a query (you do not need to do anything here it just returns top 3 vector similarity from your query)"
    )
    if query:
        with st.spinner("Querying Pinecone..."):
            doc_store = PineconeVectorStore.from_texts(
                [], embeddings, index_name="hq-broc"
            )
            docs = doc_store.similarity_search(query, k=3)
        st.write("Query results:")
        st.write(docs)


if __name__ == "__main__":
    main()
