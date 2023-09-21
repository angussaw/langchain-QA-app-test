"""Python file to serve as the frontend"""

import json
import os
import pandas as pd
import src.utils as backend
from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import streamlit_chat
# import wandb
import yaml

# Import config vars
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

config = config["streamlit"]

def validate_load_PDF(generated_dbs: list, new_collection_name: str, split_params: dict) -> list:
    """Function to ensure that PDF file can be loaded and split successfully

    Args:
        generated_dbs (list): List containing vector databases that already exist locally
        new_collection_name (str): String containing the unique identifier of the new Chroma vector database
        split_params (dict): Configuration parameters for splitting the PDF file into separate chunks

    Returns:
        list: List of error messages that flag out any issues with the settings to load and split PDF file
    """
    messages = []

    existing_collection_names = [db["collection name"] for db in generated_dbs]

    if new_collection_name in existing_collection_names:
        messages.append("Collection name already exists")

    if split_params["chunk_size"] <= split_params["chunk_overlap"]:
        messages.append("Chunk size must be greater than chunk overlap")

    return messages

@st.cache_data
def convert_chat_history_to_csv(past: list, generated: list) -> str:
    """Function to cache the conversation to prevent computation on every rerun

    Args:
        past (list): List containing all past queries from the user
        generated (list): List containing all past responses from language model

    Returns:
        str: Resulting csv format as a string
    """
    chat_history_df = pd.DataFrame()
    chat_history_df["User"] = past
    chat_history_df["Response"] = generated

    return chat_history_df.to_csv().encode('utf-8')


def main():
    """
    Main function to run PDF question-answering conversation app:
        - Uploading a new PDF file to split it into smaller chunks of text
        - Converting smaller chunks of text to vector embeddings and storing them in local vector databases 
        - Managing of PDF files that are loaded and ready to be queried
        - Setting up of conversation agent by specifying genative configuration
        - Initializing run on a Weights & Biases project to log agent execution chains and responses for refrence and evaluation
        - Chat with multiple PDF files at once via an agent fetching the correct documents based on question
        - Saving chat history as csv files
    """

    hide_default_format = """
        <style>
        .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

    tooltips = config["TOOLTIPS"]

    # From here down is all the StreamLit UI.
    st.set_page_config(page_title="Document conversation app",
                       layout="wide")
    st.markdown(hide_default_format, unsafe_allow_html=True)

    st.header("Converse with your PDFs!")
    st.caption("1. Upload a new PDF file with a specified article name and its description")
    st.caption("2. Select document(s) to converse with and enter OpenAI API key along with response parameters to generate conversation agent")
    # st.caption("3. Enter Weights & Biases project name and API key to log agent execution chains and responses for LLM evaluation")
    st.caption("3. Converse with your document(s) by entering your question!")
    st.divider()

    ##########################################
    ###### Setting up session variables ######
    ##########################################

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'agent' not in st.session_state:
        st.session_state['agent'] = None
    if 'docs_selected' not in st.session_state:
        st.session_state['docs_selected'] = []
    # if 'run' not in st.session_state:
    #     st.session_state['run'] = None

    existing_chroma_paths = [f.path for f in os.scandir("chroma_db") if f.is_dir()]
    if len(existing_chroma_paths) == 0:
        st.session_state["generated_dbs"] = []

    else:
        if "generated_dbs" not in st.session_state:
            st.session_state["generated_dbs"] = []
            for path in existing_chroma_paths:
                with open(f"{path}/metadata.json") as metadata_json:
                    metadata = json.load(metadata_json)
                st.session_state["generated_dbs"].append(metadata)

    ###########################################################
    ###### Populating AgGrid showcasing loaded PDF files ######
    ###########################################################

    generated_db_df = pd.DataFrame(st.session_state["generated_dbs"])
    gb = GridOptionsBuilder.from_dataframe(generated_db_df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb_grid_options = gb.build()
    st.caption("Select loaded PDF file(s) to converse with:")
    response = AgGrid(generated_db_df.head(5), gridOptions = gb_grid_options, use_checkbox=True)

    #########################################################################
    ###### Sidebar to load/split PDF files and store vector embeddings ######
    #########################################################################

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')
        collection_name = st.text_input("Provide article's name", help = tooltips["collection_name"])
        description = st.text_input("Provide article's description", help = tooltips["description"])
        chunk_size = st.slider('Select chunk size', 100, 1000, 500, help = tooltips["chunk_size"])
        chunk_overlap = st.slider('Select chunk overlap', 0, 500, 250, help = tooltips["chunk_overlap"])

        with st.form('myform', clear_on_submit=True):
            submitted_pdf = st.form_submit_button('Load PDF', disabled=not(uploaded_file))

    if submitted_pdf:
        generated_dbs = st.session_state["generated_dbs"]
        split_params = {"chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap}
        
        messages = validate_load_PDF(generated_dbs = generated_dbs,
                                     new_collection_name = collection_name,
                                     split_params = split_params)

        if len(messages) > 0:
            for message in messages:
                st.write(message)

        else:
            with st.sidebar:
                with st.spinner('Loading PDF and creating vector store...'):

                    metadata = backend.load_PDF(uploaded_file = uploaded_file,
                                                collection_name = collection_name,
                                                description = description,
                                                split_params = split_params)

                    with open(f"chroma_db/{collection_name}/metadata.json", "w") as fp:
                        json.dump(metadata , fp) 
                    
                    st.session_state.generated_dbs.append(metadata)
                    st.write("PDF was loaded successfully")
                    st.experimental_rerun()

    #################################################
    ###### Selected PDF files to converse with ######
    #################################################

    if st.session_state["generated_dbs"] != []:
        selected_documents = response['selected_rows']

        if selected_documents:

            delete_document = st.button("Delete document(s)")

            if selected_documents != st.session_state["docs_selected"]:
                st.session_state["docs_selected"] = selected_documents
                st.session_state["agent"] = None
                st.session_state['generated'] = []
                st.session_state['past'] = []
                # if st.session_state['run']:
                #     wandb.finish()
                # st.session_state['run'] = None

            # agent_tab, eval_tab, conv_tab = st.tabs(["Agent settings", "Logging settings", "Chat conversation"])
            agent_tab, conv_tab = st.tabs(["Agent settings", "Chat conversation"])


            ################################################
            ###### Setting up chat conversation agent ######
            ################################################

            with agent_tab:
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.caption("Set up the agent by inputting API key and response settings:")
                    openai_api_key = st.text_input('OpenAI API key', type='password')
                    temperature = st.slider('Select temperature', 0.0, 1.0, 0.7, help = tooltips["temperature"])
                    fetch_k = st.slider('Select k documents', 1, 7, 3, help = tooltips["select_k_documents"])
                    with st.form(key='my_form', clear_on_submit=True):
                        create_agent = st.form_submit_button('Create agent', disabled=not(openai_api_key and not st.session_state['agent']))

                with col2:
                    agent_container = st.container()
                    with agent_container:
                        if st.session_state['agent']:
                            st.write("Agent successfully initialized with the following document(s):")
                            for document in st.session_state["docs_selected"]:
                                st.caption(f"- {document['filename']}.pdf")

                            st.divider()
                            st.write("Chat with selected document(s) in the conversation tab!")

                if create_agent and openai_api_key.startswith('sk-') and not st.session_state['agent']:
                    with st.spinner('Retrieving vector stores and initializing agent...'):
                        collection_names = [document["collection name"] for document in selected_documents]
                        descriptions = [document["description"] for document in selected_documents]

                        vector_databases = backend.initialize_vector_databases(collection_names = collection_names,
                                                                               fetch_k = fetch_k)
                        
                        QA_chains, llm = backend.load_retrieval_QA_chains(openai_api_key = openai_api_key,
                                                                          temperature = temperature,
                                                                          retrievers=vector_databases)
                        
                        agent = backend.initialize_conversational_react_agent(tool_names = collection_names,
                                                                              tool_descriptions = descriptions,
                                                                              chains = QA_chains,
                                                                              llm = llm)

                        st.session_state['agent'] = agent
                        st.experimental_rerun()

            #######################################################
            ###### Evaluating responses using Weights&Biases ######
            #######################################################

            # with eval_tab:
            #     col1, col2 = st.columns([10, 8], gap="large")
            #     with col1:
            #         st.caption("Log responses onto Weights & Biases project by inputting API key and project name:")
            #         wandb_api_key = st.text_input('Weights & Biases API key', type='password')
            #         wandb_project_name = st.text_input('Weights & Biases project name', 'langchain_qa_app')
            #         with st.form(key='my_form_3', clear_on_submit=True):
            #             wandb_init= st.form_submit_button('Initialize Weights & Biases run', disabled=not(wandb_api_key and \
            #                                                                                                 wandb_project_name \
            #                                                                                                 and st.session_state['agent'] \
            #                                                                                                 and not st.session_state["run"]))
                        
            #     with col2:
            #         wandb_run_container = st.container()
            #         with wandb_run_container:
            #             if st.session_state['run']:
            #                 st.write("Responses are logged to Weights & Biases project.")
            #                 st.divider()
            #                 st.write("Chat with selected document(s) in the conversation tab!")
                            
            #     if wandb_init and not st.session_state['run']:
            #         with st.spinner('Creating Weights & Biases run...'):
            #             wandb.login(key=wandb_api_key)
            #             st.session_state['run'] =  wandb.init(project=wandb_project_name, job_type="generation")
            #             os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
            #             st.experimental_rerun()


            #####################################################################
            ###### Displaying chat history and input area to ask questions ######
            #####################################################################

            with conv_tab:
                col1, col2 = st.columns([6, 12], gap="large")
                with col1:
                    if st.session_state['agent']:
                        st.caption("Conversation agent successfully initialized!")
                    query_text = st.text_area("Input question to converse with document(s):", key='input', height=20)
                    with st.form(key='my_form_2', clear_on_submit=True):
                        submitted_query = st.form_submit_button('Send', disabled=not(query_text and st.session_state['agent']))
                with col2:
                    response_container = st.container()
                    response_container.caption("Chat history (showing last 3 responses)")

                if submitted_query:
                    
                    # output = st.session_state['agent']({"input":query_text})["output"]
                    output = st.session_state['agent'].run(query_text)
                    st.session_state['generated'].append(output)
                    st.session_state['past'].append(query_text)

                if st.session_state['generated']:
                    with response_container:
                        prev_n_messages = st.session_state['generated'][-config["LAST_N_RESPONSES"]:]
                        prev_n_queries = st.session_state['past'][-config["LAST_N_RESPONSES"]:]
                        for i in range(len(prev_n_messages)):
                            streamlit_chat.message(prev_n_queries[i], is_user=True, key=str(i) + '_user')
                            streamlit_chat.message(prev_n_messages[i], key=str(i))

                        chat_history = convert_chat_history_to_csv(past = st.session_state["past"], generated = st.session_state["generated"])

                        st.download_button('Download chat history', chat_history, 'chat_history.csv')

            #########################################################
            ###### Deleting loaded PDF files from vector store ######
            #########################################################

            if delete_document:
                collection_names_to_delete = [document["collection name"] for document in selected_documents]

                backend.remove_vector_databases(collection_names = collection_names_to_delete)

                indices_to_remove = [i for i in range(len(st.session_state.generated_dbs)) if st.session_state.generated_dbs[i]["collection name"] in collection_names_to_delete]
                st.session_state.generated_dbs = [db for db in st.session_state.generated_dbs if st.session_state.generated_dbs.index(db) not in indices_to_remove]
                st.experimental_rerun()

        else:
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['agent'] = None
            # if st.session_state['run']:
            #     wandb.finish()
            # st.session_state['run'] = None
            


if __name__ == "__main__":
    main()
