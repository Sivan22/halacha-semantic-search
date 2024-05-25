import streamlit as st
from streamlit.logger import get_logger
import datasets
import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import util



LOGGER = get_logger(__name__)


@st.cache_data
def get_df() ->object:
    ds = datasets.load_dataset('sivan22/yalkut-yosef-embeddings')
    df = pd.DataFrame.from_dict(ds['train'])
    return df

@st.cache_resource
def get_model()->object:
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {'device': 'cpu'} #'cpu' or 'cuda'
    encode_kwargs = {'normalize_embeddings': False}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings_model

@st.cache_resource
def get_chat_api(api_key:str):
    chat = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=api_key)
    return chat


def get_results(embeddings_model,input,df,num_of_results) -> pd.DataFrame:
    embeddings = embeddings_model.embed_query('query: '+ input)
    df['similarity'] = df['embeddings'].apply(lambda x: util.dot_score(x,embeddings))
    results = df.sort_values(by='similarity', ascending=False)
    return results.head(num_of_results)

def get_llm_results(query,chat,results):   

    prompt_template = PromptTemplate.from_template(
        """
    the question is: {query}
    the possible answers are:
    {answers}

    """   )

    messages = [
        SystemMessage(content="You're a helpful assistant. given a question, filter and sort the possible answers to the given question by relevancy, drop the irrelevant answers and return the results in the following JSON format (scores are between 0 and 1): {\"answer\": \"score\", \"answer\": \"score\"}. "),
        HumanMessage(content=prompt_template.format(query=query, answers=str.join('\n', results['text'].head(10).tolist()))),
    ]

    response =  chat.invoke(messages)
    llm_results_df = pd.read_json(response.content, orient='index')
    return llm_results_df



def run():
    
    st.set_page_config(
        page_title=" חיפוש סמנטי בילקוט יוסף",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"    
    )
    
    st.write("חיפוש חכם בספר ילקוט יוסף קיצור שולחן ערוך")
    
    embeddings_model = get_model()    
    df = get_df()
    
    user_input = st.text_input('כתוב כאן את שאלתך', placeholder='כמה נרות מדליקים בכל לילה מלילות החנוכה')    
    num_of_results = st.sidebar.slider('מספר התוצאות שברצונך להציג:',1,25,5)
    use_llm = st.sidebar.checkbox("השתמש במודל שפה כדי לשפר תוצאות", False)
    openAikey = st.sidebar.text_input("OpenAI API key", type="password")
    
    if (st.button('חפש') or user_input) and user_input!="":
        
        results = get_results(embeddings_model,user_input,df,num_of_results)  

        if use_llm:
            chat = get_chat_api(openAikey)
            llm_results = get_llm_results(user_input,chat,results)
            st.write(llm_results) 

        else:
            st.write(results[['siman','sek','text']].head(10))
            
       
if __name__ == "__main__":
    run()
