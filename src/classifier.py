"""
Document classification module for pdfkg.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt template in English for document type classification
# classification_prompt = ChatPromptTemplate(
#     input_variables=["document_text"],
#     template=(
#         "You are an intelligent assistant specialized in classifying document types. "
#         "Given the content of the document below, identify its type as one of the following options: "
#         "Technical Manual, Purchase Order, Financial Report, Contract, Conference Paper or Other. "
#         "Respond with ONLY the document type name.\n\n"
#         "Document Content:\n{document_text}"
#     )
# )


# def get_classification_system_prompt() -> str:
#     return (
#         "You are an intelligent assistant specialized in classifying document types. "
#         "Given the content of the document below, identify its type as one of the following options:"
#         "Technical Manual, Purchase Order, Financial Report, Contract, Conference Paper or Other. "
#         "Respond with ONLY the document type name.\n\n"
#     )

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful document classification assistant."),
    ("human", "Classify this document: {document_text}")
])


# Initialize the OpenAI language model (GPT-4)
language_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classify_document(document_text: str) -> str:
    """
    Classify the provided document text and return its type.

    Args:
        document_text (str): Full text of the document to classify.

    Returns:
        str: One of ["Technical Manual", "Purchase Order", "Financial Report", "Contract", "Conference Paper","Other"].
    """
    # Truncate text to respect token limits
    truncated_text = document_text[:4000]

    # Build and run the classification chain
    #classification_chain = LLMChain(
    #    llm=language_model,
    #   prompt=classification_prompt
    #)
    classification_chain = classification_prompt | language_model | StrOutputParser()
    classification_result = classification_chain.invoke({"document_text": truncated_text})
    document_type = classification_result.strip()
    return document_type

# Expose only the classify_document function
__all__ = ["classify_document"]
