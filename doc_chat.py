import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Specify a custom download directory (optional)
nltk.data.path.append('./nltk_data')

# Define a function to download NLTK data if not already present
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir='./nltk_data')

download_nltk_data()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    print("\nText extracted from pdf...")
    return text

def clean_text(text):
    # Replace newlines and extra spaces
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text

def process_text(text):
    text = clean_text(text)
    sentences = sent_tokenize(text)
    print("\nText processed...")
    return sentences

def load_qa_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(tokenizer)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    print("\nModel loaded in pipeline...")
    return qa_pipeline

def get_relevant_context(question, sentences, max_len=512):
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    vectors = vectorizer.toarray()
    question_vector = vectors[0]
    sentence_vectors = vectors[1:]
    similarities = cosine_similarity([question_vector], sentence_vectors)[0]
    
    sorted_indices = np.argsort(similarities)[::-1]
    context = ""
    current_len = 0
    
    for idx in sorted_indices:
        sentence = sentences[idx]
        sentence_len = len(sentence.split())
        if current_len + sentence_len > max_len:
            break
        context += " " + sentence
        current_len += sentence_len
    
    return context.strip()

def answer_question(qa_model, question, context):
    result = qa_model(question=question, context=context)
    return result['answer']

def score_sentences(sentences):
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    sentence_vectors = vectorizer.toarray()
    doc_vector = np.mean(sentence_vectors, axis=0).reshape(1, -1)
    scores = cosine_similarity(doc_vector, sentence_vectors)[0]
    return scores

def summarize_text(sentences, scores, top_n=5):
    top_sentence_indices = np.argsort(scores)[-top_n:]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    # print(top_sentences)
    summary = '\n'.join(top_sentences)
    return summary

def get_section_context(section_heading, sentences, max_len=512):
    section_sentences = []
    in_section = False
    for sentence in sentences:
        if section_heading.lower() in sentence.lower():
            in_section = True
        elif in_section and len(sentence.strip()) == 0:
            break
        
        if in_section:
            section_sentences.append(sentence)
    
    return '\n'.join(section_sentences).strip()

def chatbot():
    pdf_path = input("Please enter the document path you want to chat with: ")
    
    text = extract_text_from_pdf(pdf_path)
    sentences = process_text(text)
    model_path = "D:/sem-7/temp/Document_Chat/models--distilbert-base-cased-distilled-squad"
    qa_model = load_qa_model(model_path)
    
    print("\nChatbot is ready! Ask your questions or 'summarize document' or 'summarize <point>'.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "end"]:
            break
        elif user_input.lower() == "summarize document":
            scores = score_sentences(sentences)
            summary = summarize_text(sentences, scores, top_n=5)
            print(f"Summary of the document:\n{summary}\n")
        elif user_input.lower().startswith("summarize"):
            _, section_heading = user_input.split(maxsplit=1)
            section_context = get_section_context(section_heading, sentences)
            if section_context:
                section_sentences = sent_tokenize(section_context)
                scores = score_sentences(section_sentences)
                summary = summarize_text(section_sentences, scores, top_n=5)
                print(f"Summary of '{section_heading}':\n{summary}\n")
            else:
                print(f"No section found with heading '{section_heading}'.\n")
        else:
            context = get_relevant_context(user_input, sentences)
            answer = answer_question(qa_model, user_input, context)
            print(f"Chatbot: {answer}\n")

chatbot()