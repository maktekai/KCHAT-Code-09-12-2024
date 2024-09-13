import xml.etree.ElementTree as ET
import requests
import pandas as pd
import re
import os
from chromadb.api.client import SharedSystemClient

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import numpy as np
from dotenv import load_dotenv
import re
import urllib.parse

load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] =  os.getenv('OPENAI_API_KEY')
source_column = "KuratedContent_sourceUrl"
metadata_columns = ['Channel_about', 'Channel_keywords', 'Collection_about', 'Collection_keywords', 'File_about',
                    'File_keywords', 'KuratedContent_author', 'KuratedContent_datePublished',
                    'KuratedContent_dateModified', 'KuratedContent_keywords', 'KuratedContent_publisher',
                    'KuratedContent_sourceUrl', 'KuratedContent_WordpressPopupUrl', 'user_id', 'knowledge_Store_id','KuratedContent_headline']
main_content_columns = ['KuratedContent_Description_and_headline']
embeddingsModel=OpenAIEmbeddings()
#chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)


def download_and_print_xml(url):
    def fix_br_tags(xml_string):
        fixed_xml_string = re.sub(r'<br([^>]*)>', r'<br\1/>', xml_string)
        fixed_xml_string = fixed_xml_string.replace('&amp;', '').replace('&apos;', '').replace('&nbsp;', '').replace('&quot;','').replace(
            '&', 'and')
        return fixed_xml_string
    try:
        response = requests.get(url)
        if response.status_code == 200:
            xml_content = response.content.decode('utf-8')  # Decode using UTF-8
            fixed_xml_content = fix_br_tags(xml_content)
            return fixed_xml_content
        else:
            print(f"Error: Unable to fetch XML. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
def extract_text_from_element(element):
    try:
        def get_text_from_element(element):
            text = element.text or ''
            for child in element:
                text += get_text_from_element(child)
            return text

        # Extract text content from the provided element
        text_content = get_text_from_element(element)
        return text_content.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return None
def parse_xml(xml_text, wordPress_website_link):
    root = ET.fromstring(xml_text)
    data = {
        'about': root.find('.//span[@itemprop="about"]').text,
        'comment': root.find('.//span[@itemprop="comment"]').text,
        'encoding': root.find('.//span[@itemprop="encoding"]').text,
        'publisher': root.find('.//span[@itemprop="publisher"]').text,
        'author': root.find('.//span[@itemprop="author"]').text,
        'keywords': root.find('.//span[@itemprop="keywords"]').text,
        'Channels': []
    }
    for channel in root.findall('.//group'):
        channel_data = {
            'about': channel.find('.//span[@itemprop="about"]').text,
            'comment': channel.find('.//span[@itemprop="comment"]').text,
            'encoding': channel.find('.//span[@itemprop="encoding"]').text,
            'keywords': channel.find('.//span[@itemprop="keywords"]').text,
            'Collections': []}
        for collection in channel.findall('.//page'):
            collection_data = {
                'about': collection.find('.//span[@itemprop="about"]').text,
                'comment': collection.find('.//span[@itemprop="comment"]').text,
                'encoding': collection.find('.//span[@itemprop="encoding"]').text,
                'keywords': collection.find('.//span[@itemprop="keywords"]').text,
                'KuratedContent': []}
            for artical in collection.findall('.//link'):
                articals_data = {
                    'ID': artical.find('.//span[@itemprop="ID"]').text,
                    'sourceUrl': artical.find('.//meta[@itemprop="mainEntityOfPage"]').get('itemid'),
                    'WordpressPopupUrl': str(
                        wordPress_website_link + clean_wordpress_Link(channel_data['about'], collection_data['about'], str(
                            artical.find('.//h2[@itemprop="headline"]').text).strip())),
                    'headline': artical.find('.//h2[@itemprop="headline"]').text,
                    'author': artical.find('.//h3[@itemprop="author"]/span[@itemprop="name"]').text,
                    'description': extract_text_from_element(artical.find('.//span[@itemprop="description"]')),
                    'publisher': artical.find('.//div[@itemprop="publisher"]/meta[@itemprop="name"]').get('content'),
                    'datePublished': artical.find('.//meta[@itemprop="datePublished"]').get('content'),
                    'dateModified': artical.find('.//meta[@itemprop="dateModified"]').get('content'),
                    'keywords': artical.find('.//span[@itemprop="keywords"]').text
                }
                collection_data['KuratedContent'].append(articals_data)
            channel_data['Collections'].append(collection_data)
        data['Channels'].append(channel_data)
    return data

def sanitize_title_with_dashes(title, raw_title='', context='save'):
    title = re.sub(r'%([a-fA-F0-9][a-fA-F0-9])', lambda m: '---' + m.group(1) + '---', title)
    title = title.replace('%', '')
    title = re.sub(r'---([a-fA-F0-9][a-fA-F0-9])---', lambda m: '%' + m.group(1), title)

    def utf8_uri_encode(utf8_string, length=0, encode_ascii_characters=False):
        unicode = ''
        values = []
        num_octets = 1
        unicode_length = 0

        string_length = len(utf8_string)

        for i in range(string_length):
            value = ord(utf8_string[i])

            if value < 128:
                char = chr(value)
                encoded_char = urllib.parse.quote(char) if encode_ascii_characters else char
                encoded_char_length = len(encoded_char)
                if length and (unicode_length + encoded_char_length) > length:
                    break
                unicode += encoded_char
                unicode_length += encoded_char_length
            else:
                if len(values) == 0:
                    if value < 224:
                        num_octets = 2
                    elif value < 240:
                        num_octets = 3
                    else:
                        num_octets = 4
                values.append(value)
                if length and (unicode_length + (num_octets * 3)) > length:
                    break
                if len(values) == num_octets:
                    for j in range(num_octets):
                        unicode += '%' + format(values[j], 'x')

                    unicode_length += num_octets * 3

                    values = []
                    num_octets = 1

        return unicode

    def seems_utf8(s):
        length = len(s)
        i = 0
        while i < length:
            c = ord(s[i])
            if c < 0x80:
                n = 0  # 0bbbbbbb
            elif (c & 0xE0) == 0xC0:
                n = 1  # 110bbbbb
            elif (c & 0xF0) == 0xE0:
                n = 2  # 1110bbbb
            elif (c & 0xF8) == 0xF0:
                n = 3  # 11110bbb
            elif (c & 0xFC) == 0xF8:
                n = 4  # 111110bb
            elif (c & 0xFE) == 0xFC:
                n = 5  # 1111110b
            else:
                return False  # Does not match any model.

            i += 1
            for _ in range(n):  # n bytes matching 10bbbbbb follow ?
                if i >= length or (ord(s[i]) & 0xC0) != 0x80:
                    return False
                i += 1
        return True

    if seems_utf8(title):
        title = title.lower()
        title = utf8_uri_encode(title, 200)

    title = title.lower()

    if context == 'save':
        title = title.replace('%c2%a0', '-').replace('%e2%80%93', '-').replace('%e2%80%94', '-')
        title = title.replace('&nbsp;', '-').replace('&#160;', '-').replace('&ndash;', '-').replace('&#8211;', '-')
        title = title.replace('&mdash;', '-').replace('&#8212;', '-').replace('/', '-')
        title = re.sub(r'%[0-9a-fA-F]{2}', '', title)
        title = re.sub(r'%[0-9a-fA-F]{2}', '', title)

        title = re.sub(
            r'%[0-9a-fA-F]{2}',
            '',
            title,
            flags=re.IGNORECASE
        )

        title = re.sub(
            r'%[0-9a-fA-F]{2}',
            '',
            title,
            flags=re.IGNORECASE
        )

        title = re.sub(
            r'%[0-9a-fA-F]{2}',
            '',
            title,
            flags=re.IGNORECASE
        )

        title = re.sub(
            r'%[0-9a-fA-F]{2}',
            '',
            title,
            flags=re.IGNORECASE
        )

        title = title.replace('%c3%97', 'x')

    title = re.sub('&.+?;', '', title)
    title = title.replace('.', '-')
    title = re.sub('[^%a-z0-9 _-]', '', title)
    title = re.sub('-+', '-', title)
    title = title.strip('-')

    return title

def clean_text(text):
    cleaned_text=sanitize_title_with_dashes(text)
    # Replace empty spaces with hyphens
    cleaned_text = cleaned_text.replace(' ', '-')
    # Remove extra hyphens and convert to lowercase
    cleaned_text = re.sub(r'-+', '-', cleaned_text)
    return cleaned_text.strip('-')
def clean_wordpress_Link(channel_name, collection_name, articals_headline):
    channel_name = channel_name.replace(' ', '-')
    collection_name = collection_name.replace(' ', '-').lower()
    articals_headline = clean_text(articals_headline)
    return '/' + channel_name + '/' + collection_name + '/' + articals_headline
def Append_Single_file_Articals(data):
    Single_data_Collection = []
    for channel_data in data['Channels']:
        for collection_data in channel_data['Collections']:
            for articals_data in collection_data['KuratedContent']:
                row = {
                    'KuratedContent_article_id': articals_data['ID'],
                    'KuratedContent_sourceUrl': articals_data['sourceUrl'],
                    'KuratedContent_WordpressPopupUrl': articals_data['WordpressPopupUrl'],
                    'KuratedContent_headline': str(articals_data['headline']).lower(),
                    'KuratedContent_author': articals_data['author'],
                    'KuratedContent_description': str(articals_data['description']).lower(),
                    'KuratedContent_publisher': articals_data['publisher'],
                    'KuratedContent_datePublished': articals_data['datePublished'],
                    'KuratedContent_dateModified': articals_data['dateModified'],
                    'KuratedContent_keywords': articals_data['keywords'],
                    'Collection_about': collection_data['about'],
                    'Collection_comment': collection_data['comment'],
                    'Collection_encoding': collection_data['encoding'],
                    'Collection_keywords': collection_data['keywords'],
                    'Channel_about': channel_data['about'],
                    'Channel_comment': channel_data['comment'],
                    'Channel_encoding': channel_data['encoding'],
                    'Channel_keywords': channel_data['keywords'],
                    'File_about': data['about'],
                    'File_comment': data['comment'],
                    'File_encoding': data['encoding'],
                    'File_publisher': data['publisher'],
                    'File_author': data['author'],
                    'File_keywords': data['keywords']
                }
                Single_data_Collection.append(row)
    return Single_data_Collection
def Collect_Process_Data(parsed_data_against_urls, user_id, chatbot_id):
    Articals_Collection = []
    for data in parsed_data_against_urls:
        Articals_Collection.extend(Append_Single_file_Articals(data))
    Dataset_csv = pd.DataFrame(Articals_Collection)
    Dataset_csv['user_id'] = user_id
    Dataset_csv['knowledge_Store_id'] = chatbot_id
    Dataset_csv['KuratedContent_Description_and_headline'] = Dataset_csv['KuratedContent_headline'] + ':' + Dataset_csv[
        'KuratedContent_description']
    columns_to_drop = ['KuratedContent_description', 'KuratedContent_article_id',
                       'Collection_comment', 'Collection_encoding', 'Channel_comment', 'Channel_encoding',
                       'File_comment', 'File_encoding', 'File_author', 'File_publisher']
    Dataset_csv = Dataset_csv.drop(columns_to_drop, axis=1)
    Dataset_csv = Dataset_csv.drop_duplicates()
    Dataset_csv.replace({np.nan: "Not Specified"}, inplace=True)
    Dataset_csv.reset_index()
    return Dataset_csv
def Generate_documents_from_dataframe(processed_data):
    documents = []
    for index, row in processed_data.iterrows():
        data = row[main_content_columns[0]]
        metadata = {column: row[column] for column in metadata_columns}
        document = Document(page_content=data, metadata=metadata)
        documents.append(document)
    return documents

def TestAPISQouta():
    try:
        embeddings = embeddingsModel.embed_documents(
                [
                    "Test"
                ]
            )
        return True
    except:
        return False


def num_tokens_from_string(string: str, encoding_name="text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def CountTOkensFromDocs(documents):
    tokens_Count = 0
    for document in documents:
        tokens_Count += num_tokens_from_string(document.page_content)
    return tokens_Count

def GetDocumentsFromURL(user_id, knowledge_Store_id,urls, wordpressUrls ):
    Urls_for_chatbot=[urls]
    urls_corresponding_wordPress_website_links=[wordpressUrls]
    try:
        parsed_data_against_urls = []
        for url, wordpress_link in zip(Urls_for_chatbot, urls_corresponding_wordPress_website_links):
            parsed_data_against_urls.append(parse_xml(download_and_print_xml(url), wordpress_link))
        processed_data = Collect_Process_Data(parsed_data_against_urls, user_id, knowledge_Store_id)
        documents = Generate_documents_from_dataframe(processed_data)
        return documents,CountTOkensFromDocs(documents)
    except Exception as e:
        print(e)
        raise Exception(str(e))
    return False,-1

def create_chatbot(documents):
    try:
        texts = text_splitter.split_documents(documents)
        chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())
        chroma_db.add_documents(texts)
        chroma_db._client._system.stop()
        SharedSystemClient._identifer_to_system.pop(chroma_db._client._identifier, None)
        chroma_db = None
        print("docuemnts uploaded")
        return True
    except Exception as e:
        print(e)
        pass
    return False

# def create_chatbot(user_id, knowledge_Store_id,urls, wordpressUrls ):
#     Urls_for_chatbot=[urls]
#     urls_corresponding_wordPress_website_links=[wordpressUrls]
#     try:
#         parsed_data_against_urls = []
#         for url, wordpress_link in zip(Urls_for_chatbot, urls_corresponding_wordPress_website_links):
#             parsed_data_against_urls.append(parse_xml(download_and_print_xml(url), wordpress_link))
#         processed_data = Collect_Process_Data(parsed_data_against_urls, user_id, knowledge_Store_id)
#         documents = Generate_documents_from_dataframe(processed_data)
#         texts = text_splitter.split_documents(documents)
#         chroma_db.add_documents(texts)
#         return True
#     except Exception as e:
#         print(e)
#         pass
#     return False








# import xml.etree.ElementTree as ET
# import requests
# import pandas as pd
# import re
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document
# import numpy as np
# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env.
# os.environ['OPENAI_API_KEY'] =  os.getenv('OPENAI_API_KEY')
# 
# 
# source_column = "KuratedContent_sourceUrl"
# metadata_columns = ['Channel_about', 'Channel_keywords', 'Collection_about', 'Collection_keywords', 'File_about',
#                     'File_keywords', 'KuratedContent_author', 'KuratedContent_datePublished',
#                     'KuratedContent_dateModified', 'KuratedContent_keywords', 'KuratedContent_publisher',
#                     'KuratedContent_sourceUrl', 'KuratedContent_WordpressPopupUrl', 'user_id', 'knowledge_Store_id','KuratedContent_headline']
# main_content_columns = ['KuratedContent_Description_and_headline']
# 
# chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
# def download_and_print_xml(url):
#     def fix_br_tags(xml_string):
#         fixed_xml_string = re.sub(r'<br([^>]*)>', r'<br\1/>', xml_string)
#         fixed_xml_string = fixed_xml_string.replace('&amp;', '').replace('&apos;', '').replace('&nbsp;', '').replace(
#             '&', 'and')
#         return fixed_xml_string
# 
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             xml_content = response.content.decode('utf-8')  # Decode using UTF-8
#             fixed_xml_content = fix_br_tags(xml_content)
#             return fixed_xml_content
#         else:
#             print(f"Error: Unable to fetch XML. Status code: {response.status_code}")
#     except Exception as e:
#         print(f"Error: {e}")
# def extract_text_from_element(element):
#     try:
#         def get_text_from_element(element):
#             text = element.text or ''
#             for child in element:
#                 text += get_text_from_element(child)
#             return text
# 
#         # Extract text content from the provided element
#         text_content = get_text_from_element(element)
#         return text_content.strip()
#     except Exception as e:
#         print(f"Error extracting text: {e}")
#     return None
# def parse_xml(xml_text, wordPress_website_link):
#     root = ET.fromstring(xml_text)
#     data = {
#         'about': root.find('.//span[@itemprop="about"]').text,
#         'comment': root.find('.//span[@itemprop="comment"]').text,
#         'encoding': root.find('.//span[@itemprop="encoding"]').text,
#         'publisher': root.find('.//span[@itemprop="publisher"]').text,
#         'author': root.find('.//span[@itemprop="author"]').text,
#         'keywords': root.find('.//span[@itemprop="keywords"]').text,
#         'Channels': []
#     }
# 
#     for channel in root.findall('.//group'):
#         channel_data = {
#             'about': channel.find('.//span[@itemprop="about"]').text,
#             'comment': channel.find('.//span[@itemprop="comment"]').text,
#             'encoding': channel.find('.//span[@itemprop="encoding"]').text,
#             'keywords': channel.find('.//span[@itemprop="keywords"]').text,
#             'Collections': []}
#         for collection in channel.findall('.//page'):
#             collection_data = {
#                 'about': collection.find('.//span[@itemprop="about"]').text,
#                 'comment': collection.find('.//span[@itemprop="comment"]').text,
#                 'encoding': collection.find('.//span[@itemprop="encoding"]').text,
#                 'keywords': collection.find('.//span[@itemprop="keywords"]').text,
#                 'KuratedContent': []}
#             for artical in collection.findall('.//link'):
#                 articals_data = {
#                     'ID': artical.find('.//span[@itemprop="ID"]').text,
#                     'sourceUrl': artical.find('.//meta[@itemprop="mainEntityOfPage"]').get('itemid'),
#                     'WordpressPopupUrl': str(
#                         wordPress_website_link + clean_wordpress_Link(channel_data['about'], collection_data['about'], (
#                             artical.find('.//h2[@itemprop="headline"]').text).strip())),
#                     'headline': artical.find('.//h2[@itemprop="headline"]').text,
#                     'author': artical.find('.//h3[@itemprop="author"]/span[@itemprop="name"]').text,
#                     'description': extract_text_from_element(artical.find('.//span[@itemprop="description"]')),
#                     'publisher': artical.find('.//div[@itemprop="publisher"]/meta[@itemprop="name"]').get('content'),
#                     'datePublished': artical.find('.//meta[@itemprop="datePublished"]').get('content'),
#                     'dateModified': artical.find('.//meta[@itemprop="dateModified"]').get('content'),
#                     'keywords': artical.find('.//span[@itemprop="keywords"]').text
#                 }
#                 collection_data['KuratedContent'].append(articals_data)
#             channel_data['Collections'].append(collection_data)
#         data['Channels'].append(channel_data)
#     return data
# def clean_text(text):
#     # Use regular expression to remove unwanted characters
#     cleaned_text = text.replace("'", "")
#     cleaned_text = cleaned_text.replace("/", "")
#     cleaned_text = re.sub(r'[^a-zA-Z0-9\'’-“”]', '-', cleaned_text)
#     # Replace empty spaces with hyphens
#     cleaned_text = cleaned_text.replace(' ', '-')
#     # Remove extra hyphens and convert to lowercase
#     cleaned_text = re.sub(r'-+', '-', cleaned_text).lower()
#     return cleaned_text.strip('-')
# def clean_wordpress_Link(channel_name, collection_name, articals_headline):
#     channel_name = channel_name.replace(' ', '-')
#     collection_name = collection_name.replace(' ', '-').lower()
#     articals_headline = clean_text(articals_headline.replace(' ', '-').lower())
#     return '/' + channel_name + '/' + collection_name + '/' + articals_headline
# def check_vector_store_exists(vector_store_name):
#     vector_store_path = os.path.join(os.getcwd(), vector_store_name)
# 
#     try:
#         # Using os.path.join ensures a valid path
#         return os.path.exists(vector_store_path)
#     except Exception as e:
#         # Handle any exceptions that might occur during path checking
#         print(f"Error checking vector store: {e}")
#         return False
# def Append_Single_file_Articals(data):
#     Single_data_Collection = []
#     for channel_data in data['Channels']:
#         for collection_data in channel_data['Collections']:
#             for articals_data in collection_data['KuratedContent']:
#                 row = {
#                     'KuratedContent_article_id': articals_data['ID'],
#                     'KuratedContent_sourceUrl': articals_data['sourceUrl'],
#                     'KuratedContent_WordpressPopupUrl': articals_data['WordpressPopupUrl'],
#                     'KuratedContent_headline': articals_data['headline'],
#                     'KuratedContent_author': articals_data['author'],
#                     'KuratedContent_description': articals_data['description'],
#                     'KuratedContent_publisher': articals_data['publisher'],
#                     'KuratedContent_datePublished': articals_data['datePublished'],
#                     'KuratedContent_dateModified': articals_data['dateModified'],
#                     'KuratedContent_keywords': articals_data['keywords'],
#                     'Collection_about': collection_data['about'],
#                     'Collection_comment': collection_data['comment'],
#                     'Collection_encoding': collection_data['encoding'],
#                     'Collection_keywords': collection_data['keywords'],
#                     'Channel_about': channel_data['about'],
#                     'Channel_comment': channel_data['comment'],
#                     'Channel_encoding': channel_data['encoding'],
#                     'Channel_keywords': channel_data['keywords'],
#                     'File_about': data['about'],
#                     'File_comment': data['comment'],
#                     'File_encoding': data['encoding'],
#                     'File_publisher': data['publisher'],
#                     'File_author': data['author'],
#                     'File_keywords': data['keywords']
#                 }
#                 Single_data_Collection.append(row)
#     return Single_data_Collection
# def Collect_Process_Data(parsed_data_against_urls, user_id, chatbot_id):
#     Articals_Collection = []
#     for data in parsed_data_against_urls:
#         Articals_Collection.extend(Append_Single_file_Articals(data))
#     Dataset_csv = pd.DataFrame(Articals_Collection)
#     Dataset_csv['user_id'] = user_id
#     Dataset_csv['knowledge_Store_id'] = chatbot_id
#     Dataset_csv['KuratedContent_Description_and_headline'] = Dataset_csv['KuratedContent_headline'] + ':' + Dataset_csv[
#         'KuratedContent_description']
#     columns_to_drop = ['KuratedContent_description', 'KuratedContent_article_id',
#                        'Collection_comment', 'Collection_encoding', 'Channel_comment', 'Channel_encoding',
#                        'File_comment', 'File_encoding', 'File_author', 'File_publisher']
#     Dataset_csv = Dataset_csv.drop(columns_to_drop, axis=1)
#     Dataset_csv = Dataset_csv.drop_duplicates()
#     Dataset_csv.replace({np.nan: "Not Specified"}, inplace=True)
#     Dataset_csv.reset_index()
#     return Dataset_csv
# def Generate_documents_from_dataframe(processed_data):
#     documents = []
#     for index, row in processed_data.iterrows():
#         data = row[main_content_columns[0]]
#         metadata = {column: row[column] for column in metadata_columns}
#         document = Document(page_content=data, metadata=metadata)
#         documents.append(document)
#     return documents
# def create_chatbot(user_id, knowledge_Store_id,urls, wordpressUrls ):
#     Urls_for_chatbot=[urls]
#     urls_corresponding_wordPress_website_links=[wordpressUrls]
#     try:
#         parsed_data_against_urls = []
#         for url, wordpress_link in zip(Urls_for_chatbot, urls_corresponding_wordPress_website_links):
#             parsed_data_against_urls.append(parse_xml(download_and_print_xml(url), wordpress_link))
#         processed_data = Collect_Process_Data(parsed_data_against_urls, user_id, knowledge_Store_id)
#         documents = Generate_documents_from_dataframe(processed_data)
#         texts = text_splitter.split_documents(documents)
#         chroma_db.add_documents(texts)
#         return True
#     except Exception as e:
#         print(e)
#         pass
#     return False
