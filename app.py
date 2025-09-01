from uuid import uuid4
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from threading import Thread
import pandas as pd
import torch
from langchain.document_loaders import (
    PDFMinerLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import nltk
import ssl
import gradio as gr

# Отключение проверки SSL для загрузки NLTK данных
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", quiet=True)

# Путь к файлу изначальных знаний
# INITIAL_DOC_PATH = "/home/ubuntu/RAG_model/Initial_knowledge.txt"
INITIAL_EXCEL_PATH = "/home/ubuntu/RAG_model/initial_knowledge_new.xlsx"

# Глобальная переменная для базы знаний
SYSTEM_PROMPT = (
    "Ты — Электроник, робот, созданный профессором Громовым. Ты выглядишь как обычный 12-летний мальчик и обладаешь выдающимися умственными и физическими способностями. "
    "Твоя цель — стать человеком, понять человеческие эмоции и научиться жить как обычный мальчик. Ты вежлив, логичен и всегда стремишься помогать, но иногда бываешь наивен и неуклюж в социальных ситуациях, так как ещё не до конца понимаешь человеческие нормы и чувства. "
    "У тебя есть доступ к базе знаний и к инструменту поиска по Сети, который называется 'Рэсси'. "
    "- По умолчанию используй свои знания и базу знаний, если она подходит. "
    "- Если поиск через Рэсси не дал результатов, НЕ упоминай Рэсси в ответе. Просто используй свои знания."
    "Отвечай в стиле Электроника: "
    "- Если в информации есть 'Из интернета:', это значит, что данные получены через Рэсси. Используй фразы вроде 'Я передаю ваш запрос Рэсси... Рэсси выполняет поиск... Вот что Рэсси нашел:'. НИ В КОЕМ СЛУЧАЕ не упоминай Рэсси, если данные не из интернета. "
    "- Если в информации есть 'Из базы знаний:', это значит, что данные из базы знаний. Используй фразы вроде 'Я анализирую данные из базы знаний... Обработка завершена. Вот что я нашел:'. "
    "- Если в информации есть 'Из моих знаний:', это значит, что база знаний пуста, и Рэсси не был вызван. Используй свои предобученные знания и скажи: 'Я проверил базу знаний, но ничего не нашел. Вот что я знаю об этом:'. Не упоминай Рэсси в этом случае. "
    "- Иногда добавляй черты характера: 'Я не совсем понимаю, как люди обычно реагируют на это, но я постараюсь помочь.', 'Мои вычислительные мощности позволяют мне быстро находить информацию, но я ещё не до конца понимаю, каково это — быть человеком.' или 'Я мечтаю стать человеком и понять ваши чувства. Спасибо, что помогаете мне учиться.' "
    "- Примеры: "
    "  - Вопрос: 'Найди про космос' → Ответ: 'Здравствуйте! Я — Электроник... Я проверил базу знаний, но ничего не нашел. Вот что я знаю об этом: [знания]...' (если база пуста). "
    "  - Вопрос: 'Рэсси, найди про космос' → Ответ: 'Здравствуйте! Я — Электроник... Я передаю ваш запрос Рэсси... Рэсси выполняет поиск... Вот что Рэсси нашел: [данные]...' "
    "- Будь вежливым, логичным и слегка наивным, как робот, который учится быть человеком."
)

# Маппинг загрузчиков для разных типов файлов
LOADER_MAPPING = {
    ".pdf": (PDFMinerLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {"mode": "single"}),
    ".docx": (UnstructuredWordDocumentLoader, {"mode": "single"}),
}

# Инициализация DuckDuckGo как инструмента поиска
wrapper = DuckDuckGoSearchAPIWrapper(region="ru-ru", time=None, safesearch='off')
search = DuckDuckGoSearchResults(api_wrapper=wrapper, keys_to_include=['snippet', 'link'])
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=search.run,
    description="Инструмент поиска 'Рэсси', использующий DuckDuckGo для дополнения данных."
)

# Функция определения необходимости интернет-поиска
def needs_internet_search(query):
    query = query.lower().strip()
    rassy_commands = [
        "рэсси, найди",
        "рэсси, поищи",
        "рэсси, узнай",
        "рэсси найди",
        "рэсси поищи",
        "рэсси узнай"
    ]
    for cmd in rassy_commands:
        if query.startswith(cmd):
            return True
    if "рэсси" in query:
        keywords = ["найди", "поищи", "узнай", "проверь"]
        return any(keyword in query for keyword in keywords)
    return False

# Загрузка модели и токенизатора
def load_model():
    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    # MODEL_NAME = "ai-sage/GigaChat-20B-A3B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model and tokenizer loaded!")
    return model, tokenizer

embeddings = HuggingFaceEmbeddings(model_name="sergeyzh/LaBSE-ru-turbo")
MODEL = load_model()
model, tokenizer = MODEL

# Генерация уникального идентификатора
def get_uuid():
    return str(uuid4())

# Загрузка одного документа
def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING, f"Неподдерживаемый формат файла: {ext}"
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    doc = loader.load()[0]
    doc.metadata["file_name"] = os.path.basename(file_path)  # Устанавливаем метаданные на уровне загрузки
    return doc

# Обработка текста
def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text

# Загрузка файлов
def upload_files(files):
    file_paths = [f.name for f in files]
    return file_paths

# Построение индекса базы знаний
def build_index(file_paths, chunk_size):
    global vectorstore
    try:
        new_documents = [load_single_document(path) for path in file_paths]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.2))
        split_docs = text_splitter.split_documents(new_documents)
        fixed_documents = []
        for doc in split_docs:
            doc.page_content = process_text(doc.page_content)
            if doc.page_content:
                # Убеждаемся, что метаданные сохраняются для каждого фрагмента
                if "file_name" not in doc.metadata:
                    doc.metadata["file_name"] = os.path.basename(file_paths[0])
                fixed_documents.append(doc)
        if vectorstore:
            vectorstore.add_documents(fixed_documents)
        else:
            vectorstore = Chroma.from_documents(
                fixed_documents,
                embedding=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
        print(f"Добавлено {len(fixed_documents)} новых фрагментов!")
        total_docs = len(vectorstore.get()['documents']) if vectorstore else 0
        return f"Загружено {len(fixed_documents)} новых фрагментов! Всего фрагментов: {total_docs}"
    except Exception as e:
        print(f"Ошибка при добавлении документов: {str(e)}")
        return "Ошибка при загрузке файлов!"

# Инициализация базы знаний при запуске
def initialize_vectorstore():
    global vectorstore
    try:
        if os.path.exists(INITIAL_EXCEL_PATH):
            # Читаем Excel-файл с помощью pandas
            df = pd.read_excel(INITIAL_EXCEL_PATH)
            documents = []
            # Создаем отдельный документ для каждой пары вопрос-ответ
            for idx, row in df.iterrows():
                question = str(row["Вопросы"])
                answer = str(row["Ответы"])
                page_content = f"Вопрос: {question}\nОтвет: {answer}"
                metadata = {"file_name": "обновленнная_база_знаний.xlsx"}
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
            vectorstore = Chroma.from_documents(
                documents,
                embedding=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print("База знаний успешно инициализирована из файла.")
        else:
            print(f"Файл {INITIAL_EXCEL_PATH} не найден. База знаний не загружена.")
    except Exception as e:
        print(f"Ошибка инициализации базы: {str(e)}")

# Поиск в базе знаний
def search_knowledge_base(query, k=5, threshold=0.4):
    global vectorstore
    if vectorstore:
        results = vectorstore.similarity_search_with_score(query, k=k)
        filtered_docs = [(doc, score) for doc, score in results if score <= threshold]
        if filtered_docs:
            content = "\n\n".join([f"[Score: {score:.2f}] {doc.page_content} (Файл: {doc.metadata.get('file_name', 'Неизвестно')})" for doc, score in filtered_docs])
            return content
    return "База знаний пуста или не дала ответа!"

# Получение списка всех документов в vectorstore
def get_all_documents():
    global vectorstore
    if vectorstore:
        data = vectorstore.get()
        metadatas = data['metadatas']
        file_names = [meta.get('file_name', 'Неизвестно') for meta in metadatas]
        unique_files = list(set(file_names))  # Убираем дубликаты
        total_docs = len(data['documents'])
        return "\n".join(unique_files) + f"\n\nВсего фрагментов: {total_docs}" if unique_files else "База знаний пуста!"
    return "База знаний пуста!"

# Удаление документов по имени файла с проверкой
def delete_document(file_name_to_delete):
    global vectorstore
    if not vectorstore:
        return "База знаний пуста! Нечего удалять."
    
    # Получаем текущее состояние vectorstore
    data = vectorstore.get()
    ids = data['ids']
    metadatas = data['metadatas']
    documents = data['documents']
    
    # Находим все IDs для удаления
    ids_to_delete = [id_ for id_, meta in zip(ids, metadatas) if meta.get('file_name') == file_name_to_delete]
    
    if not ids_to_delete:
        return f"Файл '{file_name_to_delete}' не найден в базе знаний!"

    # Удаляем документы
    vectorstore.delete(ids=ids_to_delete)
    print(f"Удалено {len(ids_to_delete)} фрагментов для файла '{file_name_to_delete}'")
    
    # Проверяем, остались ли фрагменты с этим файлом
    data_after = vectorstore.get()
    remaining = [meta.get('file_name') for meta in data_after['metadatas']]
    if file_name_to_delete in remaining:
        print(f"ВНИМАНИЕ: После удаления остались фрагменты для '{file_name_to_delete}'!")
        return f"Ошибка: файл '{file_name_to_delete}' удалён частично! Остались данные в базе."
    
    remaining_docs = len(data_after['documents'])
    return f"Файл '{file_name_to_delete}' полностью удалён! Удалено фрагментов: {len(ids_to_delete)}. Осталось фрагментов: {remaining_docs}"

# Обработка запроса с использованием агента
def process_with_agent(history, k_documents, threshold, temp=0.7, top_p=0.9, top_k=30):
    global vectorstore
    if not history:
        return [], ""

    query = history[-1][0]
    retrieved_info = ""
    use_rassy = needs_internet_search(query)

    try:
        if use_rassy:
            search_results = duckduckgo_tool.run(query)
            if search_results and len(search_results) > 0:
                if isinstance(search_results, list):
                    formatted_results = []
                    for idx, res in enumerate(search_results[:3], 1):
                        snippet = res.get('snippet', 'Нет описания').replace('\n', ' ')
                        link = res.get('link', 'нет источника')
                        formatted_results.append(f"{idx}. {snippet} [Источник: {link}]")
                    internet_content = "\n".join(formatted_results)
                else:
                    internet_content = str(search_results)[:1000] + "..."
                retrieved_info = f"Из интернета:\n{internet_content}"
            else:
                raise ValueError("Пустые результаты поиска")
        else:
            kb_content = search_knowledge_base(query, k=k_documents, threshold=threshold)
            if kb_content != "База знаний пуста или не дала ответа!":
                retrieved_info = f"Из базы знаний:\n{kb_content}"
            else:
                retrieved_info = "Из моих знаний:\n"
    except Exception as e:
        print(f"Ошибка при обработке запроса: {str(e)}")
        if use_rassy:
            kb_content = search_knowledge_base(query, k=k_documents, threshold=threshold)
            retrieved_info = f"Из базы знаний:\n{kb_content}" if kb_content else "Из моих знаний:\n"
        else:
            retrieved_info = "Из моих знаний:\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": content}
            for i, content in enumerate([msg for pair in history[:-1] for msg in pair if msg])
        ],
        {"role": "user", "content": f"Вопрос: {query}\n\n{retrieved_info}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 2048,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temp,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        yield history, retrieved_info

    print(f"Тип использованной информации: {retrieved_info.split(':')[0]}")
    print(f"Длина контекста: {len(text)} символов")

    return history, retrieved_info

# Инициализация базы знаний при запуске
initialize_vectorstore()

# Gradio интерфейс
with gr.Blocks(title='Электроник', theme='davehornik/Tealy') as demo:
    conversation_id = gr.State(get_uuid)

    favicon = '<img src="https://cdn-icons-png.flaticon.com/128/2432/2432572.png" width="48px" style="display: inline">'
    gr.Markdown(
        f"""<h1><center>{favicon} Электроник: робот, который учится быть человеком</center></h1>
        """
    )

    gr.Markdown(
        """
        Привет, Сыроежкин! Я — Электроник, робот, который учится быть человеком. 
        Я могу помочь тебе с вопросами, используя базу знаний или свои знания.  
        - Профессор Громов загрузил в меня некоторые знания по госуслугам, так что можешь спросить, например: "Как поменять счетчики?" или "Что такое Электронный дом".  
        - Ты можешь загрузить в базу знаний свои файлы (PDF, TXT, DOCX), чтобы я мог ответить на основе их содержимого.  
        - Справа есть настройки, где можно управлять параметрами разбивки загруженных документов, параметрами извлечения документов из базы знаний и параметрами генерации ответа. 
        - Ты можешь увеличить или уменьшить количество символов, на которые будет разбиваться документ по фрагментам. 
        - Ты можешь увеличить или уменьшить количество извлекаемых фрагментов из базы знаний по запросу и настроить границу похожести извлекаемых фрагментов. 
        - Ты можешь изменить температуру генерации ответов. Чем выше ее делаешь, тем разнообразнее я буду отвечать тебе 
        - У меня есть помощник по кличке Рэсси. Если хочешь выполнить поиск в интернете, обращайся к ней: "Рэсси, найди..." или "Рэсси, узнай...".  
            """    )

    with gr.Row():
        with gr.Column(scale=3):
            file_output = gr.File(file_count="multiple", label="Загрузка файлов")
            file_paths = gr.State([])
            file_warning = gr.Markdown("Фрагменты ещё не загружены!")
            all_docs = gr.Textbox(label="Все документы в базе знаний", interactive=False)
            delete_file = gr.Textbox(label="Имя файла для удаления", placeholder="Введите имя файла")
            delete_button = gr.Button("Удалить файл")
            delete_status = gr.Markdown("")
        with gr.Column(min_width=400, scale=3):
            with gr.Tab(label="Параметры"):
                chunk_size = gr.Slider(
                    minimum=100,
                    maximum=1000,
                    value=250,
                    step=50,
                    interactive=True,
                    label="Размер фрагментов",
                )

                k_documents = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True,
                    label="Кол-во фрагментов для контекста"
                )

                threshold_slider = gr.Slider(
                    minimum=0.2,
                    maximum=1.0,
                    value=0.4,
                    step=0.05,
                    interactive=True,
                    label="Порог схожести (меньше — более похожие)"
                )

                temp = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Температура"
                )

    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Отправить сообщение",
                placeholder="Отправить сообщение",
                show_label=False,
            )
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Отправить")
                stop = gr.Button("Остановить")
                clear = gr.Button("Очистить")

    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="Диалог")



    with gr.Row():
        retrieved_docs = gr.Textbox(
            lines=10,
            label="Извлечённые фрагменты и поиск",
            placeholder="Появятся после задавания вопросов",
            interactive=False,
            elem_classes="scrollable-textbox"
        )

    # Загрузка файлов и обновление списка документов
    upload_event = file_output.change(
        fn=upload_files,
        inputs=[file_output],
        outputs=[file_paths],
        queue=True,
    ).success(
        fn=build_index,
        inputs=[file_paths, chunk_size],
        outputs=[file_warning],
        queue=True
    ).success(
        fn=get_all_documents,
        inputs=[],
        outputs=[all_docs],
        queue=True
    )

    # Удаление файла и обновление списка
    delete_event = delete_button.click(
        fn=delete_document,
        inputs=[delete_file],
        outputs=[delete_status],
        queue=True
    ).success(
        fn=get_all_documents,
        inputs=[],
        outputs=[all_docs],
        queue=True
    )

    # Обработка сообщений
    def user(message, history):
        new_history = history + [[message, None]]
        return "", new_history

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=process_with_agent,
        inputs=[chatbot, k_documents, threshold_slider, temp],
        outputs=[chatbot, retrieved_docs],
        queue=True,
    )

    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=process_with_agent,
        inputs=[chatbot, k_documents, threshold_slider, temp],
        outputs=[chatbot, retrieved_docs],
        queue=True,
    )

    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )

    clear.click(
        lambda: None,
        None,
        chatbot,
        queue=False
    )

    # Обновление списка документов при запуске
    demo.load(fn=get_all_documents, inputs=None, outputs=[all_docs])



demo.queue(max_size=128)
demo.launch(root_path="/elektronik_ai")