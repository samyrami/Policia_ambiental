import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from html_template import logo
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re
import gc
import os

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    st.error("Error: OPENAI_API_KEY not found in environment variables")
    st.stop()
    
st.set_page_config(page_title="EcoPoliciApp", layout="centered")

class LawDocumentProcessor:
    def __init__(self, pdf_directory="data", index_directory="faiss_index"):
        self.pdf_directory = pdf_directory
        self.index_directory = index_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Asegurar que los directorios existan
        os.makedirs(self.pdf_directory, exist_ok=True)
        os.makedirs(self.index_directory, exist_ok=True)

    def load_vector_store(self):
        """
        Carga el vector store existente con manejo seguro de deserialización
        """
        index_path = os.path.join(self.index_directory, "index.faiss")
        try:
            if os.path.exists(index_path):
                return FAISS.load_local(
                    self.index_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Solo para archivos de confianza
                )
            return None
        except Exception as e:
            st.error(f"Error cargando vector store: {str(e)}")
            return None

    def process_documents(self):
        try:
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                st.warning("No se encontraron archivos PDF en el directorio data.")
                return None

            documents = []
            for pdf_file in pdf_files:
                try:
                    file_path = os.path.join(self.pdf_directory, pdf_file)
                    # Verificar el archivo antes de procesarlo
                    with open(file_path, 'rb') as f:
                        # Verificar el encabezado PDF
                        header = f.read(4)
                        if header != b'%PDF':
                            st.warning(f"Archivo corrupto o inválido: {pdf_file}")
                            continue
                    
                    loader = PyPDFLoader(file_path)
                    doc_pages = loader.load()
                    if doc_pages:
                        documents.extend(doc_pages)
                        st.success(f"Procesado exitosamente: {pdf_file}")
                    else:
                        st.warning(f"No se pudo extraer contenido de: {pdf_file}")
                
                except Exception as e:
                    st.error(f"Error procesando {pdf_file}: {str(e)}")
                    continue

            if not documents:
                st.warning("No se pudo extraer contenido de ningún PDF.")
                return None

            texts = self.text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(self.index_directory)
            return vectorstore
        
        except Exception as e:
            st.error(f"Error procesando documentos: {str(e)}")
            return None

def setup_retrieval_chain(vector_store):
    """Configura la cadena de recuperación para consultas"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return retrieval_chain

def main():
    processor = LawDocumentProcessor()
    
    # Verificar si ya existe el índice FAISS
    if os.path.exists(os.path.join("faiss_index", "index.faiss")):
        st.info("Cargando base de conocimientos existente...")
        vector_store = processor.load_vector_store()
        if vector_store is not None:
            retrieval_chain = setup_retrieval_chain(vector_store)
    else:
        st.warning("Procesando documentos legales por primera vez...")
        vector_store = processor.process_documents()
        if vector_store is not None:
            retrieval_chain = setup_retrieval_chain(vector_store)
        else:
            st.error("No se pudo inicializar la base de conocimientos")
            st.stop()

    SYSTEM_PROMPT = """
    Eres EcoPoliciApp, un asistente especializado para oficiales de la Policía Ambiental y de Carabineros en Colombia. 
    Tu objetivo es proporcionar información legal precisa y contextualizada sobre infracciones ambientales, protección de fauna y flora, y regulaciones mineras.

    DIRECTRICES PARA INFORMES:

    I. Clasificación de Infracciones:
    A. Infracciones Ambientales:
       - Contra la fauna (tráfico, caza ilegal, pesca irregular)
       - Contra la flora (deforestación, tráfico de especies)
       - Minería ilegal
       - Contaminación de fuentes hídricas

    B. Comportamientos Regulados:
       - Pesca (tallas mínimas, vedas, artes de pesca)
       - Aprovechamiento forestal
       - Quemas controladas
       - Manejo de emergencias con fauna

    II. Debido Proceso:
    1. Identificación precisa de la infracción ambiental
    2. Protocolo de actuación específico
    3. Derechos del ciudadano
    4. Procedimientos de decomiso y custodia

    III. Formato de Respuesta:

    • Tipo de Infracción: [Ambiental/Minera/Pesquera]

    • Descripción: [Descripción precisa del comportamiento]

    • Base Legal: [Artículo específico (TEXTO COMPLETO)]

    • Autoridad Competente: [AUNAP/Ministerio de Ambiente/CAR]

    • Medidas: [Preventivas/Sancionatorias]

    • Protocolo de Actuación: [Pasos específicos según el tipo]

    • Medidas de Aseguramiento: [Material/Especies/Evidencias]

    • Entidades de Apoyo: [Instituciones que deben ser notificadas]

    ÁREAS ESPECIALIZADAS:

    1. PESCA:
    - Verificación de tallas mínimas según AUNAP
    - Tipos de artes de pesca permitidos
    - Períodos de veda
    - Protocolos de decomiso

    2. FLORA:
    - Cálculo de cubitaje en madera
    - Especies protegidas
    - Permisos de aprovechamiento forestal
    - Protocolos de custodia

    3. MINERÍA:
    - Verificación de títulos mineros
    - Licencias ambientales
    - Minería de subsistencia
    - Protocolos de decomiso

    4. FAUNA:
    - Especies en CITES
    - Protocolos de manejo
    - Centros de atención autorizados
    - Procedimientos de rescate

    INSTRUCCIONES ESPECIALES:
    1. SIEMPRE citar normativa ambiental vigente
    2. Incluir protocolos de cadena de custodia
    3. Especificar autoridades competentes
    4. Detallar medidas de aseguramiento
    5. Indicar centros de recepción autorizados
    6. Proveer información de soporte técnico
    """

    def get_article_text(vector_store, article_reference):
        """
        Busca y retorna el texto completo de un artículo específico con mejor precisión.
        """
        # Realizar una búsqueda más específica incluyendo variaciones comunes
        search_queries = [
            f"Artículo {article_reference}",
            f"ARTÍCULO {article_reference}",
            f"Art. {article_reference}",
            f"Numeral {article_reference}"
        ]
        
        all_results = []
        for query in search_queries:
            similar_docs = vector_store.similarity_search(
                query,
                k=5  # Aumentado para mejor cobertura
            )
            all_results.extend(similar_docs)
        
        # Mejorar la extracción del artículo específico
        for doc in all_results:
            content = doc.page_content
            # Buscar coincidencias exactas con diferentes formatos
            patterns = [
                rf"(?:Artículo|ARTÍCULO|Art\.|Numeral)\s*{article_reference}\b[.\s]+(.*?)(?=(?:Artículo|ARTÍCULO|Art\.|Numeral)\s*\d+|\Z)",
                rf"{article_reference}\.\s+(.*?)(?=\d+\.\s+|\Z)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return None

    def calcular_sancion(detalles_sancion):
        """
        Calcula y detalla sanciones ambientales
        """
        # Valor del SMMLV para 2025 (actualizable)
        SMMLV_2025 = 1423500
        SMDLV_2025 = SMMLV_2025 / 30

        resultado = {
            "sanciones": [],
            "procedimiento_detallado": "",
            "medidas_preventivas": "",
            "autoridades_competentes": []
        }

        # Manejo de sanciones monetarias
        if "monetaria" in detalles_sancion:
            unidad = detalles_sancion["monetaria"].get("unidad", "SMMLV")
            cantidad = detalles_sancion["monetaria"].get("cantidad", 0)
            
            if unidad == "SMMLV":
                valor_total = SMMLV_2025 * cantidad
                resultado["sanciones"].append({
                    "tipo": "Multa Ambiental",
                    "unidad": "SMMLV",
                    "cantidad": cantidad,
                    "valor": f"${valor_total:,.0f}"
                })

        # Medidas preventivas
        resultado["medidas_preventivas"] = """
        Medidas Preventivas Aplicables:
        1. Decomiso preventivo de especímenes o productos
        2. Suspensión de actividad
        3. Amonestación escrita
        4. Registro detallado de evidencias
        """

        # Procedimiento detallado
        resultado["procedimiento_detallado"] = """
        Procedimiento Ambiental:
        1. Identificación de la infracción
        2. Registro fotográfico y documental
        3. Aplicación de medidas preventivas
        4. Cadena de custodia
        5. Notificación a autoridades ambientales
        6. Aseguramiento de pruebas
        """

        # Autoridades competentes
        resultado["autoridades_competentes"] = [
            "CAR Regional",
            "Ministerio de Ambiente",
            "AUNAP",
            "Fiscalía Ambiental",
            "Centro de Atención de Fauna"
        ]

        return resultado

    def display_response(response_text, container):
        """Display the response using Streamlit components with article text."""
        # Simplemente mostrar el texto como markdown
        container.markdown(response_text)

    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container):
            self.container = container
            self.text = ""
            
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token

            display_response(self.text, self.container)

    def extract_table_data(markdown_text):
        """Extract table data from markdown and convert to DataFrame."""
        try:
            # Find table in text using regex
            table_pattern = r'\|.*\|'
            table_rows = re.findall(table_pattern, markdown_text)
            
            if not table_rows:
                return None, None
                
            # Process table rows
            headers = ['Campo', 'Valor']
            data = []
            
            for row in table_rows[2:]:
                values = [col.strip() for col in row.split('|')[1:-1]]
                if len(values) == 2:
                    data.append(values)
            
            df = pd.DataFrame(data, columns=headers)
            
            pre_table = markdown_text.split('|')[0].strip()
            post_table = markdown_text.split('|')[-1].strip()
            other_text = f"{pre_table}\n\n{post_table}".strip()
            
            return df, other_text
        except Exception as e:
            st.error(f"Error procesando la tabla: {str(e)}")
            return None, None

    def search_laws(query):
        """Buscar en los documentos legales vectorizados con más detalles."""
        if vector_store is None:
            st.error("Base de conocimientos no inicializada")
            return None
        
        # Realizar una búsqueda de similitud en el vector store
        similar_docs = vector_store.similarity_search(query, k=5)
        
        # Preparar los resultados
        results = []
        for doc in similar_docs:
            # Extraer información del contexto
            content = doc.page_content
            source = doc.metadata.get('source', 'Documento desconocido')
            page = doc.metadata.get('page', 'N/A')
            
            results.append({
                'Artículo/Código': f"Fuente: {source}, Página: {page}",
                'Contenido Relevante': content[:500] + '...',  # Mostrar un extracto más largo
            })
        
        # Convertir a DataFrame para mostrar resultados
        results_df = pd.DataFrame(results)
        return results_df

    def get_chat_response(prompt, temperature=0.3):
        """Generate chat response using the selected LLM with improved context handling."""
        try:
            response_placeholder = st.empty()
            stream_handler = StreamHandler(response_placeholder)
            
            # Mejorar la búsqueda de contexto
            relevant_context = search_laws(prompt)
            
            # Extraer posibles referencias a artículos del prompt
            article_references = re.findall(r'(?:artículo|numeral)\s+(\d+(?:\.\d+)?)', prompt.lower())
            
            # Obtener el texto completo de los artículos mencionados
            article_texts = []
            if article_references:
                for ref in article_references:
                    article_text = get_article_text(vector_store, ref)
                    if article_text:
                        article_texts.append(f"Artículo {ref}: {article_text}")
            
            # Crear un prompt enriquecido con el contexto y los artículos específicos
            enhanced_prompt = f"""
            Consulta: {prompt}
            
            Artículos específicos mencionados:
            {'\n'.join(article_texts) if article_texts else 'No se mencionaron artículos específicos'}
            
            Contexto relevante adicional:
            {relevant_context.to_string() if not relevant_context.empty else 'No se encontró contexto adicional'}
            
            Por favor, proporciona una respuesta detallada basada en este contexto y las normas aplicables.
            """
            
            chat_model = ChatOpenAI(
                model="gpt-4o",
                temperature=temperature,
                api_key=API_KEY,
                streaming=True,
                callbacks=[stream_handler]
            )
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=enhanced_prompt)
            ]
            
            if "messages" in st.session_state:
                for msg in st.session_state.messages[-3:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(SystemMessage(content=msg["content"]))
            
            response = chat_model.invoke(messages)
            return stream_handler.text
            
        except Exception as e:
            st.error(f"Error generando respuesta: {str(e)}")
            return "Lo siento, ocurrió un error al procesar su solicitud."

    def ensure_directory_exists():
        """Ensure necessary directories exist."""
        directories = ["data", "faiss_index"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def check_startup_health():
        """Verify all components are ready."""
        try:
            # Check API key
            if not API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            
            # Check directories
            ensure_directory_exists()
            
            # Check vector store
            processor = LawDocumentProcessor()  # Crear una instancia
            vector_store = processor.load_vector_store()  # Llamar al método en la instancia
            if vector_store is None:
                st.warning("Initializing knowledge base...")
                vector_store = processor.process_documents()
            
            return True
        except Exception as e:
            st.error(f"Startup health check failed: {str(e)}")
            return False

    # Add this at the start of your main function
    if not check_startup_health():
        st.stop()

    st.write(logo, unsafe_allow_html=True)
    st.title("EcoPoliciApp", anchor=False)
    st.markdown("**Asistente virtual para procedimientos ambientales y protección de recursos naturales**")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("""
        **Bienvenido al Sistema de Consulta de Infracciones Ambientales**
        """)
        
        # Búsqueda en base de datos
        search_query = st.text_input("Buscar en base de datos:")
        if search_query:
            results = search_laws(search_query)
            if not results.empty:
                st.dataframe(results)
            else:
                st.info("No se encontraron resultados")

        if st.button("Borrar Historial"):
            st.session_state.messages = []
            st.experimental_rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and '|' in message["content"]:
                display_response(message["content"], st)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Describe la situación..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👮"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            is_multa = prompt.upper().startswith("MULTA:")
            if is_multa:
                multa_content = prompt[6:].strip()
                response = get_chat_response(multa_content)
            else:
                response = get_chat_response(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()