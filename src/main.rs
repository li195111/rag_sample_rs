use futures::{stream::StreamExt, Stream};

use langchain_rust::{
    chain::{Chain, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder},
    document_loaders::{lo_loader::LoPdfLoader, Loader, LoaderError, TextLoader},
    embedding::ollama::ollama_embedder::OllamaEmbedder,
    fmt_message, fmt_template,
    llm::{openai::OpenAI, OpenAIConfig},
    memory::SimpleMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::{Document, Message},
    template_jinja2,
    text_splitter::TokenSplitter,
    vectorstore::{
        qdrant::{QdrantClient, StoreBuilder},
        Retriever, VecStoreOptions, VectorStore,
    },
};
use std::vec;
use std::{fs, pin::Pin, result::Result};

use qctimer_macros::async_timer;

extern crate chrono;

#[derive(Clone)]
enum Model {
    Ollama,
    // OpenaiChatgpt3_5,
    // OpenaiChatgpt4,
    // OpenaiChatgpt4o,
    // AzureOpenai,
    // Claude,
}

#[derive(Clone)]
enum VectorDB {
    Qdrant,
    // OpenSearch,
    // PostgreSQL,
    // SQLite,
    // SurrealDB,
    // Milvus,
    // Faiss,
}

#[async_timer]
async fn setup_vector_db(vector_db: VectorDB) -> Result<QdrantClient, String> {
    match vector_db {
        VectorDB::Qdrant => {
            let client = QdrantClient::from_url("http://localhost:6334")
                .build()
                .unwrap();

            Ok(client)
        }
    }
}

#[async_timer]
async fn setup_embed(model: Model) -> Result<OllamaEmbedder, String> {
    match model {
        Model::Ollama => {
            let embed_llm = OllamaEmbedder::default().with_model("llama3:latest");
            Ok(embed_llm)
        }
    }
}

#[async_timer]
async fn setup(model: Model) -> Result<OpenAI<OpenAIConfig>, String> {
    match model {
        Model::Ollama => {
            let llm = OpenAI::default()
                .with_config(
                    OpenAIConfig::default()
                        .with_api_base("http://localhost:11434/v1")
                        .with_api_key("ollama"),
                )
                .with_model("llama3:latest");

            Ok(llm)
        }
    }
}

async fn load_doc(path: &str, splitter: TokenSplitter) -> Result<Document, LoaderError> {
    let mut doc: Document = Document::default();
    match fs::read_to_string(path) {
        Ok(content) => {
            let loader = TextLoader::new(content.to_string());
            let mut documents: Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send>> =
                loader.load_and_split(splitter).await.unwrap();
            while let Some(result_doc) = documents.next().await {
                doc = result_doc.unwrap();
            }

            Ok(doc)
        }
        Err(e) => Err(LoaderError::IOError(e)),
    }
}

#[async_timer]
async fn load_pdf(path: &str, splitter: TokenSplitter) -> Vec<Document>{
    let loader = LoPdfLoader::from_path(path).expect("Failed to load pdf");
    let docs = loader
    .load_and_split(splitter)
    .await
    .unwrap()
    .map(|d| d.unwrap())
    .collect::<Vec<_>>()
    .await;

    docs
}

#[async_timer]
async fn chat_stream(msg: &str, chain: &ConversationalRetrieverChain) {
    println!();

    println!("You > {}", msg);

    let input_variables = prompt_args! {
        "question" => msg,
    };

    //If you want to stream
    print!("Bot > ");
    let mut stream = chain.stream(input_variables).await.unwrap();
    while let Some(result) = stream.next().await {
        match result {
            Ok(data) => {
                data.to_stdout().unwrap();
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let model = Model::Ollama;
    let vec_db = VectorDB::Qdrant;
    let collection_name = "langchain-rs";
    let path = "./src/documents/test_data/test_document.txt";
    // let pdf_path = "./src/documents/test_data/Standards-of-Income_112.pdf";
    let pdf_path = "./src/documents/test_data/國泰證券月對帳單.pdf";

    let llm = setup(model.clone()).await.unwrap();

    let embed_llm = setup_embed(model.clone()).await.unwrap();

    let vector_db = setup_vector_db(vec_db.clone()).await.unwrap();

    if vector_db.collection_exists(collection_name).await.unwrap() {
        let del_res = vector_db.delete_collection(collection_name).await.unwrap();
        println!(
            "Remove Exists Collection: {:?} {:?}",
            collection_name, del_res
        );
    }

    let store = StoreBuilder::new()
        .embedder(embed_llm)
        .client(vector_db)
        .collection_name("langchain-rs")
        .build()
        .await
        .unwrap();

    let splitter: TokenSplitter = TokenSplitter::default();
    let doc: Document = load_doc(path, splitter.clone()).await.unwrap();
    let mut docs: Vec<Document> = load_pdf(pdf_path, splitter.clone()).await;
    println!("Document: {:?}", docs);
    docs.push(doc);

    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();

    let prompt= message_formatter![
            fmt_message!(Message::new_system_message("You are a helpful assistant")),
            fmt_template!(HumanMessagePromptTemplate::new(
            template_jinja2!("Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Keep the answers clean, short and to the point.
            {{context}} Question:{{question}} Answer:", "context","question")))];

    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(llm)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(Retriever::new(store, 5))
        //If you want to sue the default prompt remove the .prompt()
        //Keep in mind if you want to change the prmpt; this chain need the {{context}} variable
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    // let input_variables = prompt_args! {
    //     "question" => "Hi",
    // };

    // let result = chain.invoke(input_variables).await;
    // if let Ok(result) = result {
    //     println!("Result: {:?}", result);
    // }

    chat_stream("酷喬伊科技設立時間?", &chain).await;
    chat_stream("淨收付金額是多少呢?", &chain).await;
}
