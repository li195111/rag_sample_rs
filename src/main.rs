use futures::{stream::StreamExt, Future, Stream};

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
use serde::{Deserialize, Serialize};
use serde_json;
use std::vec;
use std::{fs, pin::Pin, result::Result};

use qctimer_macros::async_timer;

extern crate chrono;
#[derive(Serialize, Deserialize, Debug, Clone)]
struct CodeRate {
    code: String,
    name: String,
    pure_profit_rate: String,
    income_standard: String,
    profit_rate: String,
    cost_rate: String,
    net_profit_rate: String,
}

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

#[async_timer]
async fn load_doc(
    path: &str,
    splitter: Option<&TokenSplitter>,
) -> Result<Vec<Document>, LoaderError> {
    let documents: Pin<
        Box<
            dyn Future<
                    Output = Result<
                        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send>>,
                        LoaderError,
                    >,
                > + Send,
        >,
    >;
    match fs::read_to_string(path) {
        Ok(content) => {
            let loader = TextLoader::new(content.to_string());
            match splitter {
                Some(splitter) => {
                    documents = loader.load_and_split(splitter.clone());
                }
                None => {
                    documents = loader.load();
                }
            }
            Ok(documents
                .await
                .unwrap()
                .map(|d| d.unwrap())
                .collect::<Vec<_>>()
                .await)
        }
        Err(e) => Err(LoaderError::IOError(e)),
    }
}

#[async_timer]
async fn load_pdf(path: &str, splitter: TokenSplitter) -> Vec<Document> {
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
async fn load_code_rate_json(path: &str) -> Result<Vec<CodeRate>, LoaderError> {
    match fs::read_to_string(path) {
        Ok(content) => {
            let code_rate_list = serde_json::from_str(&content).unwrap();
            Ok(code_rate_list)
        }
        Err(e) => Err(LoaderError::IOError(e)),
    }
}

#[async_timer]
async fn chat_stream(msg: &str, chain: &ConversationalRetrieverChain, tools: &str) {
    println!();

    println!("You > {}", msg);

    let input_variables = prompt_args! {
        "question" => msg,
        "tools" => tools,
    };

    //If you want to stream
    print!("Bot > ");
    let mut stream: Pin<
        Box<
            dyn Stream<
                    Item = Result<
                        langchain_rust::schemas::StreamData,
                        langchain_rust::chain::ChainError,
                    >,
                > + Send,
        >,
    > = chain.stream(input_variables).await.unwrap();
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
    // let pdf_path = "./src/documents/test_data/國泰證券月對帳單.pdf";
    let pdf_path =
        "./src/documents/test_data/112年度營利事業各業擴大書審純益率、所得額及同業利潤標準.json";

    let tool_path = "./src/tools/find.json";
    let tool_str = fs::read_to_string(tool_path).unwrap();
    println!("tools: {:?}", tool_str);

    // let splitter: TokenSplitter = TokenSplitter::default();
    let mut docs: Vec<Document> = Vec::<Document>::new();
    println!("Number of Documents: {:?}", docs.len());

    let docs_1: Vec<Document> = load_doc(&path, None).await.unwrap();
    docs.extend(docs_1);
    println!("Number of Documents: {:?}", docs.len());

    // let docs_2: Vec<Document> = load_doc(&pdf_path, Some(&splitter)).await.unwrap();
    let json_docs = load_code_rate_json(&pdf_path).await.unwrap();
    let num_json = json_docs.len();

    let chunk_size = 10;
    let mut docs_2 = Vec::<Document>::new();
    for i in 0..(num_json / chunk_size + 1) as usize {
        let s_idx = i * chunk_size;
        let mut e_idx: usize = i * chunk_size + chunk_size;
        if e_idx >= num_json {
            e_idx = num_json;
        }
        let chunk = &json_docs[s_idx..e_idx];
        let doc = Document::new(serde_json::to_string(&chunk).unwrap());
        docs_2.push(doc);
    }
    println!("Number of Json: {:?}", num_json);
    println!("Number of Chunk Documents: {:?}", docs_2.len());
    // for code_rate in docs_2 {
    //     println!("Doc: {:?}", code_rate.page_content);
    // }
    docs.extend(docs_2);
    println!("Number of Documents: {:?}", docs.len());
    // let mut docs: Vec<Document> = load_pdf(pdf_path, splitter.clone()).await;
    // println!("Number of Document: {:?}", docs.len());

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

    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();

    // Ask for user input
    print!("Query> ");
    // std::io::stdout().flush().unwrap();
    let query = "給我name='未分類其他紡織品製造',的'code'與'pure_profit_rate'的值";
    // std::io::stdin().read_line(&mut query).unwrap();

    let results = store
        .similarity_search(&query, 2, &VecStoreOptions::default())
        .await
        .unwrap();

    println!("Number of Results: {:?}", results.len());
    if results.is_empty() {
        println!("No results found.");
        return;
    } else {
        results.iter().for_each(|r| {
            println!("Document: {}", r.page_content);
        });
    }
    let llm = setup(model.clone()).await.unwrap();

    let prompt= message_formatter![
            fmt_message!(Message::new_system_message("You are a helpful assistant")),
            fmt_template!(HumanMessagePromptTemplate::new(
            template_jinja2!("Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep the answers clean, short and to the point. {{context}}, Question:{{question}} Answer:", "context","question")))];

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

    // chat_stream("酷喬伊科技設立時間?", &chain).await;
    chat_stream(
        "給我`name`='未分類其他紡織品製造',的`code`與`pure_profit_rate`的值",
        &chain,
        tool_str.as_str(),
    )
    .await;
}
