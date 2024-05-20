# Rust LLM RAG Sample

This is the Rust LLM RAG sample implement use `langchain_rust` `Qdrant` and `qctimer_macros`. Use QChoice Tech. LTD, company infomation to demo.

### Use

- Run qdrant docker

  ```shell
  docker run -p 6334:6334 -p 6333:6333 --name qdrant_server --rm  qdrant/qdrant:v1.9.1
  ```

- Clone the repo

  ```shell
  git clone https://github.com/li195111/rag_sample_rs.git
  cd rag_sample_rs
  ```
  
- Run `cargo run` in console.
  
  ```Shell
  cargo r -j -8
  ```


### TODO

- [ ] `faiss-rs` vector database available.
- [ ] `PostgreSQL` vector database available.
- [ ] `Milvus` vector database available.
- [ ] OpenAI API available.
- [ ] AzureOpenai API available.
- [ ] Claude API available.
