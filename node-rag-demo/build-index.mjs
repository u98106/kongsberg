// file: build-index.mjs
import 'dotenv/config';
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
//import { RecursiveCharacterTextSplitter } from "langchain";
import { ChromaClient } from "chromadb";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_API_BASE = process.env.OPENAI_API_BASE;

async function main() {
  // 1. Sample docs (replace with loaders later)
  const rawDocs = [
    "Agentic AI tools can orchestrate multiple skills autonomously.",
    "RAG uses a retriever over a vector store plus an LLM generator.",
    "ChromaDB supports persistent and in‑memory vector collections.",
  ];

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,
    chunkOverlap: 50,
  });

  const splitDocs = [];
  for (const text of rawDocs) {
    const chunks = await textSplitter.splitText(text);
    for (const chunk of chunks) {
      splitDocs.push(chunk);
    }
  }

  // 2. Embeddings via LangChain
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    openAIApiKey: OPENAI_API_KEY,
    configuration: {
      baseURL: OPENAI_API_BASE,
    },
  });

  const vectors = await embeddings.embedDocuments(splitDocs);

  // 3. Chroma client (using default in‑memory server; for prod, point to real host)
  //const client = new ChromaClient({
  //  path: "http://localhost:8000", // ensure chroma server is running here
  //});

  const client = new ChromaClient();

  const collection = await client.getOrCreateCollection({
    name: "agentic-ai-notes",
    metadata: { description: "Agentic AI training snippets" },
  });

  const ids = splitDocs.map((_, i) => `doc-${i}`);
  await collection.add({
    ids,
    documents: splitDocs,
    embeddings: vectors,
  });

  console.log("Chroma collection 'agentic-ai-notes' populated.");
}

main().catch(console.error);
