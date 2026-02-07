// file: query-rag.mjs
import 'dotenv/config';
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
//import { ChromaClient } from "@chroma-core/chromadb";
import { ChromaClient } from "chromadb";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_API_BASE = process.env.OPENAI_API_BASE;

async function ragQuery(query) {
  // 1. Connect to Chroma and collection
  const client = new ChromaClient({
    path: "http://localhost:8000",
  });

  const collection = await client.getOrCreateCollection({
    name: "agentic-ai-notes",
  });

  // 2. Embed query
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    openAIApiKey: OPENAI_API_KEY,
    configuration: {
      baseURL: OPENAI_API_BASE,
    },
  });

  const [queryVector] = await embeddings.embedDocuments([query]);

  // 3. Retrieve top‑k docs
  const k = 4;
  const results = await collection.query({
    queryEmbeddings: [queryVector],
    nResults: k,
  });

  const retrievedDocs = results.documents[0] || [];

  // 4. Build prompt for LLM
  const context = retrievedDocs.map((d, i) => `Source ${i + 1}: ${d}`).join("\n\n");

  const prompt = [
    {
      role: "system",
      content:
        "You are a concise assistant for an agentic AI training module. " +
        "Use the provided context to answer the question. If unsure, say you are not sure.",
    },
    {
      role: "user",
      content: `Context:\n${context}\n\nQuestion: ${query}`,
    },
  ];

  const llm = new ChatOpenAI({
    modelName: "gpt-4.1",
    temperature: 0.2,
    openAIApiKey: OPENAI_API_KEY,
    configuration: {
      baseURL: OPENAI_API_BASE,
    },
  });

  const response = await llm.invoke(prompt);
  return { answer: response.content, retrievedDocs };
}

async function main() {
  const query = "Explain RAG for a Node.js backend engineer.";
  const { answer, retrievedDocs } = await ragQuery(query);

  console.log("\nQ:", query);
  console.log("\nA:", answer);
  console.log("\nRetrieved snippets:");
  retrievedDocs.forEach((d, i) => {
    console.log(`- [${i + 1}] ${d.slice(0, 120)}...`);
  });
}

main().catch(console.error);
