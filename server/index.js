require('dotenv').config();
const express = require('express');
const { MilvusClient, DataType } = require('@zilliz/milvus2-sdk-node');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const client = new MilvusClient({ 
    address: process.env.ZILLIZ_URI, 
    token: process.env.ZILLIZ_TOKEN,
    timeout: 60000 
});

const collectionName = "ai_smart_memory";

// ====================== EMBEDDING ======================
async function getEmbedding(text) {
    await new Promise(r => setTimeout(r, 800));
    const model = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
    const result = await model.embedContent(text);
    return result.embedding.values;
}

// ====================== SETUP (id manual) ======================
async function setupCollection() {
    try {
        console.log("⏳ Connecting to Milvus...");
        await client.checkHealth();
        console.log("✅ Connected to Milvus");

        const exists = await client.hasCollection({ collection_name: collectionName });
        if (exists.value) {
            await client.dropCollection({ collection_name: collectionName });
        }

        await client.createCollection({
            collection_name: collectionName,
            fields: [
                { 
                    name: "id", 
                    data_type: DataType.Int64, 
                    is_primary_key: true, 
                    auto_id: false   // ← Важно: manual id
                },
                { name: "vector", data_type: DataType.FloatVector, dim: 3072 },
                { name: "text", data_type: DataType.VarChar, max_length: 2000 }
            ]
        });

        await client.createIndex({
            collection_name: collectionName,
            field_name: "vector",
            index_type: "IVF_FLAT",
            metric_type: "COSINE",
            params: { nlist: 64 }
        });

        await client.loadCollectionSync({ collection_name: collectionName });
        console.log(`🚀 Collection "${collectionName}" is ready!`);

    } catch (err) {
        console.error("❌ SETUP ERROR:", err.message);
        process.exit(1);
    }
}

// ====================== STORE ======================
let currentId = 1;   // Simple counter

app.post('/api/store', async (req, res) => {
    const { text } = req.body;
    if (!text?.trim()) return res.status(400).json({ error: "Text is required." });

    try {
        const embedding = await getEmbedding(text);

        const insertResult = await client.insert({
            collection_name: collectionName,
            data: [{
                id: currentId++,           // Manual ID
                vector: embedding,
                text: text.trim()
            }]
        });

        console.log("📊 Insert Result:", JSON.stringify(insertResult, null, 2));

        await client.flushSync({ collection_names: [collectionName] });
        await client.loadCollectionSync({ collection_name: collectionName });

        console.log(`✅ Stored (ID: ${currentId-1}): "${text.substring(0, 60)}..."`);
        res.json({ success: true, message: "Knowledge stored successfully." });

    } catch (err) {
        console.error("❌ STORE ERROR:", err.message);
        res.status(500).json({ error: err.message });
    }
});

// ====================== SEARCH ======================
app.post('/api/search', async (req, res) => {
    const { query } = req.body;
    if (!query?.trim()) return res.status(400).json({ error: "Query is required." });

    try {
        console.log(`🔎 Searching: "${query}"`);

        const queryVector = await getEmbedding(query);

        const searchRes = await client.search({
            collection_name: collectionName,
            vector: queryVector,
            limit: 5,
            output_fields: ["text"],
            params: { nprobe: 16 }
        });

        const results = searchRes.results || [];
        console.log(`✅ Found ${results.length} results`);

        const formatted = results.map(r => ({
            text: r.text || (r.entity && r.entity.text),
            score: r.score ? (r.score * 100).toFixed(1) + "%" : null
        }));

        res.json({ success: true, results: formatted });

    } catch (err) {
        console.error("❌ SEARCH ERROR:", err.message);
        res.status(500).json({ error: err.message });
    }
});

// Debug
app.get('/api/all', async (req, res) => {
    try {
        const result = await client.query({
            collection_name: collectionName,
            output_fields: ["id", "text"],
            limit: 50
        });
        res.json({ count: result.length || 0, items: result });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/health', (req, res) => res.json({ status: "ok" }));

setupCollection().then(() => {
    app.listen(5000, () => {
        console.log("✨ AI Smart Memory Server running on http://localhost:5000");
    });
});