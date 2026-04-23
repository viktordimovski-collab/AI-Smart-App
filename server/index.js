require('dotenv').config();
const express = require('express');
const { MilvusClient, DataType } = require('@zilliz/milvus2-sdk-node');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Check for API Key
if (!process.env.GEMINI_API_KEY) {
    console.error("❌ ERROR: GEMINI_API_KEY is not set in the .env file!");
    process.exit(1);
}

// Initialize clients
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const client = new MilvusClient({ 
    address: process.env.ZILLIZ_URI, 
    token: process.env.ZILLIZ_TOKEN 
});

const collectionName = "my_ai_collection";

async function setupCollection() {
    try {
        const res = await client.hasCollection({ collection_name: collectionName });
        if (!res.value) {
            await client.createCollection({
                collection_name: collectionName,
                fields: [
                    { name: "id", data_type: DataType.Int64, is_primary_key: true, auto_id: true },
                    { name: "vector", data_type: DataType.FloatVector, dim: 768 },
                    { name: "text", data_type: DataType.VarChar, max_length: 500 }
                ]
            });
            await client.createIndex({
                collection_name: collectionName,
                field_name: "vector",
                index_type: "IVF_FLAT",
                metric_type: "L2",
                params: { nlist: 1024 }
            });
            console.log("✅ Database is ready!");
        }
    } catch (err) {
        console.error("Error setting up Milvus:", err.message);
    }
}

app.post('/api/store', async (req, res) => {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "No text provided" });

    try {
        const model = genAI.getGenerativeModel({ model: "embedding-001" });
        const result = await model.embedContent(text);
        
        if (!result.embedding || !result.embedding.values) {
            throw new Error("Google did not return embedding values.");
        }

        await client.insert({
            collection_name: collectionName,
            fields_data: [{
                vector: result.embedding.values,
                text: text
            }]
        });

        res.json({ success: true, message: "Saved to Milvus!" });
    } catch (err) {
        console.error("Error saving to database:", err.message);
        res.status(500).json({ error: err.message });
    }
});

async function initServer() {
    await setupCollection();
    const PORT = 5000;
    app.listen(PORT, () => {
        console.log(`🚀 Server running at http://localhost:${PORT}`);
    });
}

initServer();