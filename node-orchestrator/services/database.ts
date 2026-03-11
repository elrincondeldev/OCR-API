import { MongoClient, Collection, Document } from "mongodb";

let client: MongoClient | null = null;

export async function connectDb(): Promise<void> {
  const uri = process.env.MONGODB_URI!;
  client = new MongoClient(uri);
  await client.connect();
  await client.db("admin").command({ ping: 1 });
  console.info("Connected to MongoDB.");
}

export async function closeDb(): Promise<void> {
  if (client) {
    await client.close();
    console.info("MongoDB connection closed.");
  }
}

export function getCollection(): Collection<Document> {
  if (!client)
    throw new Error("Database client not initialised. Call connectDb() first.");
  return client
    .db(process.env.MONGODB_DB_NAME!)
    .collection(process.env.MONGODB_COLLECTION!);
}
