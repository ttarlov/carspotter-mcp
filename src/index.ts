import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";
import Anthropic from "@anthropic-ai/sdk";
import { createClient } from "@supabase/supabase-js";
import crypto from "crypto";

// ── ENV ──────────────────────────────────────────────────
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;
const PORT = parseInt(process.env.PORT || "3000", 10);

if (!ANTHROPIC_API_KEY || !SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("Missing required env vars: ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY");
  process.exit(1);
}

const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// ── HELPERS ──────────────────────────────────────────────
async function sha256(str: string): Promise<string> {
  const hash = crypto.createHash("sha256").update(str).digest("hex");
  return hash;
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const AUTO_COLORS = [
  "#dc2626", "#ea580c", "#d97706", "#65a30d", "#059669",
  "#0891b2", "#2563eb", "#7c3aed", "#c026d3", "#e11d48",
];

// ── ANALYZE CAR IMAGE ────────────────────────────────────
async function analyzeCar(imageBase64: string, mediaType: string, hintMake?: string, hintModel?: string) {
  console.log(`[ANALYZE] Sending to Claude... (hint: ${hintMake} ${hintModel})`);

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1000,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: mediaType as any, data: imageBase64 },
          },
          {
            type: "text",
            text: `You are analyzing a car photo for a car identification quiz game.

${hintMake || hintModel ? `The user suggests this is a: ${hintMake || ""} ${hintModel || ""}` : "Please identify this car."}

Respond with ONLY a JSON object (no markdown, no backticks):

{
  "make": "manufacturer",
  "model": "model name",
  "year_era": "approximate era like 1960s, 1990s, 2020s",
  "body_style": "coupe, sedan, convertible, SUV, truck, etc",
  "class": "muscle, sports, luxury, economy, etc",
  "origin": "American, Japanese, German, etc",
  "hints": [
    { "focus": [x_pct, y_pct], "zoom": zoom_level, "description": "detail" },
    { "focus": [x_pct, y_pct], "zoom": zoom_level, "description": "detail" },
    { "focus": [x_pct, y_pct], "zoom": zoom_level, "description": "detail" }
  ],
  "distractors": [
    { "make": "Wrong1", "model": "Model1" },
    { "make": "Wrong2", "model": "Model2" },
    { "make": "Wrong3", "model": "Model3" }
  ]
}

HINTS: Hint 1 zoom 6-10x on distinctive detail, Hint 2 zoom 3-5x partial view, Hint 3 zoom 1.3-2x full car.
DISTRACTORS: Similar era/style/class cars. Different makes from the correct answer. Real cars only.`,
          },
        ],
      },
    ],
  });

  const text = response.content
    .map((block) => (block.type === "text" ? block.text : ""))
    .filter(Boolean)
    .join("\n");

  const cleaned = text.replace(/```json|```/g, "").trim();
  return JSON.parse(cleaned);
}

// ── UPLOAD TO SUPABASE ───────────────────────────────────
async function uploadCar(imageBase64: string, analysis: any) {
  const { make, model, hints, distractors } = analysis;
  console.log(`[UPLOAD] ${make} ${model}`);

  // Upload image to storage
  const buffer = Buffer.from(imageBase64, "base64");
  const fileName = `cars/${Date.now()}-${Math.random().toString(36).slice(2)}.jpg`;

  const { error: uploadError } = await supabase.storage
    .from("car-images")
    .upload(fileName, buffer, { contentType: "image/jpeg", upsert: false });

  if (uploadError) throw new Error(`Storage upload failed: ${uploadError.message}`);

  const { data: urlData } = supabase.storage.from("car-images").getPublicUrl(fileName);
  const imageUrl = urlData.publicUrl;

  // Build car record
  const hintData = hints.map((h: any) => ({ focus: h.focus, zoom: h.zoom }));
  const correct = { make, model };
  const wrongAnswers = distractors.map((d: any) => ({ make: d.make, model: d.model }));
  const options = shuffle([correct, ...wrongAnswers]);
  const color = AUTO_COLORS[Math.floor(Math.random() * AUTO_COLORS.length)];
  const answerHash = await sha256(`${make}|${model}`);
  const makeHash = await sha256(make);

  const { data: car, error: insertError } = await supabase
    .from("cars")
    .insert({
      make,
      model,
      feature: "Detail",
      hints: hintData,
      color,
      options,
      image_url: imageUrl,
      answer_hash: answerHash,
      make_hash: makeHash,
      active: true,
    })
    .select()
    .single();

  if (insertError) throw new Error(`DB insert failed: ${insertError.message}`);

  return { id: car.id, make, model, imageUrl };
}

// ── MCP SERVER ───────────────────────────────────────────
const server = new McpServer({
  name: "carspotter",
  version: "1.0.0",
});

server.tool(
  "add_car",
  "Add a car photo to the CarSpotter quiz game. Analyzes the image with AI to identify the car, generates smart quiz hints and plausible wrong answers, then uploads everything to the game database. The car becomes playable immediately.",
  {
    image_base64: z.string().describe("Base64-encoded car image (JPEG or PNG)"),
    media_type: z.string().default("image/jpeg").describe("Image MIME type (image/jpeg or image/png)"),
    make: z.string().optional().describe("Optional: car manufacturer hint (e.g. 'Ford')"),
    model: z.string().optional().describe("Optional: car model hint (e.g. 'Mustang GT')"),
  },
  async ({ image_base64, media_type, make, model }) => {
    try {
      console.log(`[ADD_CAR] Starting... (${(image_base64.length / 1024).toFixed(0)}KB base64)`);

      // Step 1: Analyze with Claude
      const analysis = await analyzeCar(image_base64, media_type, make, model);
      console.log(`[ADD_CAR] Identified: ${analysis.make} ${analysis.model} (${analysis.year_era})`);

      // Step 2: Upload to Supabase
      const result = await uploadCar(image_base64, analysis);
      console.log(`[ADD_CAR] ✅ Uploaded: ${result.make} ${result.model} (ID: ${result.id})`);

      const hintDescs = analysis.hints
        .map((h: any, i: number) => `Hint ${i + 1}: ${h.description} (${h.zoom}×)`)
        .join("\n");
      const distractorList = analysis.distractors
        .map((d: any) => `${d.make} ${d.model}`)
        .join(", ");

      return {
        content: [
          {
            type: "text" as const,
            text: `✅ Added ${analysis.make} ${analysis.model} to CarSpotter!

🏷️ ${analysis.year_era} ${analysis.body_style} (${analysis.class}, ${analysis.origin})

🔍 Hints:
${hintDescs}

❌ Wrong answers: ${distractorList}

🔗 Image: ${result.imageUrl}

The car is now live in the game!`,
          },
        ],
      };
    } catch (err: any) {
      console.error(`[ADD_CAR] ❌ Error:`, err.message);
      return {
        content: [{ type: "text" as const, text: `❌ Failed to add car: ${err.message}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  "list_cars",
  "List all cars currently in the CarSpotter game database.",
  {},
  async () => {
    try {
      const { data: cars, error } = await supabase
        .from("cars")
        .select("id, make, model, image_url, active")
        .order("created_at", { ascending: false });

      if (error) throw new Error(error.message);

      const list = (cars || [])
        .map((c: any) => `${c.active ? "✅" : "⏸️"} ${c.make} ${c.model}`)
        .join("\n");

      return {
        content: [
          {
            type: "text" as const,
            text: `🚗 CarSpotter Garage (${cars?.length || 0} cars):\n\n${list || "No cars yet!"}`,
          },
        ],
      };
    } catch (err: any) {
      return {
        content: [{ type: "text" as const, text: `❌ Error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  "remove_car",
  "Remove a car from CarSpotter by make and model name.",
  {
    make: z.string().describe("Car manufacturer (e.g. 'Ford')"),
    model: z.string().describe("Car model (e.g. 'Mustang GT')"),
  },
  async ({ make, model }) => {
    try {
      const { data: cars, error: findError } = await supabase
        .from("cars")
        .select("id, image_url")
        .ilike("make", make)
        .ilike("model", model);

      if (findError) throw new Error(findError.message);
      if (!cars || cars.length === 0) {
        return {
          content: [{ type: "text" as const, text: `⚠️ No car found matching "${make} ${model}"` }],
        };
      }

      for (const car of cars) {
        if (car.image_url) {
          const path = car.image_url.split("/car-images/")[1];
          if (path) await supabase.storage.from("car-images").remove([path]);
        }
        await supabase.from("cars").delete().eq("id", car.id);
      }

      return {
        content: [
          {
            type: "text" as const,
            text: `🗑️ Removed ${cars.length} car(s) matching "${make} ${model}" from CarSpotter.`,
          },
        ],
      };
    } catch (err: any) {
      return {
        content: [{ type: "text" as const, text: `❌ Error: ${err.message}` }],
        isError: true,
      };
    }
  }
);

// ── EXPRESS + TRANSPORT ──────────────────────────────────
const app = express();

// Store transports by session ID for multi-turn
const transports = new Map<string, StreamableHTTPServerTransport>();

app.post("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;

  let transport: StreamableHTTPServerTransport;

  if (sessionId && transports.has(sessionId)) {
    transport = transports.get(sessionId)!;
  } else {
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => crypto.randomUUID(),
      onsessioninitialized: (id) => {
        transports.set(id, transport);
        console.log(`[MCP] Session created: ${id}`);
      },
    });

    transport.onclose = () => {
      const id = [...transports.entries()].find(([, t]) => t === transport)?.[0];
      if (id) {
        transports.delete(id);
        console.log(`[MCP] Session closed: ${id}`);
      }
    };

    await server.connect(transport);
  }

  await transport.handleRequest(req, res, req.body);
});

// Handle GET for SSE (backwards compat)
app.get("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;
  if (sessionId && transports.has(sessionId)) {
    const transport = transports.get(sessionId)!;
    await transport.handleRequest(req, res);
  } else {
    res.status(400).json({ error: "No session. Send POST first." });
  }
});

// HEAD for protocol version discovery
app.head("/mcp", (_req, res) => {
  res.setHeader("MCP-Protocol-Version", "2025-06-18");
  res.status(200).end();
});

// Health check
app.get("/", (_req, res) => {
  res.json({
    name: "carspotter-mcp",
    version: "1.0.0",
    status: "running",
    tools: ["add_car", "list_cars", "remove_car"],
  });
});

app.listen(PORT, () => {
  console.log(`\n🚗 CarSpotter MCP Server running on port ${PORT}`);
  console.log(`   MCP endpoint: http://localhost:${PORT}/mcp`);
  console.log(`   Health check: http://localhost:${PORT}/\n`);
});
