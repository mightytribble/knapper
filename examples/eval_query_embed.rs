//! Offline eval harness: embed the calibration pool's queries through both
//! halves of the asymmetric pair — the query template (what `search` runs) and
//! the document template with an empty title (what a chunk with
//! `document_title = none` is embedded as). The document half is the
//! self-similarity ceiling for the anchored-cosine eval.
//!
//! Usage: KNAPPER_HOME=<arm> eval_query_embed <queries.json>
//! where queries.json is `[{"id": "...", "query": "..."}]`.
//! Emits `[{"id", "query", "q": [f32...], "d": [f32...]}]` on stdout.

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: eval_query_embed <queries.json>");
    let raw = std::fs::read_to_string(&path)?;
    let queries: Vec<serde_json::Value> = serde_json::from_str(&raw)?;

    let config = knapper::config::Config::load()?;
    let data_dir = knapper::config::Config::data_dir()?;
    let models_dir = data_dir.join("models");
    let mut embedder = knapper::llm::load_embedder(&models_dir, &config)?;

    let mut out = Vec::new();
    for q in &queries {
        let id = q["id"].as_str().expect("id");
        let text = q["query"].as_str().expect("query");
        let qv = embedder.embed_query(text)?;
        let dv = embedder
            .embed_batch(&[knapper::llm::EmbedDoc::untitled(text)])?
            .pop()
            .expect("one document vector");
        out.push(serde_json::json!({"id": id, "query": text, "q": qv, "d": dv}));
        eprintln!("embedded {id}");
    }
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
