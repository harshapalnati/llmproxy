use std::{
    collections::HashMap,
    env,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use futures_util::TryStreamExt;
use jsonschema::JSONSchema;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const TOOL_SYSTEM_PROMPT: &str = r#"
### TOOL USE INSTRUCTIONS
You have access to the following tools:
{tool_definitions}

To call a tool, you MUST strictly follow this format:
1. Wrap your JSON inside <tool_code> tags.
2. The JSON must contain "name" and "arguments".

Example:
<tool_code>
{{
  "name": "get_stock_price",
  "arguments": {{ "symbol": "AAPL" }}
}}
</tool_code>
"#;

#[derive(Clone)]
struct AppState {
    positron_url: String,
    positron_key: String,
    max_retries: usize,
    http: Client,
    tool_regex: Regex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tool {
    #[serde(rename = "type")]
    type_field: String,
    function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolFunction {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    type_field: String,
    function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "tool_call_id")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletionResponse {
    choices: Vec<Choice>,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[derive(Deserialize)]
struct ProxyOptions {
    #[serde(default = "default_true")]
    use_raph: bool,
}

fn default_true() -> bool {
    true
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    let positron_url =
        env::var("POSITRON_URL").unwrap_or_else(|_| "http://localhost:8080/v1".to_string());
    let positron_key = env::var("POSITRON_KEY").unwrap_or_else(|_| "sk-placeholder".to_string());
    let max_retries = env::var("MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);

    let state = AppState {
        positron_url,
        positron_key,
        max_retries,
        http: Client::new(),
        tool_regex: Regex::new(r"(?s)<tool_code>(.*?)</tool_code>").expect("regex"),
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(proxy_handler))
        .with_state(state.clone());

    let port: u16 = env::var("PROXY_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(9000);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("failed to bind port");

    println!(
        "🚀 Proxy running on http://0.0.0.0:{} -> Forwarding to {}",
        port, state.positron_url
    );

    axum::serve(listener, app).await.expect("server crashed");
}

async fn proxy_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(opts): Query<ProxyOptions>,
    Json(mut body): Json<Value>,
) -> Response {
    // Bypass logic via header or URL param.
    let header_bypass = headers
        .get("x-raph-mode")
        .and_then(|h| h.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let url_bypass = !opts.use_raph;

    // If bypass requested, forward directly (respect stream flag).
    if header_bypass || url_bypass {
        let wants_stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if wants_stream {
            return forward_passthrough_stream(&state, body).await;
        } else {
            return forward_passthrough_json(&state, body).await.into_response();
        }
    }

    // If no tools, choose pass-through mode (JSON or streaming).
    let tools_value = body
        .get("tools")
        .cloned()
        .unwrap_or_else(|| Value::Array(vec![]));
    let tools: Vec<Tool> = match serde_json::from_value(tools_value.clone()) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Invalid tools schema: {e}") })),
            )
                .into_response();
        }
    };

    if tools.is_empty() {
        let wants_stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if wants_stream {
            return forward_passthrough_stream(&state, body).await;
        } else {
            return forward_passthrough_json(&state, body).await.into_response();
        }
    }

    // Messages parsing.
    let mut messages: Vec<Message> = match body.get("messages") {
        Some(msgs) => match serde_json::from_value(msgs.clone()) {
            Ok(m) => m,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": format!("Invalid messages schema: {e}") })),
                )
                    .into_response();
            }
        },
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Missing messages" })),
            )
                .into_response();
        }
    };

    // Inject system prompt.
    let tool_defs = match serde_json::to_string_pretty(
        &tools.iter().map(|t| &t.function).collect::<Vec<_>>(),
    ) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize tools: {e}") })),
            )
                .into_response();
        }
    };

    let injection = TOOL_SYSTEM_PROMPT.replace("{tool_definitions}", &tool_defs);

    if let Some(first) = messages.first_mut() {
        let new_content = match &first.content {
            Some(Value::String(s)) => format!("{s}\n\n{injection}"),
            _ => injection.clone(),
        };
        first.content = Some(Value::String(new_content));
    } else {
        messages.insert(
            0,
            Message {
                role: "system".into(),
                content: Some(Value::String(injection.clone())),
                tool_calls: None,
                name: None,
                tool_call_id: None,
            },
        );
    }

    // Prepare body for Positron: remove tools/tool_choice, update messages.
    body.as_object_mut().map(|map| {
        map.remove("tools");
        map.remove("tool_choice");
        // Force non-streaming so we can buffer and retry reliably.
        map.insert("stream".to_string(), Value::Bool(false));
    });
    body["messages"] = serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(vec![]));

    // Compile schema validators once per request.
    let validators = match compile_validators(&tools) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Invalid tool schema: {e}") })),
            )
                .into_response();
        }
    };

    let mut attempt = 0usize;
    while attempt < state.max_retries {
        match forward_and_handle(
            &state,
            &mut body,
            &mut messages,
            &tools,
            &validators,
            attempt,
        )
        .await
        {
            Ok(resp) => return resp.into_response(),
            Err(e) => {
                eprintln!("Attempt {} error: {}", attempt + 1, e);
            }
        }
        attempt += 1;
    }

    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": "Max retries exceeded. Model failed to format tool call." })),
    )
        .into_response()
}

async fn forward_passthrough_json(state: &AppState, body: Value) -> (StatusCode, Json<Value>) {
    let resp = state
        .http
        .post(format!("{}/chat/completions", state.positron_url))
        .header("Authorization", format!("Bearer {}", state.positron_key))
        .json(&body)
        .send()
        .await;

    match resp {
        Ok(r) => match r.json::<Value>().await {
            Ok(json) => (StatusCode::OK, Json(json)),
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("Invalid JSON from Positron: {e}") })),
            ),
        },
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Positron request failed: {e}") })),
        ),
    }
}

async fn forward_passthrough_stream(state: &AppState, body: Value) -> Response {
    let req = match state
        .http
        .post(format!("{}/chat/completions", state.positron_url))
        .header("Authorization", format!("Bearer {}", state.positron_key))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("Positron request failed: {e}") })),
            )
                .into_response();
        }
    };

    let stream = req
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("stream error: {e}")));
    let body = Body::from_stream(stream);
    Response::new(body)
}

async fn forward_and_handle(
    state: &AppState,
    req_body: &mut Value,
    messages: &mut Vec<Message>,
    tools: &[Tool],
    validators: &HashMap<String, JSONSchema>,
    attempt: usize,
) -> Result<(StatusCode, Json<Value>), String> {
    let response = state
        .http
        .post(format!("{}/chat/completions", state.positron_url))
        .header("Authorization", format!("Bearer {}", state.positron_key))
        .json(req_body)
        .send()
        .await
        .map_err(|e| format!("Positron call failed: {e}"))?;

    let mut resp_json: CompletionResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Positron response: {e}"))?;

    if resp_json.choices.is_empty() {
        return Err("No choices returned".into());
    }

    let content_opt = resp_json.choices[0]
        .message
        .content
        .clone()
        .and_then(|c| c.as_str().map(|s| s.to_string()));

    let Some(content) = content_opt else {
        // If there's no content, just return the response.
        return Ok((
            StatusCode::OK,
            Json(serde_json::to_value(resp_json).unwrap()),
        ));
    };

    if let Some(caps) = state.tool_regex.captures(&content) {
        let raw_json = caps.get(1).map(|m| m.as_str()).unwrap_or_default().trim();

        let repaired_str = raw_json.to_string();

        let repaired: Value = match serde_json::from_str(&repaired_str)
            .or_else(|_| json5::from_str(&repaired_str))
        {
            Ok(v) => v,
            Err(e) => {
                push_retry_messages(messages, content, format!("Output is not valid JSON: {e}"));
                req_body["messages"] =
                    serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(vec![]));
                return Err(format!("Attempt {attempt} repair failed: {e}"));
            }
        };

        let (valid, error_msg, name, args) = validate_schema_and_args(&repaired, tools, validators);
        if valid {
            let arguments = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
            let tool_call = ToolCall {
                id: format!("call_{}", current_millis()),
                type_field: "function".into(),
                function: ToolCallFunction { name, arguments },
            };

            resp_json.choices[0].message.tool_calls = Some(vec![tool_call]);
            resp_json.choices[0].message.content = None;

            let value = serde_json::to_value(&resp_json).map_err(|e| e.to_string())?;
            return Ok((StatusCode::OK, Json(value)));
        } else {
            push_retry_messages(
                messages,
                content,
                format!("SYSTEM ERROR: {error_msg}. Try again using <tool_code>."),
            );
            req_body["messages"] =
                serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(vec![]));
            return Err(format!("Attempt {attempt} logic error: {error_msg}"));
        }
    }

    // No tool tags found; return as-is.
    let value = serde_json::to_value(&resp_json).map_err(|e| e.to_string())?;
    Ok((StatusCode::OK, Json(value)))
}

fn compile_validators(tools: &[Tool]) -> Result<HashMap<String, JSONSchema>, String> {
    let mut map = HashMap::new();
    for tool in tools {
        let schema_val = &tool.function.parameters;
        let compiled = JSONSchema::compile(schema_val)
            .map_err(|e| format!("Invalid schema for tool {}: {e}", tool.function.name))?;
        map.insert(tool.function.name.clone(), compiled);
    }
    Ok(map)
}

fn validate_schema_and_args(
    tool_json: &Value,
    available_tools: &[Tool],
    validators: &HashMap<String, JSONSchema>,
) -> (bool, String, String, Value) {
    if !tool_json.is_object() {
        return (
            false,
            "Output is not a valid JSON object.".into(),
            String::new(),
            Value::Null,
        );
    }

    let tool_name = tool_json
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let Some(name) = tool_name else {
        return (
            false,
            "JSON missing 'name' field.".into(),
            String::new(),
            Value::Null,
        );
    };

    let valid_names: Vec<String> = available_tools
        .iter()
        .map(|t| t.function.name.clone())
        .collect();
    if !valid_names.contains(&name) {
        return (
            false,
            format!("Tool '{name}' does not exist. Available tools: {valid_names:?}"),
            name,
            Value::Null,
        );
    }

    let args = tool_json
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));

    if let Some(validator) = validators.get(&name) {
        let validation = validator
            .validate(&args)
            .map_err(|errors| errors.map(|e| e.to_string()).collect::<Vec<_>>().join("; "));

        if let Err(combined) = validation {
            return (
                false,
                format!("Arguments failed validation: {combined}"),
                name,
                args,
            );
        }
    }

    (true, String::new(), name, args)
}

fn push_retry_messages(messages: &mut Vec<Message>, content: String, error: String) {
    messages.push(Message {
        role: "assistant".into(),
        content: Some(Value::String(content)),
        tool_calls: None,
        name: None,
        tool_call_id: None,
    });
    messages.push(Message {
        role: "user".into(),
        content: Some(Value::String(error)),
        tool_calls: None,
        name: None,
        tool_call_id: None,
    });
}

fn current_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
