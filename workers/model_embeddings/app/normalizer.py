def normalize_input(source_model: str, payload: dict) -> str:

    if source_model == "ocr":
        lines = [item["text"] for item in payload if "text" in item]
        return "\n".join(lines)

    if source_model == "moondream":
        return str(payload)

    if source_model == "raw_text":
        return payload["text"]

    raise ValueError(f"Unknown source_model: {source_model}")
