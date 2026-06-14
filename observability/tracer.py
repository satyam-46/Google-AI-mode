"""Simple tracer stub for LangSmith/OpenTelemetry integration."""

def record_trace(record: dict) -> None:
    # placeholder: write to sqlite or logging
    print("TRACE:", record)
