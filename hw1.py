#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_deepseek import ChatDeepSeek

    system_prompt = (
        "You transcribe supermarket receipts (Hong Kong style, mixed Chinese and "
        "English) into structured JSON. You are literal and meticulous: you report "
        "exactly what is printed. You never compute, never round and never invent "
        "a number."
    )

    human_prompt = """Transcribe this supermarket receipt photo.

Return ONLY one JSON object (no commentary, no markdown fence) with exactly these
four keys:

{
  "lines": [{"label": "<line text>", "amount": <printed number, keep its sign>}],
  "subtotal": <number on the 小計 / SUBTOTAL / 合計 line, or null>,
  "rounding": <number on the ROUNDING line, or null>,
  "final_payment": <amount actually paid, or null>
}

How to fill each key:

1. "lines" lists every priced line printed ABOVE the subtotal line, in printed order:
   ordinary item lines (positive) and EVERY discount / promotion / coupon / member /
   app / packaging-damage line (negative), whatever language the label uses.
2. Keep the printed sign. Discounts are negative; never flip them to positive.
3. Some discount labels embed a number that is NOT a charged amount, such as
   "Buy 2 Save $12.8", "Buy 3 Save $10.8" or "5% OFF". In those cases the deducted
   amount is printed as a separate negative number on the same line or the line just
   below it. Record each negative amount exactly once, and never emit the label's own
   number as an extra line.
4. "subtotal" is copied from the subtotal line. Do not compute it yourself.
5. "rounding" is the amount on the ROUNDING line (often -0.01 or -0.02). Use null when
   that line is absent.
6. "final_payment" is what was actually charged: the amount printed beside the payment
   method (OCTOPUS / 八達通 / CASH / 現金 / VISA / 扣除金額) or the last amount before the
   change line. Never use 找錢 (change) or 餘額 (balance).
7. IGNORE everything below the payment block: card numbers, machine / shop / cashier
   IDs, loyalty points, balance, change, dates, times, barcodes, phone numbers.
8. Output plain JSON only, no explanation.
"""

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=4096,
        max_retries=3,
        timeout=180,
        # Without this the model can spend its entire output budget on hidden
        # reasoning tokens and return an empty message, which the parser then
        # rejects. Disabling it also cuts latency roughly 7x on dense receipts.
        extra_body={"thinking": {"type": "disabled"}},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {"type": "text", "text": "{instructions}"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )

    def parse_transcription(message: Any) -> dict[str, Any]:
        """Parse the model's reply into JSON, tolerating fences or stray prose."""
        content = getattr(message, "content", message)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            content = "\n".join(parts)
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            metadata = getattr(message, "response_metadata", {}) or {}
            raise ValueError(
                "model reply contained no JSON object "
                f"(chars={len(text)}, finish_reason={metadata.get('finish_reason')!r})"
            )
        return json.loads(text[start : end + 1])

    def to_decimal(value: Any) -> Decimal | None:
        """Read a printed money value out of whatever the model returned."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        text = str(value).replace(",", "").replace("HK$", "").replace("$", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text.replace(" ", ""))
        if match is None:
            return None
        try:
            return Decimal(match.group(0))
        except InvalidOperation:
            return None

    def per_receipt(data: Any) -> dict[str, Any]:
        """Turn one receipt's transcription into this receipt's two answers.

        The arithmetic lives here, in Python, not in the model: the model only
        transcribes printed numbers. Receipt identity is ``sum(lines) == subtotal``,
        which makes a missed discount line detectable instead of silent.
        """
        if not isinstance(data, dict):
            data = {}

        amounts = [
            amount
            for amount in (
                to_decimal(line.get("amount")) if isinstance(line, dict) else to_decimal(line)
                for line in (data.get("lines") or [])
            )
            if amount is not None
        ]

        subtotal = to_decimal(data.get("subtotal"))
        rounding = to_decimal(data.get("rounding"))
        final_payment = to_decimal(data.get("final_payment"))

        # Every negative printed amount above the subtotal line is a discount.
        discounts = sum((-amount for amount in amounts if amount < 0), Decimal("0"))
        positives = sum((amount for amount in amounts if amount > 0), Decimal("0"))
        lines_total = sum(amounts, Decimal("0"))

        # Fallbacks keep a partially readable receipt from poisoning the folder sum.
        if subtotal is None:
            subtotal = lines_total
        if final_payment is None:
            final_payment = subtotal + (rounding or Decimal("0"))

        paid = final_payment or Decimal("0")

        # Query 2 = SUBTOTAL plus every discount added back as a positive number,
        # deliberately excluding ROUNDING. Two independent routes compute it:
        #   A) the printed SUBTOTAL + the discount lines actually read
        #   B) the sum of the positive item lines
        # Both agree when nothing is missed. The receipt identity
        # sum(lines) == SUBTOTAL tells which route to trust: if the lines overrun
        # the printed SUBTOTAL, a negative discount line was dropped, so route B
        # (which never depends on discount lines) is the safe one. Route A is used
        # in the opposite case, where a positive item line was dropped instead.
        gap = lines_total - subtotal
        without_discount = positives if gap > Decimal("0.05") else subtotal + discounts

        return {
            "paid": paid,
            "without_discount": without_discount,
            "subtotal": subtotal,
            "discount_total": discounts,
            "positives_total": positives,
            "rounding": rounding,
            "lines_total": lines_total,
            "discount_lines": sum(1 for amount in amounts if amount < 0),
            "item_lines": sum(1 for amount in amounts if amount > 0),
            "balance_gap": gap,
            "self_check_ok": abs(gap) <= Decimal("0.05"),
        }

    prepare = RunnableLambda(
        lambda payload: {
            "instructions": human_prompt,
            "image_url": payload["image_url"],
        }
    )

    transcribe = (prepare | prompt | model | RunnableLambda(parse_transcription)).with_retry(
        stop_after_attempt=3
    )
    return transcribe | RunnableLambda(per_receipt)


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    # One independent transcription per receipt, run in parallel. Each result is
    # a per-receipt dict produced by the chain; the folder total is then summed
    # here in Python so no language model ever touches the arithmetic.
    payloads = [{"image_url": image_data_url(path)} for path in images]
    results: list[Any] = list(
        chain.batch(payloads, config={"max_concurrency": 4}, return_exceptions=True)
    )

    # A batch failure must never take the whole submission down: retry the
    # offending receipt on its own, then simply skip it if it keeps failing.
    for index, result in enumerate(results):
        if not isinstance(result, BaseException):
            continue
        for _ in range(2):
            try:
                results[index] = chain.invoke(payloads[index])
                break
            except Exception:  # noqa: BLE001 - a single bad receipt must not crash the run
                continue

    paid_total = Decimal("0")
    without_total = Decimal("0")
    for result in results:
        if isinstance(result, BaseException) or not isinstance(result, dict):
            continue
        paid_total += result.get("paid") or Decimal("0")
        without_total += result.get("without_discount") or Decimal("0")

    # Each response must contain exactly one number, so the answers are formatted
    # to two decimals and carry no other digits.
    return {
        QUERY_1: f"HK${paid_total.quantize(Decimal('0.01')):.2f}",
        QUERY_2: f"HK${without_total.quantize(Decimal('0.01')):.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
