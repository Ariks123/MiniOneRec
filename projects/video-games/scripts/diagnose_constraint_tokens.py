#!/usr/bin/env python3
"""Inspect response boundaries using real tokenizers, without model weights.

Only tokenizer files are loaded. Applying a chat template is an explicit step;
loading an Instruct tokenizer does not apply its template to ordinary text.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


HEADER = "### Response:\n"
SID_PARTS = ["<a_1>", "<b_2>", "<c_3>"]
SID = "".join(SID_PARTS)
USER_TEXT = f"Recommend the next item after {SID}."
PLAIN_PROMPT = f"### User Input:\n{USER_TEXT}\n\n{HEADER}"


def token_view(tokenizer, text, *, add_special_tokens=False):
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return {
        "text": text,
        "length": len(ids),
        "ids": ids,
        "tokens": tokenizer.convert_ids_to_tokens(ids),
    }


def inspect_phase(tokenizer):
    header = token_view(tokenizer, HEADER)
    full_response = token_view(tokenizer, HEADER + SID + "\n")
    prompt = token_view(tokenizer, PLAIN_PROMPT)
    result = {
        "header": header,
        "header_with_tokenizer_defaults": token_view(
            tokenizer, HEADER, add_special_tokens=True
        ),
        "sid": token_view(tokenizer, SID),
        "completion": token_view(tokenizer, SID + "\n"),
        "full_response": full_response,
        "header_is_token_prefix_of_full_response": (
            full_response["ids"][: header["length"]] == header["ids"]
        ),
        "plain_prompt": prompt,
        "legacy_three_token_root": full_response["ids"][:3],
        "plain_prompt_matches_legacy_root": (
            prompt["ids"][-3:] == full_response["ids"][:3]
        ),
    }
    messages = [{"role": "user", "content": USER_TEXT}]
    try:
        chat_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result["explicit_chat_prompt"] = {
            "text": chat_text,
            "length": len(chat_ids),
            "ids": chat_ids,
            "tokens": tokenizer.convert_ids_to_tokens(chat_ids),
            "matches_legacy_three_token_root": (
                chat_ids[-3:] == full_response["ids"][:3]
            ),
        }
    except ValueError as exc:
        result["explicit_chat_prompt"] = {
            "available": False,
            "reason": str(exc),
        }
    return result


def inspect_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    result = {
        "model": model_name,
        "tokenizer_class": type(tokenizer).__name__,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "has_chat_template": bool(tokenizer.chat_template),
        "original": inspect_phase(tokenizer),
    }
    # SID tokens must remain visible to batch_decode(skip_special_tokens=True).
    # Match TokenExtender's ordinary added tokens instead of special tokens.
    result["added_sid_tokens"] = tokenizer.add_tokens(SID_PARTS)
    result["with_registered_sid_tokens"] = inspect_phase(tokenizer)
    restored_sid = tokenizer.decode(
        tokenizer.encode(SID, add_special_tokens=False), skip_special_tokens=True
    )
    result["registered_sid_round_trip"] = restored_sid
    if result["with_registered_sid_tokens"]["sid"]["length"] != len(SID_PARTS):
        raise AssertionError(f"SID tokens did not register as atomic tokens: {model_name}")
    if restored_sid != SID:
        raise AssertionError(f"SID tokens disappeared or changed during decoding: {model_name}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--instruct-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "tmp/constraint_token_diagnostics.json",
    )
    args = parser.parse_args()
    report = {
        "scope": "Real tokenizer inspection only; no model weights or training.",
        "notes": [
            "prefix_index in the legacy code counts response-header tokens, not SID parts.",
            "An Instruct tokenizer does not apply its chat template automatically.",
            "The current evaluation path uses a plain prompt, not apply_chat_template.",
            "Registered SID tokens can tokenize identically in base and Instruct models.",
        ],
        "models": [inspect_model(args.base_model), inspect_model(args.instruct_model)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model in report["models"]:
        print(model["model"])
        for label in ("original", "with_registered_sid_tokens"):
            phase = model[label]
            chat = phase["explicit_chat_prompt"]
            print(
                f"  {label}: header={phase['header']['ids']}; "
                f"SID length={phase['sid']['length']}; "
                f"plain prompt length={phase['plain_prompt']['length']}; "
                f"explicit chat prompt length={chat.get('length', 'unavailable')}"
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
