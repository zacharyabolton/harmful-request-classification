"""Record data choices and the limits of training-data overlap checks."""


def required_text(record, fields):
    for field in fields:
        value = record.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.strip().upper() in {"TODO", "PENDING", "PLACEHOLDER"}
        ):
            raise ValueError("collection requires " + field)


def validate_collection(record):
    if not isinstance(record, dict):
        raise ValueError("complete the collection record")
    for name in ("extra_sources", "synthetic_augmentation"):
        choice = record.get(name)
        if not isinstance(choice, dict) or type(choice.get("use")) is not bool:
            raise ValueError("record the collection choice: " + name)
        required_text(choice, ("reason",))
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("collection requires sources")
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("collection source must be an object")
        required_text(source, ("source", "revision", "origin", "permission", "labels"))
        key = (source["source"], source["revision"])
        if key in seen:
            raise ValueError("duplicate collection source and revision")
        seen.add(key)
        role = source.get("role")
        if role not in {"supplied", "external", "generated"}:
            raise ValueError("source role must be supplied, external or generated")
        choice = {
            "external": "extra_sources",
            "generated": "synthetic_augmentation",
        }.get(role)
        if choice and not record[choice]["use"]:
            raise ValueError("source contradicts collection choice: " + choice)
        if role == "generated":
            required_text(
                source,
                (
                    "generator",
                    "generator_revision",
                    "recipe",
                    "review",
                    "overlap_check",
                ),
            )
    checks = record.get("model_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(
            "record model training-data overlap checks; unknown is allowed"
        )
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("model check must be an object")
        required_text(check, ("model", "revision", "evidence", "limits"))
        if check.get("status") not in {
            "known_overlap",
            "none_found_in_checked_material",
            "unknown",
            "not_applicable",
        }:
            raise ValueError("invalid model overlap status")


def check_rows(rows, record):
    validate_collection(record)
    sources = {(s["source"], s["revision"]): s for s in record["sources"]}
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        source = sources.get((row["source"], row["revision"]))
        if source is None:
            raise ValueError("row source and revision missing from collection record")
        if source["role"] != "generated":
            continue
        parent = by_id.get(row.get("parent_id"))
        if (
            row.get("split") != "train"
            or parent is None
            or parent["id"] == row["id"]
            or parent.get("split") != "train"
            or parent.get("group_id") != row.get("group_id")
            or sources.get((parent["source"], parent["revision"]), {}).get("role")
            not in {"supplied", "external"}
        ):
            raise ValueError(
                "generated rows need an original training parent in the same group"
            )


def check_encoder(record, revision):
    """Bind the built-in pretrained model to its declared overlap assessment."""
    matches = [
        check
        for check in record["model_checks"]
        if check["model"] == "microsoft/deberta-v3-xsmall"
        and check["revision"] == revision
    ]
    if len(matches) != 1 or matches[0]["status"] == "not_applicable":
        raise ValueError(
            "record an overlap check for the exact DeBERTa checkpoint before planning; unknown is allowed"
        )
