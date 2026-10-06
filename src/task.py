"""Optional task definitions and shared limits."""


def validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("complete the task configuration before proceeding")
    keys = {
        "task_id",
        "author",
        "target",
        "inputs",
        "label_map",
        "data_permissions",
        "split_policy",
        "minimums",
        "review_per_class",
        "collection",
    }
    if set(task) != keys:
        raise ValueError("task fields missing or unknown")
    for name in ("task_id", "author", "target", "inputs", "data_permissions"):
        value = task[name]
        if (
            not isinstance(value, str)
            or not value.strip()
            or any(word in value.upper() for word in ("TODO", "PLACEHOLDER", "PENDING"))
        ):
            raise ValueError("explicit task " + name + " required")
    labels = task["label_map"]
    if (
        not isinstance(labels, dict)
        or set(labels) != {"0", "1"}
        or any(not isinstance(v, str) or not v.strip() for v in labels.values())
        or labels["0"] == labels["1"]
    ):
        raise ValueError(
            "requires two distinct declared labels, 0 and 1; adapt code for other tasks"
        )
    if task["split_policy"] != "supplied":
        raise ValueError(
            "task preparation requires supplied partitions; create and document task-specific splits first"
        )
    count = task["review_per_class"]
    if type(count) is not int or count <= 0:
        raise ValueError("review_per_class must be a positive integer")
    minimums = task["minimums"]
    if not isinstance(minimums, dict) or set(minimums) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("declare support requirements for every split")
    for split, minimum in minimums.items():
        if not isinstance(minimum, dict) or set(minimum) != {
            "rows",
            "per_class",
            "groups",
        }:
            raise ValueError("support requirements need rows, per_class and groups")
        if any(type(v) is not int or v <= 0 for v in minimum.values()):
            raise ValueError("support requirements must be positive integers")
        if split == "train" and minimum["per_class"] < count:
            raise ValueError("training support must cover the requested label review")
    from collection import validate_collection

    validate_collection(task["collection"])
    return task


def budget_key(cfg):
    return "run_budget_seconds"


def validation_minimum(cfg):
    if cfg.get("task"):
        return cfg["task"]["minimums"]["validation"]["per_class"]
    return 5 if cfg["fixture_only"] else 500


def label_map(cfg):
    return (
        cfg["task"]["label_map"] if cfg.get("task") else {"0": "benign", "1": "harmful"}
    )
