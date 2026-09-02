import os
import sys
import pathlib
from typing import Dict, List

import typer
import yaml


def _expand_vars(value: str, vars_map: Dict[str, str]) -> str:
    if not isinstance(value, str):
        return value
    # Build a combined env with YAML vars taking precedence
    env = os.environ.copy()
    env.update({k: str(v) for k, v in vars_map.items()})
    # Support ${VAR} expansion
    out = value
    for k, v in env.items():
        out = out.replace(f"${{{k}}}", str(v))
    return out


def _normalize_input_entry(entry, vars_map: Dict[str, str]) -> str:
    # Accept multiple shapes and normalize to "file:label"
    if isinstance(entry, str):
        s = _expand_vars(entry, vars_map)
        if ":" in s:
            return s
        # No explicit label provided: derive from filename stem
        p = pathlib.Path(s)
        return f"{s}:{p.stem}"

    if isinstance(entry, dict):
        # Shape 1: {file: ..., label: ...} (or {path: ..., label: ...})
        file_val = entry.get("file") or entry.get("path")
        label_val = entry.get("label")
        if file_val is not None:
            file_s = _expand_vars(str(file_val), vars_map)
            if label_val is None:
                label_s = pathlib.Path(file_s).stem
            else:
                label_s = _expand_vars(str(label_val), vars_map)
            return f"{file_s}:{label_s}"

        # Shape 2: {Label: File} (single-item mapping)
        if len(entry) == 1:
            (label_key, file_value), = entry.items()
            file_s = _expand_vars(str(file_value), vars_map)
            label_s = _expand_vars(str(label_key), vars_map)
            return f"{file_s}:{label_s}"

    raise ValueError(f"Unsupported input entry format: {entry}")


def _resolve_inputs(
    task_cfg: Dict,
    inputs_dict: Dict[str, List[str]] | Dict[str, Dict],
    vars_map: Dict[str, str],
    labels_dict: Dict[str, List[str]] | None = None,
    versions_dict: Dict[str, List[str]] | None = None,
    sets_dict: Dict[str, Dict[str, str]] | None = None,
) -> List[str]:
    # Resolve the source from either inline 'inputs' or 'inputs_ref'
    if "inputs" in task_cfg:
        src = task_cfg["inputs"]
    elif "inputs_ref" in task_cfg:
        ref = task_cfg["inputs_ref"]
        if ref not in inputs_dict:
            raise KeyError(f"inputs_ref '{ref}' not found in top-level 'inputs'")
        src = inputs_dict[ref]
    else:
        return []

    # Normalize src to an iterable of entries
    if isinstance(src, dict):
        # Supported dict forms:
        # 1) Map form: {Label: File, ...}
        # 2) Structured form: {files: [...], labels: [...], labels_ref: name}
        # 3) Template form: {template: "...${VERSION}...", versions: [...], versions_ref: name, labels/labels_ref}
        # 4) Set form: {template: "...${VERSION}...", set: {version: label, ...} | set_ref: name}
        if ("set" in src or "set_ref" in src) and "template" in src:
            template = _expand_vars(str(src.get("template")), vars_map)
            # task-level override
            task_set = task_cfg.get("set")
            task_set_ref = task_cfg.get("set_ref")
            set_raw = task_set if task_set is not None else src.get("set")
            set_ref = task_set_ref if task_set_ref is not None else src.get("set_ref")

            if set_raw is None:
                if not set_ref or sets_dict is None:
                    raise ValueError("Set form requires 'set' mapping or 'set_ref'.")
                if set_ref not in sets_dict:
                    raise KeyError(f"set_ref '{set_ref}' not found in top-level 'sets'")
                set_raw = sets_dict[set_ref]

            if not isinstance(set_raw, dict):
                raise ValueError("'set' must be a mapping of version -> label")

            out: List[str] = []
            for ver_key, label_val in set_raw.items():
                ver = _expand_vars(str(ver_key), vars_map)
                lbl = _expand_vars(str(label_val), vars_map)
                file_s = template.replace("${VERSION}", ver)
                out.append(f"{file_s}:{lbl}")
            return out
        if "template" in src:
            template = _expand_vars(str(src.get("template")), vars_map)
            # Allow task-level override
            task_versions = task_cfg.get("versions")
            task_versions_ref = task_cfg.get("versions_ref")
            versions_raw = task_versions if task_versions is not None else src.get("versions")
            versions_ref = task_versions_ref if task_versions_ref is not None else src.get("versions_ref")

            if versions_raw is None and versions_ref and versions_dict is not None:
                if versions_ref not in versions_dict:
                    raise KeyError(f"versions_ref '{versions_ref}' not found in top-level 'versions'")
                versions_raw = versions_dict[versions_ref]

            if not versions_raw or not isinstance(versions_raw, list):
                raise ValueError("Template form requires 'versions' list or 'versions_ref'.")

            # Resolve labels
            task_labels = task_cfg.get("labels")
            task_labels_ref = task_cfg.get("labels_ref")
            labels_raw = task_labels if task_labels is not None else src.get("labels")
            labels_ref = task_labels_ref if task_labels_ref is not None else src.get("labels_ref")
            if labels_raw is None and labels_ref and labels_dict is not None:
                if labels_ref not in labels_dict:
                    raise KeyError(f"labels_ref '{labels_ref}' not found in top-level 'labels'")
                labels_raw = labels_dict[labels_ref]

            versions_list = [_expand_vars(str(v), vars_map) for v in versions_raw]
            labels_list: List[str] | None = None
            if labels_raw is not None:
                labels_list = [_expand_vars(str(l), vars_map) for l in labels_raw]
                if len(labels_list) != len(versions_list):
                    raise ValueError(
                        f"labels/versions length mismatch: labels={len(labels_list)} versions={len(versions_list)}"
                    )

            out: List[str] = []
            for i, ver in enumerate(versions_list):
                file_s = template.replace("${VERSION}", ver)
                if labels_list is not None:
                    out.append(f"{file_s}:{labels_list[i]}")
                else:
                    out.append(_normalize_input_entry(file_s, vars_map))
            return out

        if "files" in src:
            files_raw = src.get("files", [])
            # Allow task-level labels override
            task_labels = task_cfg.get("labels")
            task_labels_ref = task_cfg.get("labels_ref")
            labels_raw = task_labels if task_labels is not None else src.get("labels")
            labels_ref = task_labels_ref if task_labels_ref is not None else src.get("labels_ref")

            if labels_raw is None and labels_ref and labels_dict is not None:
                if labels_ref not in labels_dict:
                    raise KeyError(f"labels_ref '{labels_ref}' not found in top-level 'labels'")
                labels_raw = labels_dict[labels_ref]

            files = [_expand_vars(f if isinstance(f, str) else str(f), vars_map) for f in files_raw]

            labels: List[str] | None = None
            if labels_raw is not None:
                labels = [_expand_vars(str(l), vars_map) for l in labels_raw]
                if len(labels) != len(files):
                    raise ValueError(
                        f"labels/files length mismatch: labels={len(labels)} files={len(files)}"
                    )

            out: List[str] = []
            for i, f in enumerate(files):
                if labels is not None:
                    out.append(f"{f}:{labels[i]}")
                else:
                    out.append(_normalize_input_entry(f, vars_map))
            return out

        # Map form fallback
        reserved_keys = {"additional_text"}
        entries = [{"label": k, "file": v} for k, v in src.items() if k not in reserved_keys]
        return [_normalize_input_entry(e, vars_map) for e in entries]

    if isinstance(src, list):
        entries = src
        return [_normalize_input_entry(e, vars_map) for e in entries]

    raise ValueError("inputs/inputs_ref must be a list or a mapping")


def _select_tasks(tasks_cfg: Dict[str, Dict], groups_cfg: Dict[str, List[str]], what: str) -> List[str]:
    # what: 'all' or comma-separated list of tasks and/or groups
    if what == "all":
        selected = [
            name
            for name, cfg in tasks_cfg.items()
            if not cfg.get("exclude_from_all", False)
        ]
        return selected
    wanted = [w.strip() for w in what.split(",") if w.strip()]
    existing_tasks = set(tasks_cfg.keys())
    existing_groups = set(groups_cfg.keys())
    selected: List[str] = []
    seen = set()

    for item in wanted:
        if item in existing_groups:
            for task_name in groups_cfg[item]:
                if task_name in existing_tasks and task_name not in seen:
                    selected.append(task_name)
                    seen.add(task_name)
        elif item in existing_tasks and item not in seen:
            selected.append(item)
            seen.add(item)

    return selected


def _resolve_additional_text(additional_text_cfg, vars_map: Dict[str, str]):
    if additional_text_cfg is None:
        return None

    if not isinstance(additional_text_cfg, list):
        raise ValueError("additional_text must be a list of [x, y, text] entries")

    resolved = []
    for idx, item in enumerate(additional_text_cfg):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"additional_text entry at index {idx} must be [x, y, text]")
        x_pos, y_pos, text = item
        resolved.append((float(x_pos), float(y_pos), _expand_vars(str(text), vars_map)))
    return resolved


def _resolve_task_additional_text(task_cfg: Dict, inputs_dict: Dict, default_additional_text, vars_map: Dict[str, str]):
    if "additional_text" in task_cfg:
        return _resolve_additional_text(task_cfg.get("additional_text"), vars_map)

    if "inputs_ref" in task_cfg:
        ref = task_cfg["inputs_ref"]
        input_cfg = inputs_dict.get(ref)
        if isinstance(input_cfg, dict) and "additional_text" in input_cfg:
            return _resolve_additional_text(input_cfg.get("additional_text"), vars_map)

    return default_additional_text


def main(
    config: str = typer.Option(..., "-c", "--config", help="YAML config file"),
    what: str = typer.Option("all", "-w", "--what", help="Tasks to run: 'all' or comma-separated list"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions without executing"),
    target_version: str = typer.Option(None, "--target-version", help="Override vars.TARGET_VERSION for ${VERSION} templates"),
    target_label: str = typer.Option(None, "--target-label", help="Override vars.TARGET_LABEL used in sets or labels"),
    target: str = typer.Option(None, "--target", help="Set both version and label using 'version:label' syntax"),
    force: bool = typer.Option(False, "--force", help="Skip overwrite confirmation when publishing"),
):
    cfg_path = pathlib.Path(config)
    if not cfg_path.exists():
        typer.echo(f"Config not found: {cfg_path}")
        raise typer.Exit(code=2)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    vars_map: Dict[str, str] = cfg.get("vars", {}) or {}
    if target:
        if ":" not in target:
            raise typer.BadParameter("--target must be in 'version:label' format")
        ver, lbl = target.split(":", 1)
        vars_map["TARGET_VERSION"] = ver
        vars_map["TARGET_LABEL"] = lbl
    if target_version:
        vars_map["TARGET_VERSION"] = target_version
    if target_label:
        vars_map["TARGET_LABEL"] = target_label
    target_dir = _expand_vars(cfg.get("target_dir", ""), vars_map)
    tasks_cfg: Dict[str, Dict] = cfg.get("tasks", {}) or {}
    groups_cfg: Dict[str, List[str]] = cfg.get("groups", {}) or {}
    inputs_dict: Dict[str, List[str]] = cfg.get("inputs", {}) or {}
    labels_dict: Dict[str, List[str]] = cfg.get("labels", {}) or {}
    versions_dict: Dict[str, List[str]] = cfg.get("versions", {}) or {}
    sets_dict: Dict[str, Dict[str, str]] = cfg.get("sets", {}) or {}
    default_additional_text = _resolve_additional_text(cfg.get("additional_text"), vars_map)

    if not tasks_cfg:
        typer.echo("No tasks defined in YAML under 'tasks'. Nothing to do.")
        raise typer.Exit(code=2)

    to_run = _select_tasks(tasks_cfg, groups_cfg, what)
    if not to_run:
        typer.echo("No matching tasks to run. Check --what or YAML tasks.")
        raise typer.Exit(code=2)

    for name in to_run:
        tcfg = tasks_cfg[name]
        module = _expand_vars(tcfg.get("module", ""), vars_map)
        what_key = tcfg.get("what", "")
        inputs = _resolve_inputs(tcfg, inputs_dict, vars_map, labels_dict, versions_dict, sets_dict)
        input_files_arg = ",".join(inputs)
        task_additional_text = _resolve_task_additional_text(
            tcfg,
            inputs_dict,
            default_additional_text,
            vars_map,
        )

        if dry_run:
            typer.echo(
                f"[dry-run] draw: task={name} module={module} what={what_key} target_dir={target_dir} inputs={len(inputs)} text={len(task_additional_text or [])}"
            )
            continue

        # Call existing draw entrypoint
        from draw import draw as draw_entry  # local import to avoid ROOT import on dry-run
        draw_entry(
            input_files=input_files_arg,
            module=module,
            what=what_key,
            target_dir=target_dir,
            additional_text=task_additional_text,
            force_publish=force,
        )


if __name__ == "__main__":
    typer.run(main)
