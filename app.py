#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gradio>=4.0.0",
#   "requests",
#   "pillow",
#   "python-dotenv",
#   "numpy",
#   "minio"
# ]
# ///
"""
Gradio app: WP Layout annotation (bboxes)
- Load image from URL
- Draw rectangles by clicking two points
- Label them
- Save to API (cases + layout_annotations)

Run:
  python app.py
or:
  uv run app.py

Env (local) :
  API_BASE_URL=http://localhost:8000
  API_KEY=...
"""
import os
import io
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from urllib.parse import quote, urlparse

import requests
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from minio import Minio

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT_URL")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "images")
MINIO_PUBLIC_BASE = os.getenv("MINIO_PUBLIC_BASE_URL", "https://minio.smartbiblia.fr").rstrip("/")

MINIO_ENABLED = bool(MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY)
minio_client = None
if MINIO_ENABLED:
    secure = not (
        MINIO_ENDPOINT.startswith("http://")
        or "localhost" in MINIO_ENDPOINT
        or "127.0.0.1" in MINIO_ENDPOINT
    )
    clean_endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    minio_client = Minio(
        clean_endpoint,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=secure,
    )

BUCKET = BUCKET_NAME
TREE_ROOT = Path("./.bucket_tree") / BUCKET

MEMOIRE_TYPE_OPTIONS = {
    "dea": "Mémoire de DEA",
    "dess": "Mémoire de DESS",
    "maitrise": "Mémoire de maîtrise",
    "master": "Mémoire de master",
    "hdr": "HDR",
    "autres": "Autres",
    "undefined": "Indeterminé",
}
# -----------------------
# API helpers
# -----------------------
headers = {
  "X-API-Key": API_KEY,
  "Accept": "application/json",
  "Content-Type": "application/json",
}

def api_get(path: str, params=None):
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict):
    r = requests.post(f"{API_BASE}{path}", json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()
    
# -----------------------
# Image helpers
# -----------------------
def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def get_annotated_source_refs(limit: int = 2000) -> set[str]:
    rows = api_get("/cases", params={"limit": limit})
    refs = set()
    for row in rows or []:
        source_ref = row.get("source_ref") if isinstance(row, dict) else None
        if source_ref:
            refs.add(str(source_ref).strip())
    return refs


def source_ref_from_object_name(bucket: str, object_name: str) -> str:
    parts = [p for p in object_name.replace("\\", "/").split("/") if p]
    if len(parts) >= 4:
        return "/".join([bucket, parts[-4], parts[-3], parts[-2], parts[-1]])
    return f"{bucket}/{object_name.replace('\\', '/').lstrip('/')}"


def sync_bucket_tree(bucket: str, tree_root: Path, excluded_source_refs: set[str] | None = None):
    tree_root.mkdir(parents=True, exist_ok=True)
    for child in tree_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    total_objects = 0
    kept_objects = 0
    skipped_objects = 0
    for obj in minio_client.list_objects(bucket, recursive=True):
        object_name = (obj.object_name or "").strip("/")
        if not object_name or object_name.endswith("/"):
            continue
        total_objects += 1
        if excluded_source_refs:
            obj_source_ref = source_ref_from_object_name(bucket, object_name)
            if obj_source_ref in excluded_source_refs:
                skipped_objects += 1
                continue
        placeholder = tree_root / object_name
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch(exist_ok=True)
        kept_objects += 1
    return total_objects, kept_objects, skipped_objects

def fetch_minio_image(object_name: str) -> Image.Image:
    response = minio_client.get_object(BUCKET_NAME, object_name)
    try:
        image_bytes = response.read()
    finally:
        response.close()
        response.release_conn()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def refresh_tree():
    if not MINIO_ENABLED or minio_client is None:
        return gr.update(), "MinIO n'est pas configure dans .env."
    try:
        excluded_source_refs = get_annotated_source_refs(limit=2000)
        total, kept, skipped = sync_bucket_tree(BUCKET, TREE_ROOT, excluded_source_refs=excluded_source_refs)
        refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return gr.update(), (
            f"Dernier rafraichissement : {refreshed_at} | "
            f"Total: {total} | Affichés: {kept} | Masqués (deja annotés): {skipped}"
        )
    except Exception as e:
        return gr.update(), f"Erreur de rafraichissement : {e}"

def normalize_object_name(selected_str: str, candidate: Path, bucket: str, tree_root: Path) -> str:
    def _extract_from_path(path_str: str) -> str:
        p = (path_str or "").replace("\\", "/").strip().strip('"').strip("'")
        parts = [x for x in p.split("/") if x and x != "."]

        # Robust extraction for paths that include ".../.bucket_tree/<bucket>/..."
        for i in range(len(parts) - 1):
            if parts[i] == ".bucket_tree" and parts[i + 1] == bucket:
                return "/".join(parts[i + 2 :])

        # Common case: path starts with "<bucket>/..."
        if parts and parts[0] == bucket:
            return "/".join(parts[1:])

        return "/".join(parts)

    for raw in (selected_str, candidate.as_posix()):
        object_name = _extract_from_path(raw)
        if object_name:
            tree_root_norm = tree_root.resolve().as_posix().rstrip("/")
            if object_name.startswith(tree_root_norm + "/"):
                object_name = object_name[len(tree_root_norm) + 1 :]

            tree_root_rel = str(tree_root).replace("\\", "/").rstrip("/")
            if object_name.startswith(tree_root_rel + "/"):
                object_name = object_name[len(tree_root_rel) + 1 :]

            if object_name.startswith(f"{bucket}/"):
                object_name = object_name[len(bucket) + 1 :]

            if object_name:
                return object_name
    return ""

def load_image_from_tree(selected_path=None, auto_metadata=True, evt: gr.SelectData = None):
    print(f"[tree] raw selected_path={selected_path!r}", flush=True)
    if evt is not None:
        print(f"[tree] evt.value={getattr(evt, 'value', None)!r}", flush=True)
    print(f"[tree] auto_metadata={auto_metadata!r}", flush=True)

    def _ret(
        img_val=gr.skip(),
        status_msg="",
        clean_val=gr.skip(),
        anns_val=None,
        table_val=None,
        pending_val=None,
        image_url_val=gr.skip(),
        doc_type_val=gr.skip(),
        doc_id_val=gr.skip(),
        collection_code_val=gr.skip(),
        source_ref_val=gr.skip(),
        case_id_val=None,
    ):
        if anns_val is None:
            anns_val = []
        if table_val is None:
            table_val = []
        return (
            img_val,
            status_msg,
            clean_val,
            anns_val,
            table_val,
            pending_val,
            image_url_val,
            doc_type_val,
            doc_id_val,
            collection_code_val,
            source_ref_val,
            case_id_val,
        )

    if isinstance(selected_path, bool) and evt is not None and getattr(evt, "value", None):
        auto_metadata = selected_path
        selected_path = None

    if (selected_path is None or selected_path == "") and evt is not None and getattr(evt, "value", None):
        selected_path = evt.value

    if not selected_path:
        print("[tree] empty selection", flush=True)
        return _ret(status_msg="La selection de l'arborescence est vide.")
    if isinstance(selected_path, (list, tuple)):
        if not selected_path:
            print("[tree] empty list/tuple selection", flush=True)
            return _ret(status_msg="La liste de selection de l'arborescence est vide.")
        selected_path = selected_path[0]
    if isinstance(selected_path, dict):
        selected_path = (
            selected_path.get("path")
            or selected_path.get("value")
            or selected_path.get("name")
            or ""
        )

    selected_str = str(selected_path).strip().strip('"').strip("'")
    print(f"[tree] selected_str={selected_str!r}", flush=True)
    if not selected_str:
        return _ret(status_msg="Impossible d'analyser la selection de l'arborescence.")

    candidate = Path(selected_str)
    if not candidate.is_absolute():
        candidate = (TREE_ROOT / candidate).resolve()
    print(f"[tree] candidate={candidate}", flush=True)
    if candidate.exists() and candidate.is_dir():
        return _ret(status_msg=f"Dossier selectionne : {candidate}")

    object_name = normalize_object_name(selected_str, candidate, BUCKET, TREE_ROOT)

    if not object_name:
        print("[tree] object_name empty after normalization", flush=True)
        return _ret(status_msg=f"Impossible de deduire le chemin de l'objet a partir de : {selected_str}")

    public_url = f"{MINIO_PUBLIC_BASE}/{BUCKET}/{quote(object_name, safe='/')}"
    print(f"[tree] object_name={object_name!r}", flush=True)
    print(f"[tree] public_url={public_url}", flush=True)
    meta_doc_type, meta_doc_id, meta_collection_code, meta_source_ref = metadata_updates_from_image_ref(public_url, auto_metadata)
    img_array, status, clean_img, _, _ = load_image_and_prepare(public_url)
    if clean_img is None:
        print(f"[tree] URL load failed: {status}", flush=True)
        return _ret(
            status_msg=f"{status} | object={object_name} | url={public_url}",
            image_url_val=public_url,
            doc_type_val=meta_doc_type,
            doc_id_val=meta_doc_id,
            collection_code_val=meta_collection_code,
            source_ref_val=meta_source_ref,
        )
    print("[tree] URL load success", flush=True)
    return _ret(
        img_val=img_array,
        status_msg=f"{status} | object={object_name} | url={public_url}",
        clean_val=clean_img,
        anns_val=[],
        table_val=[],
        pending_val=None,
        image_url_val=public_url,
        doc_type_val=meta_doc_type,
        doc_id_val=meta_doc_id,
        collection_code_val=meta_collection_code,
        source_ref_val=meta_source_ref,
    )

def draw_boxes_on_image(image: Image.Image, boxes, pending_point=None, selected_index=None):
    """Helper to draw boxes and pending point on image. Returns numpy array for efficiency."""
    if image is None: return None
    
    # Create a new image for drawing
    out_img = image.copy()
    draw = ImageDraw.Draw(out_img)
    w, h = image.size

    # Draw existing boxes
    for idx, box_data in enumerate(boxes):
        # box_data is now [x1, y1, x2, y2, block_code]
        x1, y1, x2, y2 = box_data[0:4]
        is_selected = selected_index is not None and idx == selected_index
        color = "red" if is_selected else "green"
        width = 4 if is_selected else 3
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        # Optionally draw block code on the image, for now just green rects

    # Draw pending point if exists
    if pending_point:
        x, y = pending_point
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill="yellow", outline="black")
        # Draw crosshair guides
        draw.line([(0, y), (w, y)], fill="cyan", width=1)
        draw.line([(x, 0), (x, h)], fill="cyan", width=1)

    # Convert to numpy array for more efficient Gradio updates
    return np.array(out_img)


# -----------------------
# Core logic
# -----------------------
def refresh_reference_data():
    """Return (campaign_choices, block_choices, campaigns_raw, blocks_raw)."""
    campaigns = api_get("/campaigns")
    blocks = api_get("/block-types")
    if not campaigns:
        raise RuntimeError("No campaigns found. Create at least one campaign in DB.")
    if not blocks:
        raise RuntimeError("No block-types found. Seed block_types in DB.")

    camp_choices = [f'{c["campaign_name"]} | {c["id"]}' for c in campaigns]
    block_choices = [f'{b["code"]} | {b["label"]}' for b in blocks]
    return camp_choices, block_choices, campaigns, blocks

def _case_id_from_row(row: dict) -> str:
    return str(row.get("case_id") or row.get("id") or "").strip()

def _build_image_url_from_case(row: dict) -> str:
    image_uri = (row.get("image_uri") or "").strip()
    if image_uri:
        return image_uri
    source_ref = (row.get("source_ref") or "").strip().lstrip("/")
    if source_ref:
        if source_ref.startswith("http://") or source_ref.startswith("https://"):
            return source_ref
        return f"{MINIO_PUBLIC_BASE}/{source_ref}"
    return ""

def _block_code_from_annotation_row(row: dict) -> str:
    direct = (row.get("block_code") or row.get("block_type_code") or "").strip()
    if direct:
        return direct
    nested = row.get("block_type")
    if isinstance(nested, dict):
        nested_code = (nested.get("code") or "").strip()
        if nested_code:
            return nested_code
    return ""

def _annotation_boxes_from_api_rows(rows: list) -> list:
    boxes = []
    for ann in rows or []:
        try:
            x1 = int(ann.get("x1"))
            y1 = int(ann.get("y1"))
            x2 = int(ann.get("x2"))
            y2 = int(ann.get("y2"))
        except Exception:
            continue
        boxes.append([x1, y1, x2, y2, _block_code_from_annotation_row(ann)])
    return boxes

def _safe_int_or_default(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _find_case_row_by_choice(cases_rows: list, case_choice: str) -> dict | None:
    if not case_choice or not isinstance(cases_rows, list):
        return None
    parts = [p.strip() for p in str(case_choice).split("|")]
    target_case_id = parts[1] if len(parts) >= 2 else ""
    for row in cases_rows:
        if _case_id_from_row(row) == target_case_id:
            return row
    return None

def _format_case_choice(row: dict) -> str:
    case_name = (row.get("case_name") or "case_sans_nom").strip()
    case_id = _case_id_from_row(row) or "id_manquant"
    doc_type = (row.get("doc_type") or "-").strip()
    source_ref = (row.get("source_ref") or "-").strip()
    return f"{case_name} | {case_id} | {doc_type} | {source_ref}"

def filter_existing_case_choices(search_query: str, cases_rows: list):
    query = (search_query or "").strip().lower()
    if not isinstance(cases_rows, list):
        return gr.update(choices=[], value=None), "Aucun cas existant trouve."

    filtered_rows = []
    for row in cases_rows:
        if not isinstance(row, dict):
            continue
        source_ref = str(row.get("source_ref") or "").strip().lower()
        case_id = _case_id_from_row(row).lower()
        if not query or query in source_ref or query in case_id:
            filtered_rows.append(row)

    choices = [_format_case_choice(r) for r in filtered_rows]
    if not choices:
        if query:
            return (
                gr.update(choices=[], value=None),
                f"Aucun cas trouve pour la recherche '{search_query}'.",
            )
        return gr.update(choices=[], value=None), "Aucun cas existant trouve."

    status = (
        f"{len(choices)} cas correspondant(s) a la recherche '{search_query}'."
        if query
        else f"{len(choices)} cas charges. Selectionnez un cas puis cliquez sur Charger."
    )
    return gr.update(choices=choices, value=choices[0]), status

def refresh_existing_cases(limit: int = 2000):
    rows = api_get("/cases", params={"limit": limit}) or []
    cases_rows = [r for r in rows if isinstance(r, dict)]
    choices = [_format_case_choice(r) for r in cases_rows]
    memoire_choices = [
        _format_case_choice(r)
        for r in cases_rows
        if (r.get("doc_type") or "").strip() == "memoire" and not (r.get("memoire_type_code") or "").strip()
    ]
    if not choices:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            [],
            gr.update(value=""),
            "Aucun cas existant trouve.",
            "Aucun memoire sans type trouve.",
        )
    return (
        gr.update(choices=choices, value=choices[0]),
        gr.update(choices=memoire_choices, value=memoire_choices[0] if memoire_choices else None),
        cases_rows,
        gr.update(value=""),
        f"{len(choices)} cas charges. Selectionnez un cas puis cliquez sur Charger.",
        (
            f"{len(memoire_choices)} memoire(s) sans type charges. "
            "Selectionnez un cas puis cliquez sur Charger."
            if memoire_choices
            else "Aucun memoire sans type trouve."
        ),
    )


def update_memoire_type_visibility(doc_type_value: str, current_memoire_type_code: str | None = None):
    is_memoire = (doc_type_value or "").strip() == "memoire"
    return gr.update(visible=is_memoire, value=current_memoire_type_code if is_memoire else None)


def update_save_buttons(is_existing_case: bool):
    return gr.update(visible=not is_existing_case), gr.update(visible=bool(is_existing_case))


def set_existing_case_mode():
    return True


def set_new_annotation_mode():
    return False


def normalize_memoire_type_code(doc_type: str, memoire_type_code: str | None):
    if (doc_type or "").strip() != "memoire":
        return None
    normalized = (memoire_type_code or "").strip()
    return normalized or None


def load_image_and_prepare(url: str):
    """Loads image and returns (numpy_array, status_msg, pil_image_for_state, [], [])."""
    if not url or not url.strip():
        return None, "Veuillez coller une URL d'image.", None, [], []
    try:
        img = fetch_image(url.strip())
        w, h = img.size
        # Return numpy array for display, PIL image for state
        img_array = np.array(img)
        return img_array, f"Image chargee {w}Ã—{h}", img, [], []
    except Exception as e:
        return None, f"Erreur de chargement de l'image : {e}", None, [], []


def parse_image_metadata_from_ref(image_ref: str):
    if not image_ref or not image_ref.strip():
        return None
    ref = image_ref.strip()
    if ref.startswith("minio://"):
        object_path = ref[len("minio://") :]
    else:
        parsed = urlparse(ref)
        if parsed.scheme and parsed.netloc:
            object_path = parsed.path.lstrip("/")
        else:
            object_path = ref.lstrip("/")
    object_path = object_path.replace("\\", "/")
    parts = [p for p in object_path.split("/") if p]
    if len(parts) < 5:
        return None
    bucket_name = parts[-5]
    raw_doc_type_segment = parts[-4].strip()
    raw_doc_type = raw_doc_type_segment.lower()
    if raw_doc_type == "theses":
        doc_type = "these"
    elif raw_doc_type == "memoires":
        doc_type = "memoire"
    else:
        doc_type = "other"
    collection_code = parts[-3]
    doc_id = parts[-2]
    filename = parts[-1]
    source_ref = "/".join([bucket_name, raw_doc_type_segment, collection_code, doc_id, filename])
    return doc_type, doc_id, collection_code, source_ref


def metadata_updates_from_image_ref(image_ref: str, auto_metadata: bool = True):
    if not auto_metadata:
        return gr.update(value=None), "", "", ""
    parsed = parse_image_metadata_from_ref(image_ref)
    if not parsed:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return parsed


def load_image_and_prepare_with_metadata(url: str, auto_metadata: bool):
    img_array, status, clean_img, anns, table = load_image_and_prepare(url)
    doc_type_val, doc_id_val, collection_code_val, source_ref_val = metadata_updates_from_image_ref(url, auto_metadata)
    return (
        img_array,
        status,
        clean_img,
        anns,
        table,
        doc_type_val,
        doc_id_val,
        collection_code_val,
        source_ref_val,
        None,
    )


def on_auto_metadata_toggle(auto_metadata: bool, image_url: str):
    doc_type_val, doc_id_val, collection_code_val, source_ref_val = metadata_updates_from_image_ref(
        image_url, auto_metadata
    )
    return doc_type_val, doc_id_val, collection_code_val, source_ref_val


def build_case_name(doc_id: str, page_no: int):
    if doc_id and doc_id.strip():
        return f"{doc_id.strip()}_p{int(page_no):04d}"
    return f"doc_unknown_p{int(page_no):04d}"

def infer_is_humatheque_from_collection_code(collection_code: str):
    code = (collection_code or "").strip().lower()
    return code not in {"theses.fr", "dumas"}

def load_existing_case(case_choice: str, campaign_choice: str, cases_rows: list):
    row = _find_case_row_by_choice(cases_rows, case_choice)
    if row is None:
        return (
            gr.skip(),
            "Cas introuvable dans la liste actuelle. Rafraichissez la liste des cas.",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            None,
        )

    case_id = _case_id_from_row(row)
    image_url_val = _build_image_url_from_case(row)
    if not image_url_val:
        return (
            gr.skip(),
            f"Cas charge mais image introuvable pour case_id={case_id}.",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            None,
        )

    img_array, img_status, clean_img, _, _ = load_image_and_prepare(image_url_val)
    if clean_img is None:
        return (
            gr.skip(),
            f"Echec chargement image du cas {case_id} : {img_status}",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            image_url_val,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            None,
        )

    campaign_id = ""
    if campaign_choice:
        try:
            campaign_id = campaign_choice.split("|")[-1].strip()
        except Exception:
            campaign_id = ""
    ann_params = {"case_id": case_id}
    if campaign_id:
        ann_params["campaign_id"] = campaign_id
    try:
        anns = api_get("/layout-annotations", params=ann_params) or []
    except Exception:
        anns = []
    boxes = _annotation_boxes_from_api_rows(anns)
    vis_img = draw_boxes_on_image(clean_img, boxes, None)

    return (
        vis_img if vis_img is not None else img_array,
        f"{img_status} | Cas charge: {case_id} | {len(boxes)} annotation(s)",
        clean_img,
        boxes,
        boxes,
        None,
        image_url_val,
        row.get("doc_type"),
        row.get("doc_id") or "",
        row.get("collection_code") or "",
        row.get("source_ref") or "",
        bool(row.get("is_humatheque")) if row.get("is_humatheque") is not None else False,
        _safe_int_or_default(row.get("page_no"), 1),
        _safe_int_or_default(row.get("year"), 2000),
        row.get("memoire_type_code"),
        row.get("notes") or "",
        case_id or None,
    )

def save_metadata_only(
    image_url: str,
    image_with_boxes: Image.Image,
    annotations: list,
    campaign_choice: str,
    case_id: str | None,
    doc_type: str,
    doc_id: str,
    page_no: int,
    year: int,
    source_ref: str,
    is_humatheque: bool,
    collection_code: str,
    memoire_type_code: str,
    notes: str,
):
    if not doc_id and source_ref:
        parsed = parse_image_metadata_from_ref(source_ref)
        if parsed:
            _, parsed_doc_id, _, _ = parsed
            doc_id = parsed_doc_id
    memoire_type_code = normalize_memoire_type_code(doc_type, memoire_type_code)

    case_name = build_case_name(doc_id, page_no)
    orig_w, orig_h = (None, None)
    if image_with_boxes is not None:
        orig_w, orig_h = image_with_boxes.size

    case_payload = {
        "case_id": case_id or None,
        "case_name": case_name,
        "doc_type": doc_type,
        "doc_id": doc_id or None,
        "page_no": int(page_no),
        "year": int(year) if year else None,
        "source_ref": source_ref or None,
        "image_uri": image_url or None,
        "image_sha256": None,
        "notes": notes or None,
        "is_humatheque": bool(is_humatheque) if is_humatheque is not None else None,
        "collection_code": (collection_code or None),
        "memoire_type_code": memoire_type_code,
        "image_width": int(orig_w) if orig_w else None,
        "image_height": int(orig_h) if orig_h else None,
    }
    try:
        case_id = api_post("/cases/upsert", case_payload)
    except Exception as e:
        return f"Erreur pendant la mise a jour des metadonnees : {e}", None

    rects = annotations if isinstance(annotations, list) else []
    if not campaign_choice:
        return f"Metadonnees mises a jour pour le cas '{case_name}'.", json.dumps({"case_id": case_id}, ensure_ascii=False, indent=2)

    try:
        campaign_id = campaign_choice.split("|")[-1].strip()
    except Exception:
        return "Format de campagne invalide.", None

    replace_payload = {
        "campaign_id": campaign_id,
        "case_id": case_id,
        "annotations": [
            {
                "block_code": block_code,
                "x1": int(x_min),
                "y1": int(y_min),
                "x2": int(x_max),
                "y2": int(y_max),
                "source": "manual",
                "notes": None,
            }
            for x_min, y_min, x_max, y_max, block_code in rects
            if block_code
        ],
    }
    try:
        saved = api_post("/layout-annotations/replace", replace_payload)
    except Exception as e:
        return f"Metadonnees mises a jour, mais echec de synchronisation des annotations : {e}", None

    try:
        anns = api_get("/layout-annotations", params={"campaign_id": campaign_id, "case_id": case_id})
    except Exception as e:
        return f"Metadonnees et annotations mises a jour ({saved}), mais echec de recuperation : {e}", case_id

    return (
        f"Cas '{case_name}' mis a jour ({saved} annotation(s) synchronisee(s)).",
        json.dumps(anns, ensure_ascii=False, indent=2),
    )


def save_annotations(
    image_url: str,
    image_with_boxes: Image.Image,
    annotations: list, # Now contains [x1, y1, x2, y2, block_code]
    campaign_choice: str,
    case_id: str | None,
    # block_choice: str, # No longer needed here, taken from individual annotations
    doc_type: str,
    doc_id: str,
    page_no: int,
    year: int,
    source_ref: str,
    is_humatheque: bool,
    collection_code: str,
    memoire_type_code: str,
    notes: str,
):
    """
    Save rectangles with their respective labels to DB.
    """

    if image_with_boxes is None:
        return "Aucune image chargee.", None

    if not campaign_choice:
        return "Aucune campagne selectionnee.", None

    # Parse campaign_id from choice
    try:
        campaign_id = campaign_choice.split("|")[-1].strip()
    except Exception:
        return "Format de campagne invalide.", None
    memoire_type_code = normalize_memoire_type_code(doc_type, memoire_type_code)

    case_name = build_case_name(doc_id, page_no)

    # Upsert case
    orig_w, orig_h = image_with_boxes.size

    # Upsert case
    case_payload = {
        "case_id": case_id or None,
        "case_name": case_name, "doc_type": doc_type, "doc_id": doc_id or None,
        "page_no": int(page_no), "year": int(year) if year else None,
        "source_ref": source_ref or None, "image_uri": image_url, "image_sha256": None, "notes": notes or None,
        "is_humatheque": bool(is_humatheque) if is_humatheque is not None else None,
        "collection_code": (collection_code or None),
        "memoire_type_code": memoire_type_code,
        "image_width": int(orig_w),
        "image_height": int(orig_h),
    }
    try:
        case_id = api_post("/cases/upsert", case_payload)
    except Exception as e:
        return f"Erreur pendant l'upsert du cas : {e}", None

    rects = annotations if isinstance(annotations, list) else []
    if not rects:
        return "Aucun rectangle detecte. Dessinez au moins une boite.", case_id
    sx, sy = 1.0, 1.0  # Assume no scaling for now

    replace_payload = {
        "campaign_id": campaign_id,
        "case_id": case_id,
        "annotations": [
            {
                "block_code": individual_block_code,
                "x1": int(x_min * sx),
                "y1": int(y_min * sy),
                "x2": int(x_max * sx),
                "y2": int(y_max * sy),
                "source": "manual",
                "notes": None,
            }
            for x_min, y_min, x_max, y_max, individual_block_code in rects
            if individual_block_code
        ],
    }
    errors = len(rects) - len(replace_payload["annotations"])
    try:
        saved = api_post("/layout-annotations/replace", replace_payload)
    except Exception as e:
        return f"Erreur pendant l'enregistrement des annotations : {e}", None

    try:
        anns = api_get("/layout-annotations", params={"campaign_id": campaign_id, "case_id": case_id})
    except Exception as e:
        return f"{saved} rectangle(s) enregistres, mais echec de recuperation des annotations : {e}", case_id

    msg = f"Enregistre {saved} rectangle(s) pour le cas '{case_name}'. ({errors} erreur(s))"
    if errors:
        msg += f" (Certaines operations ont echoue)"

    return msg, json.dumps(anns, ensure_ascii=False, indent=2)


def on_image_select(evt: gr.SelectData, pending_pt, boxes, clean_img, block_choice):
    """Handle click on input image to define boxes."""
    if clean_img is None: return gr.skip(), pending_pt, boxes, boxes

    x, y = evt.index
    if pending_pt is None:
        # First click: store pending point and redraw
        new_pending = (x, y)
        vis_img = draw_boxes_on_image(clean_img, boxes, new_pending)
        return vis_img, new_pending, boxes, boxes
    else:
        # Second click: complete the box
        x1, y1 = pending_pt
        x2, y2 = x, y
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        
        # Extract the code part from the block_choice (e.g., "title | <uuid>" -> "title")
        current_block_code = block_choice.split(" | ")[0].strip()

        # Store bbox with its block_code
        new_box_data = bbox + [current_block_code]
        new_boxes = boxes + [new_box_data]
        vis_img = draw_boxes_on_image(clean_img, new_boxes, None)
        return vis_img, None, new_boxes, new_boxes

def undo_last_box(pending_pt, boxes, clean_img):
    """Undo the last click or remove the last box."""
    if clean_img is None: return gr.update(), None, boxes, gr.update()

    if pending_pt is not None:
        vis_img = draw_boxes_on_image(clean_img, boxes, None)
        return vis_img, None, boxes, boxes
    if boxes:
        boxes.pop()
        vis_img = draw_boxes_on_image(clean_img, boxes, None)
        return vis_img, None, boxes, boxes
    return gr.update(), None, boxes, boxes

def clear_all_boxes(clean_img):
    if clean_img is None: return gr.update(), [], []
    # Convert PIL to numpy for display
    return np.array(clean_img), [], []

def select_box_row(evt: gr.SelectData, block_choice, boxes, clean_img):
    """Select a row in the dataframe and highlight the corresponding box in red.
    
    Note: This requires an image redraw to show the red highlight, which is a 
    necessary tradeoff for the important visual feedback of which box is selected.
    """
    if evt is None or evt.index is None:
        return gr.update(), boxes, boxes, gr.skip()
    row = int(evt.index[0])
    
    # We need to redraw to show the red highlight (visual feedback is worth it)
    if clean_img is None:
        return row, boxes, boxes, gr.skip()
    
    # Redraw image with the selected box highlighted in red
    vis_img = draw_boxes_on_image(clean_img, boxes, None, row)
    return row, boxes, boxes, vis_img

def update_row_label(row_index, block_choice, boxes):
    """Updates the block label for a single row in the dataframe."""
    if not boxes:
        return [], []
    if row_index is None:
        return boxes, boxes
    try:
        row_idx = int(row_index)
    except Exception:
        return boxes, boxes
    if row_idx < 0 or row_idx >= len(boxes):
        return boxes, boxes

    new_block_code = block_choice.split(" | ")[0].strip()
    updated_boxes = list(boxes)
    x1, y1, x2, y2 = updated_boxes[row_idx][0:4]
    updated_boxes[row_idx] = [x1, y1, x2, y2, new_block_code]
    return updated_boxes, updated_boxes

def update_row_label_and_redraw(row_index, block_choice, boxes, clean_img):
    """Update label and redraw only if there's an actual change."""
    if row_index is None or not boxes:
        return boxes, boxes, gr.skip()
    
    try:
        row_idx = int(row_index)
    except Exception:
        return boxes, boxes, gr.skip()
    
    if row_idx < 0 or row_idx >= len(boxes):
        return boxes, boxes, gr.skip()
    
    # Check if the label is actually changing
    new_block_code = block_choice.split(" | ")[0].strip()
    current_block_code = boxes[row_idx][4] if len(boxes[row_idx]) > 4 else None
    
    if current_block_code == new_block_code:
        # No change needed
        return boxes, boxes, gr.skip()
    
    # Actually update the label
    updated_boxes, _ = update_row_label(row_index, block_choice, boxes)
    
    if clean_img is None:
        return updated_boxes, updated_boxes, gr.skip()
    
    # Redraw with selection highlight
    selected_idx = int(row_index)
    vis_img = draw_boxes_on_image(clean_img, updated_boxes, None, selected_idx)
    return updated_boxes, updated_boxes, vis_img

# -----------------------
# UI
# -----------------------
def make_app():
    camp_choices, block_choices, _, _ = refresh_reference_data()

    custom_css = """
    .app-shell {max-width: 1320px; margin: 0 auto; padding: 10px 6px 18px;}
    .hero {
      border: 1px solid #d9e2ef;
      border-radius: 14px;
      padding: 14px 16px;
      background: linear-gradient(120deg, #f8fbff 0%, #f4f7ff 40%, #f8fbff 100%);
      margin-bottom: 10px;
    }
    .panel {
      border: 1px solid #dfe7f5;
      border-radius: 12px;
      background: #fbfdff;
      padding: 10px;
    }
    .highlight-accordion {
      border: 1px solid #f3d7b1;
      border-radius: 12px;
      background: #fffaf3;
      margin-bottom: 10px;
      overflow: hidden;
    }
    .highlight-accordion > div[role="button"] {
      background: #fff1dc;
      border-bottom: 1px solid #f3d7b1;
    }
    .highlight-accordion > div[role="button"] span {
      color: #9a5a12;
      font-weight: 600;
    }
    """
    if MINIO_ENABLED and minio_client is not None:
        try:
            excluded_source_refs = get_annotated_source_refs(limit=2000)
        except Exception:
            excluded_source_refs = set()
        sync_bucket_tree(BUCKET, TREE_ROOT, excluded_source_refs=excluded_source_refs)
    else:
        TREE_ROOT.mkdir(parents=True, exist_ok=True)
    
    with gr.Blocks(
        title="Annotation app",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="stone"),
        css=custom_css,
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                <div class="hero">
                  <h1 style="margin:0;">Application d'annotation de blocs</h1>
                  <p style="margin:6px 0 0 0;">
                    Objectif (phase préparatoire VLM) : sur la base d'un échantilon, obtenir des statistiques objectives sur la répartition formelle (localisation) de l'information bibliographique 
                    selon les types de documents (thèses vs mémoires), les années ou les disciplines.
                  </p>
                </div>
                """
            )
            with gr.Accordion("Guide d'utilisation", open=False, elem_classes=["highlight-accordion"]):
                gr.Markdown(
                    "- Choisissez un fichier dans l'arborescence MinIO (ou optionnellement renseignez une URL d'image).\n"
                    "- Pour reprendre un travail existant, utilisez **Annotations existantes** ; pour cibler les mémoires sans type, utilisez **Annotations existantes - Mémoires sans type**.\n"
                    "- Cliquez deux fois sur l'image pour créer une boite (coin 1 puis coin 2), puis sélectionnez une ligne du tableau pour assigner son label.\n"
                    "- Le champ **Type de mémoire** n'apparait que si **Type de document** vaut `memoire` ; pour les autres types il reste vide.\n"
                    "- Si vous chargez un cas existant, seul le bouton de mise à jour des métadonnées est affiché ; pour une nouvelle image, seul le bouton d'enregistrement de l'annotation est affiché."
                )

            # States
            clean_img_state = gr.State()
            pending_point_state = gr.State()
            annotations_state = gr.State([])
            existing_cases_state = gr.State([])
            is_existing_case_state = gr.State(False)
            current_case_id_state = gr.State(None)

            with gr.Accordion("Annotations existantes", open=False, elem_classes=["highlight-accordion"]):
                with gr.Row():
                    existing_case_search = gr.Textbox(
                        label="Filtrer par source_ref ou case_id",
                        placeholder="Ex. images/theses/... ou un case_id",
                        value="",
                        scale=3,
                    )
                    existing_case_choice = gr.Dropdown(
                        label="Cas deja enregistres",
                        choices=[],
                        value=None,
                        scale=4,
                    )
                    load_case_btn = gr.Button("Charger le cas", scale=1, variant="primary")
                    refresh_cases_btn = gr.Button("Rafraichir les cas", scale=1, variant="neutral")                   
                existing_cases_status = gr.Markdown("")
            with gr.Accordion("Annotations existantes - Mémoires sans type", open=False, elem_classes=["highlight-accordion"]):
                with gr.Row():
                    memoire_case_choice = gr.Dropdown(
                        label="Memoires sans type",
                        choices=[],
                        value=None,
                        scale=6,
                    )
                    load_memoire_case_btn = gr.Button("Charger le cas", scale=1, variant="primary")
                    refresh_memoire_cases_btn = gr.Button("Rafraichir les cas", scale=1, variant="neutral")                    
                memoire_cases_status = gr.Markdown("")

            with gr.Row():
                image_url = gr.Textbox(
                    label="URL de l'image",
                    placeholder="https://minio.smartbiblia.fr/test/2014PA131046/p1.png",
                    value="",
                    scale=5,
                )
                load_btn = gr.Button("Charger l'image", scale=1, variant="primary")

            status = gr.Markdown("")

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["panel"]):
                    gr.Markdown("## Fichiers du bucket MinIO")
                    refresh_tree_btn = gr.Button("Rafraichir l'arborescence")
                    tree_status = gr.Markdown("")
                    file_explorer = gr.FileExplorer(
                        root_dir=str(TREE_ROOT),
                        glob="**/*",
                        file_count="single",
                        interactive=True,
                        label="Fichiers du bucket (vue arborescente)",
                    )

                with gr.Column(scale=5, elem_classes=["panel"]):
                    img = gr.Image(
                        label="Zone de dessin (cliquez deux fois pour tracer un rectangle)",
                        type="numpy",
                        interactive=True,
                    )
                    with gr.Row():
                        undo_btn = gr.Button("Annuler le dernier point/rectangle")
                        clear_btn = gr.Button("Effacer tous les rectangles")

                with gr.Column(scale=4, elem_classes=["panel"]):
                    gr.Markdown("## Gestion des annotations de boites")
                    with gr.Row():
                        with gr.Column(scale=5, min_width=520):
                            box_display = gr.Dataframe(
                                headers=["x1", "y1", "x2", "y2", "Label du bloc"],
                                label="Coordonnees des boites",
                                value=[],
                                interactive=False,
                                column_widths=[72, 72, 72, 72, 160],
                            )
                        with gr.Column(scale=2, min_width=220):
                            block = gr.Dropdown(
                                label="Label de la boite selectionnee",
                                choices=block_choices,
                                value=block_choices[0],
                            )
                            selected_row = gr.Number(
                                label="Ligne selectionnee",
                                value=0,
                                precision=0,
                            )
                            gr.Markdown(
                                "- Selectionnez une ligne dans le tableau.\n"
                                "- Modifiez son label via la liste deroulante.\n"
                                "- Les nouvelles boites utilisent le label courant."
                            )

            with gr.Accordion("Metadonnees", open=True, elem_classes=["highlight-accordion"]):
                gr.Markdown("## Métadonnées")
                campaign = gr.Dropdown(
                    label="Campagne",
                    choices=camp_choices,
                    value=camp_choices[0],
                )
                auto_metadata = gr.Checkbox(
                    label="Remplir automatiquement les metadonnees depuis le chemin de l'image",
                    value=True,
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Source du document / 1")
                        with gr.Row():
                            doc_type = gr.Dropdown(
                                label="Type de document",
                                choices=["these", "memoire", "other"],
                                value=None,
                            )
                            doc_id = gr.Textbox(
                                label="Identifiant du document",
                                placeholder="123456789",
                            )                       
                        with gr.Row():
                            year = gr.Number(label="Annee", value=2000, precision=0)
                            page_no = gr.Number(label="Numero de page", value=1, precision=0)
                        is_humatheque = gr.Checkbox(label="Humatheque ?", value=False)
                        with gr.Row():
                            collection_code = gr.Textbox(
                                label="Code de collection",
                                placeholder="Ex : EPHE / theses.fr ...",
                            )
                            memoire_type_code = gr.Dropdown(
                                label="Type de mémoire",
                                choices=[(label, code) for code, label in MEMOIRE_TYPE_OPTIONS.items()],
                                value=None,
                                visible=False,
                            )

                    with gr.Column():
                        gr.Markdown("#### Source du document / 2")                       
                        source_ref = gr.Textbox(
                            label="Reference source",
                            placeholder="thesis/123456789/p0001.png",
                        )  
                        gr.Markdown("#### Notes d'annotation")
                        notes = gr.Textbox(
                            label="Notes",
                            lines=8,
                            placeholder="Notes complementaires sur cette annotation...",
                        )

            with gr.Row():
                save_btn = gr.Button("Enregistrer une nouvelle annotation", variant="primary", visible=True)
                save_meta_btn = gr.Button("Mettre à jour ce cas", variant="primary", visible=False)

            with gr.Accordion("Reponse API", open=False, elem_classes=["highlight-accordion"]):
                out_msg = gr.Textbox(label="Statut", lines=2, interactive=False)
                out_json = gr.Code(label="Annotations enregistrees (JSON)", language="json")

        # Actions
        demo.load(
            fn=refresh_tree,
            outputs=[file_explorer, tree_status],
        )
        
        demo.load(
            fn=refresh_existing_cases,
            outputs=[
                existing_case_choice,
                memoire_case_choice,
                existing_cases_state,
                existing_case_search,
                existing_cases_status,
                memoire_cases_status,
            ],
        )

        refresh_tree_btn.click(
            fn=refresh_tree,
            outputs=[file_explorer, tree_status],
        )
        
        refresh_cases_btn.click(
            fn=refresh_existing_cases,
            outputs=[
                existing_case_choice,
                memoire_case_choice,
                existing_cases_state,
                existing_case_search,
                existing_cases_status,
                memoire_cases_status,
            ],
        )

        refresh_memoire_cases_btn.click(
            fn=refresh_existing_cases,
            outputs=[
                existing_case_choice,
                memoire_case_choice,
                existing_cases_state,
                existing_case_search,
                existing_cases_status,
                memoire_cases_status,
            ],
        )

        existing_case_search.input(
            fn=filter_existing_case_choices,
            inputs=[existing_case_search, existing_cases_state],
            outputs=[existing_case_choice, existing_cases_status],
        )

        load_case_btn.click(
            fn=load_existing_case,
            inputs=[existing_case_choice, campaign, existing_cases_state],
            outputs=[
                img,
                status,
                clean_img_state,
                annotations_state,
                box_display,
                pending_point_state,
                image_url,
                doc_type,
                doc_id,
                collection_code,
                source_ref,
                is_humatheque,
                page_no,
                year,
                memoire_type_code,
                notes,
                current_case_id_state,
            ],
        ).then(
            fn=set_existing_case_mode,
            outputs=[is_existing_case_state],
        ).then(
            fn=update_save_buttons,
            inputs=[is_existing_case_state],
            outputs=[save_btn, save_meta_btn],
        ).then(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )

        load_memoire_case_btn.click(
            fn=load_existing_case,
            inputs=[memoire_case_choice, campaign, existing_cases_state],
            outputs=[
                img,
                status,
                clean_img_state,
                annotations_state,
                box_display,
                pending_point_state,
                image_url,
                doc_type,
                doc_id,
                collection_code,
                source_ref,
                is_humatheque,
                page_no,
                year,
                memoire_type_code,
                notes,
                current_case_id_state,
            ],
        ).then(
            fn=set_existing_case_mode,
            outputs=[is_existing_case_state],
        ).then(
            fn=update_save_buttons,
            inputs=[is_existing_case_state],
            outputs=[save_btn, save_meta_btn],
        ).then(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )

        file_explorer.change(
            fn=load_image_from_tree,
            inputs=[file_explorer, auto_metadata],
            outputs=[
                img,
                status,
                clean_img_state,
                annotations_state,
                box_display,
                pending_point_state,
                image_url,
                doc_type,
                doc_id,
                collection_code,
                source_ref,
                current_case_id_state,
            ],
        ).then(
            fn=set_new_annotation_mode,
            outputs=[is_existing_case_state],
        ).then(
            fn=update_save_buttons,
            inputs=[is_existing_case_state],
            outputs=[save_btn, save_meta_btn],
        ).then(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )
        
        file_explorer.select(
            fn=load_image_from_tree,
            inputs=[auto_metadata],
            outputs=[
                img,
                status,
                clean_img_state,
                annotations_state,
                box_display,
                pending_point_state,
                image_url,
                doc_type,
                doc_id,
                collection_code,
                source_ref,
                current_case_id_state,
            ],
        ).then(
            fn=set_new_annotation_mode,
            outputs=[is_existing_case_state],
        ).then(
            fn=update_save_buttons,
            inputs=[is_existing_case_state],
            outputs=[save_btn, save_meta_btn],
        ).then(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )

        load_btn.click(
            fn=load_image_and_prepare_with_metadata,
            inputs=[image_url, auto_metadata],
            outputs=[
                img,
                status,
                clean_img_state,
                annotations_state,
                box_display,
                doc_type,
                doc_id,
                collection_code,
                source_ref,
                current_case_id_state,
            ],
        ).then(
            fn=set_new_annotation_mode,
            outputs=[is_existing_case_state],
        ).then(
            fn=update_save_buttons,
            inputs=[is_existing_case_state],
            outputs=[save_btn, save_meta_btn],
        ).then(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )

        auto_metadata.change(
            fn=on_auto_metadata_toggle,
            inputs=[auto_metadata, image_url],
            outputs=[doc_type, doc_id, collection_code, source_ref],
        ).then(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )

        doc_type.change(
            fn=update_memoire_type_visibility,
            inputs=[doc_type, memoire_type_code],
            outputs=[memoire_type_code],
        )
        
        collection_code.change(
            fn=infer_is_humatheque_from_collection_code,
            inputs=[collection_code],
            outputs=[is_humatheque],
        )

        img.select(
            fn=on_image_select,
            inputs=[pending_point_state, annotations_state, clean_img_state, block], # Added 'block' input
            outputs=[img, pending_point_state, annotations_state, box_display]
        )

        undo_btn.click(
            fn=undo_last_box,
            inputs=[pending_point_state, annotations_state, clean_img_state],
            outputs=[img, pending_point_state, annotations_state, box_display]
        )

        clear_btn.click(
            fn=clear_all_boxes,
            inputs=[clean_img_state],
            outputs=[img, annotations_state, box_display]
        )

        box_display.select(
            fn=select_box_row,
            inputs=[block, annotations_state, clean_img_state],
            outputs=[selected_row, annotations_state, box_display, img]
        )

        block.change(
            fn=update_row_label_and_redraw,
            inputs=[selected_row, block, annotations_state, clean_img_state],
            outputs=[annotations_state, box_display, img]
        )


        save_btn.click(
            fn=save_annotations,
            inputs=[
                image_url, clean_img_state, annotations_state, campaign, current_case_id_state, # Use clean_img_state instead of img
                doc_type, doc_id, page_no, year, source_ref, is_humatheque, collection_code, memoire_type_code, notes
            ],
            outputs=[out_msg, out_json],
        )
        
        save_meta_btn.click(
            fn=save_metadata_only,
            inputs=[
                image_url, clean_img_state, annotations_state, campaign, current_case_id_state,
                doc_type, doc_id, page_no, year, source_ref, is_humatheque, collection_code, memoire_type_code, notes
            ],
            outputs=[out_msg, out_json],
        )

    return demo

if __name__ == "__main__":
    demo = make_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, css=".gradio-container {background-color: #e0e8ec;}")

