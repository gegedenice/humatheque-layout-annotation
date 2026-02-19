#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gradio>=4.0.0",
#   "requests",
#   "pillow",
#   "python-dotenv",
#   "numpy"
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

BUCKET = "images"
LOCAL_ROOT = Path("./.bucket_cache") / BUCKET 

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

def sync_bucket(bucket: str, local_root: Path):                                                                                                                                                             
    local_root.mkdir(parents=True, exist_ok=True)                                                                                                                                                           
    for obj in minio_client.list_objects(bucket, recursive=True):                                                                                                                                           
        dst = local_root / obj.object_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        minio_client.fget_object(bucket, obj.object_name, str(dst))

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
        return gr.update(value=[]), "MinIO is not configured in .env."
    try:
        sync_bucket(BUCKET, LOCAL_ROOT)
        refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return gr.update(value=[]), f"Last refresh: {refreshed_at}"
    except Exception as e:
        return gr.update(value=[]), f"Refresh error: {e}"

def load_image_from_tree(selected_path):
    if not selected_path:
        return [gr.skip()] * 7
    if isinstance(selected_path, list):
        if not selected_path:
            return [gr.skip()] * 7
        selected_path = selected_path[0]

    selected_str = str(selected_path).strip()
    if not selected_str:
        return [gr.skip()] * 7

    local_path = Path(selected_str)
    if not local_path.is_absolute():
        local_path = (Path.cwd() / local_path).resolve()

    if local_path.is_dir():
        return [gr.skip()] * 7

    if local_path.exists():
        try:
            loaded_img = Image.open(local_path).convert("RGB")
            w, h = loaded_img.size
            object_name = local_path.name
            try:
                object_name = local_path.relative_to(LOCAL_ROOT.resolve()).as_posix()
            except Exception:
                posix_path = local_path.as_posix()
                marker = f"/{BUCKET}/"
                if marker in posix_path:
                    object_name = posix_path.split(marker, 1)[1]
            status = f"Loaded local image {w}x{h}: {local_path}"
            image_uri = f"minio://{BUCKET_NAME}/{object_name}"
            # Return numpy array for display, PIL for state
            img_array = np.array(loaded_img)
            return img_array, status, loaded_img, [], [], None, image_uri
        except Exception as e:
            return gr.skip(), f"Error loading local image: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    normalized = selected_str.replace("\\", "/")
    marker = f"/{BUCKET}/"
    if marker in normalized:
        object_name = normalized.split(marker, 1)[1]
    else:
        object_name = normalized.lstrip("./")
        local_root_norm = str(LOCAL_ROOT).replace("\\", "/").lstrip("./")
        if object_name.startswith(local_root_norm + "/"):
            object_name = object_name[len(local_root_norm) + 1 :]

    if object_name.startswith(f"{BUCKET_NAME}/"):
        object_name = object_name[len(BUCKET_NAME) + 1 :]

    if not object_name or not MINIO_ENABLED or minio_client is None:
        return [gr.skip()] * 7

    try:
        loaded_img = fetch_minio_image(object_name)
        w, h = loaded_img.size
        status = f"Loaded MinIO image {w}x{h}: {object_name}"
        image_uri = f"minio://{BUCKET_NAME}/{object_name}"
        # Return numpy array for display, PIL for state
        img_array = np.array(loaded_img)
        return img_array, status, loaded_img, [], [], None, image_uri
    except Exception as e:
        return gr.skip(), f"Error loading MinIO image: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

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


def load_image_and_prepare(url: str):
    """Loads image and returns (numpy_array, status_msg, pil_image_for_state, [], [])."""
    if not url or not url.strip():
        return None, "Please paste an image URL.", None, [], []
    try:
        img = fetch_image(url.strip())
        w, h = img.size
        # Return numpy array for display, PIL image for state
        img_array = np.array(img)
        return img_array, f"Loaded image {w}×{h}", img, [], []
    except Exception as e:
        return None, f"Error loading image: {e}", None, [], []


def build_case_name(doc_id: str, page_no: int):
    if doc_id and doc_id.strip():
        return f"{doc_id.strip()}_p{int(page_no):04d}"
    return f"doc_unknown_p{int(page_no):04d}"


def save_annotations(
    image_url: str,
    image_with_boxes: Image.Image,
    annotations: list, # Now contains [x1, y1, x2, y2, block_code]
    campaign_choice: str,
    # block_choice: str, # No longer needed here, taken from individual annotations
    doc_type: str,
    doc_id: str,
    page_no: int,
    year: int,
    language: str,
    source_ref: str,
    is_humatheque: bool,
    collection_code: str,
    annotator: str,
    notes: str,
):
    """
    Save rectangles with their respective labels to DB.
    """

    if image_with_boxes is None:
        return "No image loaded.", None

    if not campaign_choice:
        return "No campaign selected.", None

    # Parse campaign_id from choice
    try:
        campaign_id = campaign_choice.split("|")[-1].strip()
    except Exception:
        return "Invalid campaign format.", None

    case_name = build_case_name(doc_id, page_no)

    # Upsert case
    orig_w, orig_h = image_with_boxes.size

    # Upsert case
    case_payload = {
        "case_name": case_name, "doc_type": doc_type, "doc_id": doc_id or None,
        "page_no": int(page_no), "year": int(year) if year else None, "language": language or None,
        "source_ref": source_ref or None, "image_uri": image_url, "image_sha256": None, "notes": notes or None,
        "is_humatheque": bool(is_humatheque) if is_humatheque is not None else None,
        "collection_code": (collection_code or None),
        "image_width": int(orig_w),
        "image_height": int(orig_h),
    }
    try:
        case_id = api_post("/cases/upsert", case_payload)
    except Exception as e:
        return f"Error upserting case: {e}", None

    rects = annotations if isinstance(annotations, list) else []
    if not rects:
        return "No rectangles found. Draw at least one bbox.", case_id
    sx, sy = 1.0, 1.0  # Assume no scaling for now

    saved, errors = 0, 0
    # Iterate through each annotation, which now includes its block_code
    for box_data in rects:
        x_min, y_min, x_max, y_max, individual_block_code = box_data
        
        # Ensure individual_block_code is not empty
        if not individual_block_code:
            errors += 1
            continue

        payload = {
            "campaign_id": campaign_id,
            "case_id": case_id,
            "block_code": individual_block_code, # Use individual_block_code for this box
            "x1": int(x_min * sx),
            "y1": int(y_min * sy),
            "x2": int(x_max * sx),
            "y2": int(y_max * sy),
            "source": "manual",
            "annotator": annotator or None,
            "notes": None, # Case notes are global, not per-annotation
        }
        try:
            api_post("/layout-annotations", payload)
            saved += 1
        except Exception:
            errors += 1

    try:
        anns = api_get("/layout-annotations", params={"campaign_id": campaign_id, "case_id": case_id})
    except Exception as e:
        return f"Saved {saved} rect(s), but failed to fetch annotations: {e}", case_id

    msg = f"âœ… Saved {saved} rectangle(s) for case '{case_name}'. ({errors} error(s) occurred)"
    if errors:
        msg += f" (Some errors occurred)"

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
    """
    sync_bucket(BUCKET, LOCAL_ROOT)
    
    with gr.Blocks(
        title="WP1 Layout - bbox annotation",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=custom_css,
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                <div class="hero">
                  <h2 style="margin:0;">WP1 Layout Annotation</h2>
                  <p style="margin:6px 0 0 0;">
                    Load an image URL, draw bounding boxes with two clicks, assign block labels,
                    and save annotations through the API.
                  </p>
                </div>
                """
            )

            # States
            clean_img_state = gr.State()
            pending_point_state = gr.State()
            annotations_state = gr.State([])

            with gr.Row():
                image_url = gr.Textbox(
                    label="Image URL",
                    placeholder="https://minio.smartbiblia.fr/images/.../p1.png",
                    scale=5,
                )
                load_btn = gr.Button("Load image", scale=1, variant="primary")

            status = gr.Markdown("")

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["panel"]):
                    gr.Markdown("### MinIO Bucket Files")
                    refresh_tree_btn = gr.Button("Refresh tree")
                    tree_status = gr.Markdown("")
                    file_explorer = gr.FileExplorer(
                        root_dir=str(LOCAL_ROOT),
                        glob="**/*",
                        file_count="single",
                        interactive=True,
                        label="Bucket Files (tree view)",
                    )

                with gr.Column(scale=5, elem_classes=["panel"]):
                    img = gr.Image(
                        label="Canvas (click twice to draw a rectangle)",
                        type="numpy",
                        interactive=True,
                    )
                    with gr.Row():
                        undo_btn = gr.Button("Undo last point/box")
                        clear_btn = gr.Button("Clear all boxes")

                with gr.Column(scale=4, elem_classes=["panel"]):
                    gr.Markdown("### Box annotation manager")
                    with gr.Row():
                        with gr.Column(scale=5, min_width=520):
                            box_display = gr.Dataframe(
                                headers=["x1", "y1", "x2", "y2", "Block Label"],
                                label="Box coordinates",
                                interactive=False,
                                column_widths=[72, 72, 72, 72, 160],
                            )
                        with gr.Column(scale=2, min_width=220):
                            block = gr.Dropdown(
                                label="Label for selected box",
                                choices=block_choices,
                                value=block_choices[0],
                            )
                            selected_row = gr.Number(
                                label="Selected row",
                                value=0,
                                precision=0,
                            )
                            gr.Markdown(
                                "- Select a row in the table.\n"
                                "- Change its label with the dropdown.\n"
                                "- New boxes use the current dropdown label."
                            )

            with gr.Accordion("Metadata and save options", open=True):
                with gr.Row():
                    campaign = gr.Dropdown(
                        label="Campaign",
                        choices=camp_choices,
                        value=camp_choices[0],
                    )
                    annotator = gr.Textbox(
                        label="Annotator",
                        value=os.getenv("USER", "ggeoffroy"),
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Document source/1")
                        with gr.Row():
                            doc_type = gr.Dropdown(
                                label="Document type",
                                choices=["these", "memoire", "other"],
                                value="these",
                            )
                            doc_id = gr.Textbox(
                                label="Document ID (PPN recommended)",
                                placeholder="123456789",
                            )
                        page_no = gr.Number(label="Page number", value=1, precision=0)
                        with gr.Row():
                            year = gr.Number(label="Year", value=2000, precision=0)
                            language = gr.Textbox(label="Language", value="fr")
                        source_ref = gr.Textbox(
                            label="Source ref (optional)",
                            placeholder="thesis/123456789/p0001.png",
                        )                        

                    with gr.Column():
                        gr.Markdown("#### Document source/2")
                        is_humatheque = gr.Checkbox(label="Humathèque ?", value=False)
                        collection_code = gr.Textbox(
                            label="Collection code (optional)",
                            placeholder="Ex: HUM-ARCH / HUM-THESES ...",
                        )
                        gr.Markdown("#### Annotation notes")
                        notes = gr.Textbox(
                            label="Notes",
                            lines=8,
                            placeholder="Any notes about this annotation...",
                        )

            save_btn = gr.Button("Save all drawn rectangles", variant="primary")

            with gr.Accordion("API response", open=False):
                out_msg = gr.Textbox(label="Status", lines=2, interactive=False)
                out_json = gr.Code(label="Stored annotations (JSON)", language="json")

        # Actions
        demo.load(
            fn=refresh_tree,
            outputs=[file_explorer, tree_status],
        )

        refresh_tree_btn.click(
            fn=refresh_tree,
            outputs=[file_explorer, tree_status],
        )

        file_explorer.change(
            fn=load_image_from_tree,
            inputs=[file_explorer],
            outputs=[img, status, clean_img_state, annotations_state, box_display, pending_point_state, image_url],
        )
        
        file_explorer.select(
            fn=load_image_from_tree,
            inputs=[file_explorer],
            outputs=[img, status, clean_img_state, annotations_state, box_display, pending_point_state, image_url],
        )

        load_btn.click(
            fn=load_image_and_prepare,
            inputs=[image_url],
            outputs=[img, status, clean_img_state, annotations_state, box_display],
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
                image_url, clean_img_state, annotations_state, campaign, # Use clean_img_state instead of img
                doc_type, doc_id, page_no, year, language, source_ref, is_humatheque, collection_code, annotator, notes
            ],
            outputs=[out_msg, out_json],
        )

    return demo

if __name__ == "__main__":
    demo = make_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)

