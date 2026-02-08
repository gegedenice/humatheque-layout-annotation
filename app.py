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
import requests
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY")

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

def draw_boxes_on_image(image: Image.Image, boxes, pending_point=None, selected_index=None):
    """Helper to draw boxes and pending point on image."""
    if image is None: return None
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

    return out_img


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
    """Loads image and returns (pil_image, width, height)."""
    if not url or not url.strip():
        return None, "Please paste an image URL.", None, [], []
    try:
        img = fetch_image(url.strip())
        w, h = img.size
        # Return the clean image for display, and also store it in a state for redrawing
        return img, f"Loaded image {w}×{h}", img, [], []
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
    case_payload = {
        "case_name": case_name, "doc_type": doc_type, "doc_id": doc_id or None,
        "page_no": int(page_no), "year": int(year) if year else None, "language": language or None,
        "source_ref": source_ref or None, "image_uri": image_url, "image_sha256": None, "notes": notes or None,
    }
    try:
        case_id = api_post("/cases/upsert", case_payload)
    except Exception as e:
        return f"Error upserting case: {e}", None

    rects = annotations if isinstance(annotations, list) else []
    if not rects:
        return "No rectangles found. Draw at least one bbox.", case_id

    orig_w, orig_h = image_with_boxes.size
    sx, sy = 1.0, 1.0 # Assume no scaling for now

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

    msg = f"✅ Saved {saved} rectangle(s) for case '{case_name}'. ({errors} error(s) occurred)"
    if errors:
        msg += f" (Some errors occurred)"

    return msg, json.dumps(anns, ensure_ascii=False, indent=2)


def on_image_select(evt: gr.SelectData, pending_pt, boxes, clean_img, block_choice):
    """Handle click on input image to define boxes."""
    if clean_img is None: return gr.update(), pending_pt, boxes, gr.update()

    x, y = evt.index
    if pending_pt is None:
        new_pending = (x, y)
        vis_img = draw_boxes_on_image(clean_img, boxes, new_pending)
        return vis_img, new_pending, boxes, boxes
    else:
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
    return clean_img, [], []

def select_box_row(evt: gr.SelectData, block_choice, boxes, clean_img):
    if evt is None or evt.index is None:
        return gr.update(), boxes, boxes, gr.update()
    row = int(evt.index[0])
    updated_boxes, _ = update_row_label(row, block_choice, boxes)
    vis_img = draw_boxes_on_image(clean_img, updated_boxes, None, row)
    return row, updated_boxes, updated_boxes, vis_img

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
    updated_boxes, _ = update_row_label(row_index, block_choice, boxes)
    if clean_img is None:
        return updated_boxes, updated_boxes, gr.update()
    selected_idx = int(row_index) if row_index is not None else None
    vis_img = draw_boxes_on_image(clean_img, updated_boxes, None, selected_idx)
    return updated_boxes, updated_boxes, vis_img

# -----------------------
# UI
# -----------------------
def make_app():
    camp_choices, block_choices, _, _ = refresh_reference_data()

    with gr.Blocks(title="WP1 Layout — bbox annotation", theme=gr.themes.Default(primary_hue="blue", secondary_hue="neutral")) as demo:
        gr.Markdown("# WP1 Layout — Annotation de blocs (bboxes)")
        gr.Markdown("Load an image by url, draw boxes, select a label, then save in Postgres Database by API middleware.")

        # States
        clean_img_state = gr.State()
        pending_point_state = gr.State()
        annotations_state = gr.State([])

        with gr.Row():
            image_url = gr.Textbox(label="Image URL", placeholder="https://raw.githubusercontent.com/.../123456789/p0001.png", scale=4)
            load_btn = gr.Button("Load", scale=1)

        status = gr.Markdown("")
        
        img = gr.Image(label="Image (click twice to draw a rectangle)", type="pil", interactive=True)

        with gr.Row():
            undo_btn = gr.Button("Undo last point/box")
            clear_btn = gr.Button("Clear all boxes")
        
        box_display = gr.Dataframe(
            headers=["x1", "y1", "x2", "y2", "Block Label"],
            label="Drawn Boxes Coordinates",
            interactive=False,
        )

        gr.Markdown(
            "### How to Annotate\n" 
            "- **Workflow**: Choose a **label** from the dropdown below, click **twice** on the image to draw a bounding box, repeat to add more boxes.\n" 
            "- **Undo** removes the last point or the last completed box. **Clear** erases all boxes on the image.\n" 
            "- The coordinates of the drawn boxes, along with their associated block label, appear in the table above.\n" 
            "- After saving, you can clear the boxes with **Clear** to start a new batch with a different label."
        )

        with gr.Accordion("Metadata (Annotation Details)", open=True):
            with gr.Row():
                with gr.Column(scale=2): # Campaign and Block are main classification, give them more space
                    campaign = gr.Dropdown(label="Campaign", choices=camp_choices, value=camp_choices[0])
                with gr.Column(scale=1):
                    block = gr.Dropdown(label="Block label", choices=block_choices, value=block_choices[0])

            with gr.Row():
                selected_row = gr.Number(label="Selected row index", value=0, precision=0)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Document Information")
                    doc_type = gr.Dropdown(label="doc_type", choices=["these", "memoire", "other"], value="these")
                    doc_id = gr.Textbox(label="doc_id (PPN recommended)", placeholder="123456789")
                    page_no = gr.Number(label="page_no", value=1, precision=0)
                    year = gr.Number(label="year", value=2000, precision=0)
                    language = gr.Textbox(label="language", value="fr")
                    source_ref = gr.Textbox(label="source_ref (optional)", placeholder="thesis/123456789/p0001.png")

                with gr.Column():
                    gr.Markdown("#### Annotation Details")
                    annotator = gr.Textbox(label="annotator", value=os.getenv("USER", "ggeoffroy"))
                    notes = gr.Textbox(label="notes", lines=4, placeholder="Any notes about this annotation...")

        save_btn = gr.Button("✅ Save all drawn rectangles", variant="primary")
        
        with gr.Accordion("API Response", open=False):
            out_msg = gr.Textbox(label="Status", lines=2, interactive=False)
            out_json = gr.Code(label="Stored annotations (JSON)", language="json")

        # Actions
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
                image_url, img, annotations_state, campaign, # block is now in annotations_state
                doc_type, doc_id, page_no, year, language, source_ref, annotator, notes
            ],
            outputs=[out_msg, out_json],
        )

    return demo

if __name__ == "__main__":
    demo = make_app()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
