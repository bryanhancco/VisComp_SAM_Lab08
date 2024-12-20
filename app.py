import gradio as gr
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw, Image
from utils.tools import box_prompt, format_results, point_prompt
from utils.tools_gradio import fast_process
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga del modelo preentrenado
sam_checkpoint = "./weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

# Description
title = "<center><strong><font size='8'>Ejemplo aplicativo de MobileSAM<font></strong></center>"

fotos = [
    ["assets/picture3.jpg"],
    ["assets/picture4.jpg"],
    ["assets/picture5.jpg"],
    ["assets/picture6.jpg"],
    ["assets/picture1.jpg"],
    ["assets/picture2.jpg"],
]

videos = [
    ["assets/video1.mp4"],
]

default_foto = fotos[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

global_bboxes = []
global_bbox_label = []

def obtener_fotograma(video_path, position):
    cap = cv2.VideoCapture(video_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(frame_rgb)

def obtener_duracion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  
    duration = total_frames / fps 
    
    cap.release()
    
    return total_frames, duration

@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    return fig, image

def segment_with_points(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    nd_image = np.array(image)
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    return fig, image

def segment_with_box(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
    bounding_box=None 
):
    global global_bboxes
    print("Geeeee", global_bboxes)
    
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    # Verificar si el bounding box ha sido seleccionado
    if len(global_bboxes) < 2:
        print("No bounding box provided")
        return image, image

    # Escalar los puntos del bounding box
    scaled_bbox = [int(coord * scale) for coord in global_bboxes[0] + global_bboxes[1]]
    
    nd_image = np.array(image)
    predictor.set_image(nd_image)

    # Realizar la predicción de la segmentación usando el rectángulo
    masks, scores, logits = predictor.predict(
        box=np.array([scaled_bbox]), 
        multimask_output=True,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = box_prompt(
        results, scaled_bbox, new_h, new_w  
    )
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    return fig, image

def get_points_with_draw(image, label, evt: gr.SelectData):    
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == "Agregar" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Agregar" else 0)

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image

def get_box_with_draw(image, label, evt: gr.SelectData):
    global global_bboxes, global_bbox_label

    x, y = evt.index[0], evt.index[1]

    point_radius, point_color = 15, (255, 255, 0) if label == "Encuadrar" else (255, 0, 255)

    first_point = None
    second_point = None

    print("Gaaaaaaa", global_bboxes)

    global_bboxes.append([x, y])
    global_bbox_label.append(1 if label == "Encuadrar" else 0)
    
    if len(global_bboxes) >= 2:
        first_point = global_bboxes[-2]
        second_point = global_bboxes[-1]

        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [tuple(first_point), tuple(second_point)],
            outline=point_color,
            width=3,
        )

    if first_point is not None:
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [(first_point[0] - point_radius, first_point[1] - point_radius), 
             (first_point[0] + point_radius, first_point[1] + point_radius)],
            fill=point_color,
        )

    if second_point is not None:
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [(second_point[0] - point_radius, second_point[1] - point_radius),
             (second_point[0] + point_radius, second_point[1] + point_radius)],
            fill=point_color,
        )
        
    print("GAASADASA", global_bboxes)

    return image

def get_options(image, label, evt: gr.SelectData):
    if label == "Agregar":
        return get_points_with_draw(image, label, evt)
    elif label == "Encuadrar":
        return get_box_with_draw(image, label, evt)
    elif label == "Segmentar todo": 
        return 
  
def segment_options(image, label):
    if label == "Agregar":
        return segment_with_points(image)
    elif label == "Encuadrar":
        return segment_with_box(image)
    elif label == "Segmentar todo": 
        return segment_everything(image)
    
def actualizar_slider(video_path):
    total_frames, duration = obtener_duracion_video(video_path)
    return gr.update(maximum=total_frames, step=1)

img_inicial = gr.Image(label="Imagen inicial", value=default_foto[0], type="pil")
img_segmentada = gr.Image(label="Imagen segmentada", interactive=False, type="pil")

vid_inicial_point = gr.Video(label="Video inicial", sources=["upload"], format="mp4", include_audio=False) 
fotograma = gr.Image(label="Fotograma", type="pil")
fotograma_segmentado = gr.Image(label="Fotograma segmentada", interactive=False, type="pil")

vid_inicial_frame = gr.Video(label="Video inicial", sources=["upload"], format="mp4", include_audio=False) 
vid_segmentado_frame = gr.Video(label="Video Segmentado", format="mp4", include_audio=False)

global_points = []
global_point_label = []

with gr.Blocks(css=css, title="Ejemplo aplicativo de MobileSAM") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(title)

    with gr.Tab("Imagenes"):
        # Imagenes
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_inicial.render()

            with gr.Column(scale=1):
                img_segmentada.render()

        # Enviar y limpiar
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        opc_segmetacion_img = gr.Radio(
                            ["Agregar", "Segmentar todo", "Remover"],
                            value="Agregar",
                            label="Opciones disponibles"
                        )

                    with gr.Column():
                        btn_segmentar = gr.Button(
                            "Comenzar la segmentación!", variant="primary"
                        )
                        btn_limpiar = gr.Button("Reiniciar", variant="secondary")

                gr.Markdown("Ejemplos")
                gr.Examples(
                    examples=fotos,
                    inputs=[img_inicial],
                    examples_per_page=6,
                    label=None
                )
    with gr.Tab("Videos"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                vid_inicial_point = gr.Video(label="Video Inicial", sources=["upload"], format="mp4", include_audio=False)
            
            with gr.Column(scale=1):
                fotograma = gr.Image(label="Imagen inicial", type="pil")
                slider_pos = gr.Slider(minimum=0, maximum=100, step=1, label="Posición del video")


        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=1):
                        opc_segmetacion_vid = gr.Radio(
                            ["Agregar", "Encuadrar"],
                            value="Agregar",
                            label="Opciones disponibles"
                        )

                    with gr.Column(scale=1):
                        btn_segmentar_vid = gr.Button(
                            "Comenzar la segmentación!", variant="primary"
                        )
                        btn_limpiar = gr.Button("Reiniciar", variant="secondary")
                
        with gr.Row():
            with gr.Column(scale=1):
                fotograma_segmentado.render()        
        
        vid_inicial_point.upload(fn=actualizar_slider, inputs=vid_inicial_point, outputs=slider_pos)
        slider_pos.change(fn=obtener_fotograma, inputs=[vid_inicial_point, slider_pos], outputs=fotograma)


    img_inicial.select(get_options, [img_inicial, opc_segmetacion_img], img_inicial)
    fotograma.select(get_options, [fotograma, opc_segmetacion_vid], fotograma)

    btn_segmentar.click(
        segment_options, inputs=[img_inicial, opc_segmetacion_img], outputs=[img_segmentada, img_inicial]
    )
    btn_segmentar_vid.click(
        segment_options, inputs=[fotograma, opc_segmetacion_vid], outputs=[fotograma_segmentado, fotograma]
    )

    def clear():
        return None, None

    def clear_text():
        return None, None, None

    btn_limpiar.click(clear, outputs=[img_inicial, img_segmentada])
    
    # Limpiar imagen
    #btn_limpiar_imagen.click(clear, outputs=[img_inicial, img_segmentada])

    # Limpiar video
    #btn_limpiar_video.click(clear_video, outputs=[vid_inicial, vid_segmentado])

demo.queue()
demo.launch()
