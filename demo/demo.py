import gradio as gr
from func import slider_max_update, audio_control, text_control

CUSTOM_CSS = """
    .audio {
        height: 200px !important;
    }
"""

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Unified Text and Audio Control for Music Mixing Style Transfer</h1></center>")
    
    gr.Markdown("")
    gr.Markdown("<h2>Upload Raw Tracks</h2>")
    with gr.Row():
        with gr.Column():
            raw_file_1 = gr.Audio(label="Raw Track 1", type="filepath", sources="upload", elem_classes="audio")
            raw_file_5 = gr.Audio(label="Raw Track 5", type="filepath", sources="upload", elem_classes="audio")
        with gr.Column():
            raw_file_2 = gr.Audio(label="Raw Track 2", type="filepath", sources="upload", elem_classes="audio")
            raw_file_6 = gr.Audio(label="Raw Track 6", type="filepath", sources="upload", elem_classes="audio")
        with gr.Column():
            raw_file_3 = gr.Audio(label="Raw Track 3", type="filepath", sources="upload", elem_classes="audio")
            raw_file_7 = gr.Audio(label="Raw Track 7", type="filepath", sources="upload", elem_classes="audio")
        with gr.Column():
            raw_file_4 = gr.Audio(label="Raw Track 4", type="filepath", sources="upload", elem_classes="audio")
            raw_file_8 = gr.Audio(label="Raw Track 8", type="filepath", sources="upload", elem_classes="audio")

    gr.Markdown("")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h2>Audio Control</h2>")
            ref_file = gr.Audio(label="Ref song", type="filepath", sources="upload", elem_classes="audio")
            ref_audio_st = gr.Slider(label="Ref Start Time (s)", minimum=0, maximum=0, step=0.1, interactive=True)
            btn_audio_gen = gr.Button("Generate Mixing with Audio Control")
        with gr.Column():
            gr.Markdown("<h2>Text Control</h2>")
            with gr.Row():
                text_style_dropdown = gr.Dropdown(
                    choices=["Brightness", "Punchy", "Loudness", "Panning"],
                    label="Modified Style", interactive=True
                )
                target_track_dropdown = gr.Dropdown(
                    choices=["Track 1", "Track 2", "Track 3", "Track 4", "Track 5", "Track 6", "Track 7", "Track 8", "Mastering"], 
                    label="Modified Tracks", interactive=True
                )
            text_strength_slider = gr.Slider(label="Text Control Strength", minimum=0, maximum=100, step=0.01, interactive=True)
            style_strength_slider = gr.Slider(label="Style Strength", minimum=0, maximum=100, step=0.01, interactive=True)
            ref_st = gr.Slider(label="Ref Start Time (s)", minimum=0, maximum=0, step=0.1, interactive=True)
            btn_text_gen = gr.Button("Generate Mixing with Text Control")

    gr.Markdown("<h2>Output Mixed Track</h2>")
    with gr.Row():
        output_mix = gr.Audio(label="Output Mixed Track", type="numpy", elem_classes="audio")
    with gr.Row():
        with gr.Column():
            output_track_1 = gr.Audio(label="Output Track 1", type="numpy", elem_classes="audio")
            output_track_5 = gr.Audio(label="Output Track 5", type="numpy", elem_classes="audio")
        with gr.Column():
            output_track_2 = gr.Audio(label="Output Track 2", type="numpy", elem_classes="audio")
            output_track_6 = gr.Audio(label="Output Track 6", type="numpy", elem_classes="audio")
        with gr.Column():
            output_track_3 = gr.Audio(label="Output Track 3", type="numpy", elem_classes="audio")
            output_track_7 = gr.Audio(label="Output Track 7", type="numpy", elem_classes="audio")
        with gr.Column():
            output_track_4 = gr.Audio(label="Output Track 4", type="numpy", elem_classes="audio")
            output_track_8 = gr.Audio(label="Output Track 8", type="numpy", elem_classes="audio")

    ## === Functions === ##
    ref_file.change(
        fn=slider_max_update,
        inputs=[ref_file],
        outputs=[ref_audio_st]
    )

    btn_audio_gen.click(
        fn=audio_control,
        inputs=[
            raw_file_1, raw_file_2, raw_file_3, raw_file_4, 
            raw_file_5, raw_file_6, raw_file_7, raw_file_8, 
            ref_file, ref_audio_st
        ],
        outputs=[output_mix, output_track_1, output_track_2, output_track_3, output_track_4,
                output_track_5, output_track_6, output_track_7, output_track_8]
    )

    btn_text_gen.click(
        fn=text_control,
        inputs=[
            raw_file_1, raw_file_2, raw_file_3, raw_file_4, 
            raw_file_5, raw_file_6, raw_file_7, raw_file_8,
            text_style_dropdown, target_track_dropdown,
            text_strength_slider, style_strength_slider, ref_st
        ],
        outputs=[output_mix, output_track_1, output_track_2, output_track_3, output_track_4,
                output_track_5, output_track_6, output_track_7, output_track_8]
    )

demo.launch(theme=gr.themes.Default(), css=CUSTOM_CSS)