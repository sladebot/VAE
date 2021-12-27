import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import utils
from PIL import Image


st.set_page_config("VAE MNIST Pytorch Lightning")
st.title("VAE Playground")
title_img = Image.open("images/title_img.jpg")

st.image(title_img)
st.markdown(
    "This is a simple streamlit app to showcase how a simple VAEs."
)

def load_model_files():
    files = os.listdir("./saved_models/")
    # Docker creates some whiteout files which mig
    files = [i for i in files if ".ckpt" in i]
    clean_names = [utils.parse_model_file_name(name) for name in files]
    return {k: v for k, v in zip(clean_names, files)}


file_name_map = load_model_files()
files = list(file_name_map.keys())

st.header("🖼️ Image Reconstruction", "recon")

with st.form("reconstruction"):
    model_name = st.selectbox("Choose Model:", files,
                              key="recon_model_select")
    recon_model_name = file_name_map[model_name]
    recon_canvas = st_canvas(
        # Fixed fill color with some opacity
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=8,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="recon_canvas",
    )
    submit = st.form_submit_button("Perform Reconstruction")
    if submit:
        recon_model = utils.load_model(recon_model_name)
        inp_tens = utils.canvas_to_tensor(recon_canvas)
        _, _, out = recon_model(inp_tens)
        out = (out+1)/2
        out_img = utils.resize_img(utils.tensor_to_img(out), 150, 150)
if submit:
    st.image(out_img)
    
