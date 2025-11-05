# Import all dependencies
import streamlit as st
import os
import shutil
import imageio
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model


st.set_page_config(layout='wide')

# clear temp_videos folder each run
shutil.rmtree("temp_videos", ignore_errors=True)
os.makedirs("temp_videos", exist_ok=True)

#  Sidebar 
with st.sidebar:
    st.image('https://i.postimg.cc/MGkwqVkK/Speak-Easy.png', width=250)
    st.title('Speak Easy - LipReading')
    if st.button('Meet the Team'):
        st.write("1. AB Hemanth Varma")
        st.write("2. D DINESH REDDY")
        st.write("3. R YASWANTH KRISHNA KOWSHIK")
    
st.title('Speak Easy - Lipreading Full Stack App')

# Data folder
data_path = os.path.join('..', 'data', 's1')

if not os.path.exists(data_path):
    st.error(f"Data folder not found: {data_path}")
else:
    options = os.listdir(data_path)
    selected_video = st.selectbox(' Choose a video to analyze:', options)

    
    col1, col2 = st.columns(2)

    if options:
        #  Left column: Display original video 
        with col1:
            st.info('The video below displays the selected sample in MP4 format')

            file_path = os.path.join(data_path, selected_video)
            video_name = os.path.splitext(selected_video)[0]
            converted_path = os.path.join("temp_videos", f"{video_name}_converted.mp4")

            # Convert to mp4 only if not already converted
            if not os.path.exists(converted_path):
                result = os.system(f'ffmpeg -y -i "{file_path}" -vcodec libx264 "{converted_path}"')

                if result != 0 or not os.path.exists(converted_path):
                    st.error(f" Conversion failed for: {selected_video}. Ensure ffmpeg is installed and working.")
                else:
                    st.success(f" Successfully converted {selected_video} to MP4.")
            else:
                st.info(f"Using cached version of {selected_video}.")

            # Display the converted video
            if os.path.exists(converted_path):
                with open(converted_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)

        #  Right column: Model visualization 
        with col2:
            st.info('This is what the ML model sees during prediction')

            # Load and preprocess video
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            
            # Convert to NumPy for visualization
            video_np = video.numpy() if isinstance(video, tf.Tensor) else video

            # Normalize for display
            if video_np.max() <= 1.0:
                video_np = (video_np * 255).astype(np.uint8)

            # Handle grayscale frames
            if video_np.ndim == 4 and video_np.shape[-1] == 1:
                video_np = np.squeeze(video_np, axis=-1)

            
            st.text(f"Video shape: {video_np.shape}, dtype: {video_np.dtype}")

            
            try:
                gif_path = os.path.join("temp_videos", f"{video_name}_animation.gif")
                imageio.mimsave(gif_path, video_np, fps=10)
                st.image(gif_path, width=400)
            except Exception as e:
                st.error(f"Error saving GIF: {e}")

            #  Model Prediction 
            st.info('Predicted output tokens:')

            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            #st.text(yhat)

            # Decode using CTC
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert tokens to readable text
            st.info('Decoded prediction (words):')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.success(converted_prediction)
