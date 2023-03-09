import streamlit as st
from PIL import Image
from classifier import classify

# load labels
with open('labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
    

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Distracted Driver Detection")
st.write("")

file_up = st.file_uploader("Upload an image")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")

    
    AlexNetTitle = '<p style="font-family:sans-serif; color:Green; font-size: 22px;">AlexNet</p>'
    VGG16Title = '<p style="font-family:sans-serif; color:Green; font-size: 22px;">VGG16</p>'
    VGG19Title = '<p style="font-family:sans-serif; color:Green; font-size: 22px;">VGG19</p>'
    ResNetTitle = '<p style="font-family:sans-serif; color:Green; font-size: 22px;">ResNet(Kaggle)</p>'

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(AlexNetTitle, unsafe_allow_html=True)  

        pred1, inf1 = classify(image, "alexnet")

        st.write(labels[pred1], "\n\ntime: ", inf1, "s")
    with col2:
        st.markdown(VGG16Title, unsafe_allow_html=True)
        
        pred2, inf2 = classify(image, "vgg16")

        st.write(labels[pred2], "\n\ntime: ", inf2, "s")
    with col3:
        st.markdown(VGG19Title, unsafe_allow_html=True)

        pred3, inf3 = classify(image, "vgg19")

        st.write(labels[pred3], "\n\ntime: ", inf3, "s")
    with col4:
        st.markdown(ResNetTitle, unsafe_allow_html=True)

        pred4, inf4 = classify(image, "resnet50")

        st.write(labels[pred4], "\n\ntime: ", inf4, "s")
