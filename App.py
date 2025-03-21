import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown h1 {
        color: #4CAF50;
        text-align: center;
    }
    .stMarkdown h2 {
        color: #333333;
    }
    .stMarkdown p {
        color: #555555;
    }
    .stImage>img {
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.markdown("<h1>AI-Driven Furniture Detection and Interaction</h1>", unsafe_allow_html=True)

# Upload an image
st.markdown("<h2>Upload an Image of Your Room</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Load the trained YOLOv8 model
model = YOLO("best.pt")  

# Function to generate AI-driven interactions
def generate_interaction(detected_labels):
    interactions = []
    
    if "Chair" in detected_labels:
        interactions.extend([
            "Would you like to turn on ambient lighting? Adding a small lamp or LED strip near the chair can create a cozy reading nook.",
            "Consider adding a throw blanket or cushion to the chair for extra comfort and style.",
            "You could place a small side table next to the chair to hold books, drinks, or a reading lamp.",
            "If the chair is near a window, adding some curtains or blinds can help control natural light and create a relaxing atmosphere.",
            "A footrest or ottoman paired with the chair can make it even more comfortable for lounging."
        ])
    
    if "Table" in detected_labels:
        interactions.extend([
            "How about placing a vase or laptop on the table? A vase with fresh flowers can brighten up the room, while a laptop stand can make the table more functional for work.",
            "You could add a table runner or placemat to protect the surface and add a decorative touch.",
            "Consider organizing the table with a tray or organizer to keep items like remotes, pens, or keys neatly arranged.",
            "If the table is in a dining area, adding a centerpiece like candles or a fruit bowl can make meals more enjoyable.",
            "A small plant or succulent on the table can bring a touch of nature indoors."
        ])
    
    if "Sofa" in detected_labels:
        interactions.extend([
            "You could add a coffee table or rug to complement the sofa. A coffee table provides a place for drinks and books, while a rug can tie the seating area together and add warmth.",
            "Consider adding some throw pillows or a cozy blanket to the sofa for extra comfort and a pop of color.",
            "If the sofa is against a wall, hanging some artwork or shelves above it can create a focal point in the room.",
            "A side table next to the sofa can be useful for placing lamps, drinks, or decorative items.",
            "If the sofa is in a living room, rearranging the seating layout to face a TV or fireplace can improve the room's functionality."
        ])
    
    if not interactions:
        interactions.append("No specific recommendations for this room. Consider adding some personal touches, like artwork or photos, to make the space feel more like home.")
    
    return interactions


if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.markdown("<h2>Uploaded Image</h2>", unsafe_allow_html=True)
    st.image(image_rgb, caption="Your Uploaded Image", use_column_width=True)

    # Perform object detection
    results = model(image)

    # Extract detected objects
    detected_objects = results[0].boxes.cls.tolist()  
    class_names = model.names  
    detected_labels = [class_names[int(obj)] for obj in detected_objects]

    # Display detected objects
    st.markdown("<h2>Detected Objects</h2>", unsafe_allow_html=True)
    st.write(", ".join(detected_labels))

    # Generate AI-driven interactions
    interactions = generate_interaction(detected_labels)

    # Display interactions
    st.markdown("<h2>AI-Driven Interactions</h2>", unsafe_allow_html=True)
    for suggestion in interactions:
        st.markdown(f"<p style='color: #4CAF50; font-size: 18px;'>- {suggestion}</p>", unsafe_allow_html=True)

    # Display the annotated image
    st.markdown("<h2>Annotated Image</h2>", unsafe_allow_html=True)
    annotated_image = results[0].plot()  

    # Convert annotated image from BGR to RGB
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image_rgb, caption="Detected Objects", use_column_width=True)