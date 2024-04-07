import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import numpy as np
import io
from streamlit_image_coordinates import streamlit_image_coordinates



#configurations
st.set_page_config(
    page_title="Swiftify anything",
    page_icon="🎵",
)
def read_taylor():
    imagePath2 = f'static//Tswift//img{np.random.choice([1, 2, 3])}.png'
    img2 = cv2.imread(imagePath2)
    return img2
    
def face_detection(gray_imgOG):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = face_classifier.detectMultiScale(
    gray_imgOG, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    OG_rgb = cv2.cvtColor(imgOG, cv2.COLOR_BGR2RGB)
    return face

def markfaces(img1,face,noface=[]):
  imgOG2 = img1.copy()
  for (x, y, w, h) in face:
    cv2.rectangle(imgOG2, (x, y), (x + w, y + h), (0, 255, 0), 4)
  return imgOG2

def print_faces(fb,img1,img2):
  images=[]
  imgOG2 = img1.copy()
  for (x, y, w, h) in list(fb):
      img=cv2.resize(img2, (w,w))
      images.append(img)
  for (x, y, w, h), img in zip(list(fb), images):
    imgOG2[y:y+w, x:x+w] = img
    
  return imgOG2
  #plt.figure(figsize=(20,10))
  #plt.imshow(cv2.cvtColor(imgOG2, cv2.COLOR_BGR2RGB))
  #plt.axis('off')


def check_points(face, noface):
  is_in_face = []
  try:
    for face_box in face:
        x1, y1, w, h = face_box
        for point in noface:
            x, y = point
            if x1 <= x*2 <= x1 + w and y1 <= y*2 <= y1 + h:
                is_in_face.append(face_box)
    is_in_face_array = np.array(is_in_face)
    delta = set(map(tuple, is_in_face_array))
    idx = [tuple(x) not in delta for x in face]
    elements_not_in_is_in_face = face[idx]
   
  except ValueError:
    elements_not_in_is_in_face = face    
    pass
  return elements_not_in_is_in_face


if "points" not in st.session_state:
    st.session_state["points"] = []




def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
st.markdown(
    """

    # 🎸Memories - Taylor's Version.
    ## Does your significant other always dream of hanging out with Taylor Swift? 
    """
)
st.markdown(
    """
    **Use AI to make their dreams come true. Kindof.** 
    
    step1: Upload a picture. Also, things get deleted right after you refresh.  \n
    Step2: Click "**Process Image**". Then use your mouse(or finger) to mark the green boxes that you _***DON'T***_ want Taylor'ized. Ex: maybe don't Taylor'ize your partner's face. \n
    Step3: Click on "**View Taylor's Version**". Voila! There should also be a download button. \n
    Step4: Refresh and Taylorize more memories. \n

    *Note: If you mess up, just refresh the page, and start over.* \n 
    *Note: If you're worried about privacy and stuff? Dont be, here is some [(soothing music)](https://www.youtube.com/watch?v=79kpoGF8KWU) to help you destress. Honestly though, the page doesnt store anything beyond your session.* 

    """
)


img = st.file_uploader("Choose a picture, ideally one with faces",type=["jpg","png","jpeg"])
if img is not None:
    image = Image.open(img)
    st.text("Original Image")
    st.image(img,use_column_width=True)
    st.session_state['fname']=img.name
   
   
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    imgOG = cv2.imdecode(file_bytes, 1)
    max_width = 1080
    width, height = imgOG.shape[1], imgOG.shape[0]
    aspect_ratio = width / height

    if width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
        imgOG = cv2.resize(imgOG, (new_width, new_height))

    st.write("Resized Image shape: ", imgOG.shape)

session_state = st.session_state
if 'face' not in session_state:
    session_state['face'] = np.array([])
if 'noface' not in session_state:
    session_state['noface'] = np.array([])
if 'imgMarked' not in session_state:
    session_state['imgMarked'] = np.array([])

if st.button("Process Image"):
    try:
        gray_imgOG = cv2.cvtColor(imgOG, cv2.COLOR_BGR2GRAY)
        face = face_detection(gray_imgOG)
        session_state['face'] = face
        noface=[]
        imgMarked=markfaces(imgOG,face,noface)
        session_state['imgMarked']=cv2.cvtColor(imgMarked, cv2.COLOR_BGR2RGB)
    except NameError:
        st.write("Please upload an image first.")
        pass
    
#-------------------
if session_state['imgMarked'] !=[]:
    img = Image.fromarray(np.uint8(session_state['imgMarked']))
    
    img = img.resize((img.width // 2, img.height // 2))
    draw = ImageDraw.Draw(img)

    # Draw an ellipse at each coordinate in points
    for point in st.session_state["points"]:
        coords = get_ellipse_coords(point)
        draw.ellipse(coords, fill="red")

    value = streamlit_image_coordinates(img, key="pil")

    if value is not None:
        point = value["x"], value["y"]

        if point not in st.session_state["points"]:
            st.session_state["points"].append(point)
            st.rerun()


  

if st.button("View Taylors'Version"):
    img2=read_taylor()
    is_in_face = check_points(session_state['face'], session_state['points'])
    imgOG2=print_faces(is_in_face,imgOG,img2)
    st.image(imgOG2, channels="BGR")
    
    imout=cv2.cvtColor(imgOG2, cv2.COLOR_BGR2RGB)
    ret, img_enco = cv2.imencode(".png", imout[:, :, [2, 1, 0]])  #numpy.ndarray
    srt_enco = img_enco.tostring()  #bytes
    img_BytesIO = io.BytesIO(srt_enco) #_io.BytesIO
    img_bytes = io.BufferedReader(img_BytesIO) #_io.BufferedReader

    btn = st.download_button(
            label=":⬇️Download image",
            data=img_bytes,
            file_name=f"{session_state['fname'][:10]}_Swifted.png",
            mime="image/png"
    )
