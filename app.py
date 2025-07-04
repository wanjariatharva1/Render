from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import base64


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

genai.configure(api_key="AIzaSyB28IhNewiq2q3ag7_2m6IOEDtxa_bQqro") 
model = genai.GenerativeModel('gemini-1.5-flash')


canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  
prev_pos = None  
ai_triggered = False  
output_text = "" 
flip_canvas = False  

def get_hand_info(img):
    """Detect hand gestures and landmarks in the given image."""
    hands, _ = detector.findHands(img, draw=False, flipType=True)  
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw_on_canvas(info, prev_pos, canvas, flip=False):
    """Draw on the canvas based on detected hand gestures."""
    if info:
        fingers, lmList = info
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:  
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, tuple(map(int, prev_pos)), tuple(map(int, current_pos)), (255, 0, 255), 10)
        elif fingers == [1, 0, 0, 0, 0]:  
            canvas = np.zeros_like(canvas)

        if flip:
            
            canvas = cv2.flip(canvas, 1)
        
        return current_pos, canvas
    return prev_pos, canvas

def generate_ai_output(fingers, user_prompt):
    """Generate AI content based on hand gestures and an optional user prompt."""
    global ai_triggered, output_text
    if fingers == [1, 1, 1, 1, 0] and not ai_triggered:  
        pil_image = Image.fromarray(canvas).resize((640, 480))

        try:
            prompt = user_prompt or "Generate based on canvas content"
            response = model.generate_content([prompt, pil_image])  
            ai_triggered = True
            output_text = response.text
            return output_text
        except Exception as e:
            print(f"Error during AI generation: {e}")
            return "Error generating AI content."
    return output_text


@app.post("/api/hand-gesture")
async def hand_gesture_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form(None)
):
    """Handle hand gesture detection and canvas drawing."""
    global canvas, prev_pos, ai_triggered, output_text, flip_canvas

    
    img_data = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data."})

    img = cv2.flip(img, 1)  
    hand_info = get_hand_info(img)  

    if hand_info and hand_info[0] == [0, 1, 0, 1, 0]:  
        flip_canvas = not flip_canvas  

    prev_pos, canvas = draw_on_canvas(hand_info, prev_pos, canvas, flip_canvas)

    if hand_info and not ai_triggered:
        output_text = generate_ai_output(hand_info[0], prompt)

    if hand_info and hand_info[0] == [1, 0, 0, 0, 0]:  
        ai_triggered = False
        output_text = ""

    _, buffer = cv2.imencode('.jpg', canvas)
    canvas_b64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"outputText": output_text, "canvas": canvas_b64})

@app.post("/api/drawing")
async def process_drawing(
    image: UploadFile = File(...),  
    prompt: str = Form(None),      
):
    """Process drawing image to generate AI content."""
    img = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data."})

    pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((640, 480))  

    try:
        prompt = prompt if prompt else "Default AI prompt for your drawing."
        response = model.generate_content([prompt, pil_image])  
        ai_output = response.text  
    except Exception as e:
        print(f"Error while generating AI content: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "AI processing failed."})

    _, buffer = cv2.imencode('.jpg', img)
    canvas_b64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"outputText": ai_output, "canvas": canvas_b64})

