import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AirSketch AI",
    page_icon="✋",
    layout="wide"
)

st.title("✋ AirSketch AI")
st.write("Real-Time Hand Gesture Drawing and Recognition")

# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🎨 Drawing Controls")

start_camera = st.sidebar.button("▶ Start Camera")
stop_camera = st.sidebar.button("⏹ Stop Camera")
clear_canvas = st.sidebar.button("🗑 Clear Drawing")

# ONLY LIGHT BLUE
drawing_color = (255, 200, 100)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = False

if "canvas" not in st.session_state:
    st.session_state.canvas = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

if "previous_point" not in st.session_state:
    st.session_state.previous_point = None

# --------------------------------------------------
# BUTTON ACTIONS
# --------------------------------------------------

if start_camera:
    st.session_state.running = True

if stop_camera:
    st.session_state.running = False
    st.session_state.previous_point = None

if clear_canvas:
    st.session_state.canvas = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )
    st.session_state.previous_point = None

# --------------------------------------------------
# FINGER COUNTING
# --------------------------------------------------

def count_fingers(hand_landmarks):

    fingers = 0

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers += 1

    # Index finger
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        fingers += 1

    # Middle finger
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        fingers += 1

    # Ring finger
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        fingers += 1

    # Little finger
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        fingers += 1

    return fingers

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

frame_placeholder = st.empty()
gesture_placeholder = st.empty()

if not st.session_state.running:

    st.info(
        "Click 'Start Camera' from the sidebar to begin."
    )

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

if st.session_state.running:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while st.session_state.running:

            success, frame = cap.read()

            if not success:
                st.error("Camera could not be opened.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe processing
            results = hands.process(rgb_frame)

            # --------------------------------------------------
            # HAND DETECTED
            # --------------------------------------------------

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Count fingers
                finger_count = count_fingers(
                    hand_landmarks
                )

                # --------------------------------------------------
                # GESTURE RECOGNITION
                # --------------------------------------------------

                if finger_count == 1:

                    gesture_text = "☝️ DRAWING"

                elif finger_count == 2:

                    gesture_text = "✌️ TWO FINGERS"

                elif finger_count == 3:

                    gesture_text = "🤟 THREE FINGERS"

                elif finger_count == 4:

                    gesture_text = "4 FINGERS"

                elif finger_count == 5:

                    gesture_text = "🖐️ FIVE FINGERS"

                else:

                    gesture_text = "✊ CLOSED HAND"

                gesture_placeholder.success(
                    f"Detected Gesture: {gesture_text}"
                )

                # --------------------------------------------------
                # INDEX FINGER POSITION
                # --------------------------------------------------

                index_finger = hand_landmarks.landmark[8]

                x = int(index_finger.x * 640)
                y = int(index_finger.y * 480)

                # Keep inside canvas
                x = max(0, min(x, 639))
                y = max(0, min(y, 479))

                # --------------------------------------------------
                # DRAWING
                # --------------------------------------------------

                if finger_count == 1:

                    if st.session_state.previous_point is not None:

                        previous_x, previous_y = (
                            st.session_state.previous_point
                        )

                        cv2.line(
                            st.session_state.canvas,
                            (previous_x, previous_y),
                            (x, y),
                            drawing_color,
                            5
                        )

                    st.session_state.previous_point = (
                        x,
                        y
                    )

                else:

                    st.session_state.previous_point = None

            # --------------------------------------------------
            # NO HAND
            # --------------------------------------------------

            else:

                st.session_state.previous_point = None

                gesture_placeholder.warning(
                    "No hand detected"
                )

            # --------------------------------------------------
            # COMBINE CAMERA + DRAWING
            # --------------------------------------------------

            display = cv2.addWeighted(
                frame,
                1,
                st.session_state.canvas,
                0.7,
                0
            )

            # Display
            frame_placeholder.image(
                cv2.cvtColor(
                    display,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB"
            )

    cap.release()