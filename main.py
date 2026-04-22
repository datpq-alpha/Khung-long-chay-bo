import pygame, cv2, mediapipe as mp, os, numpy as np

# ================= MEDIA PIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(1)
draw = mp.solutions.drawing_utils

# ================= GAME INIT =================
pygame.init()
W, H = 800, 400
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Dino AI Controller")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 25)
font_big = pygame.font.SysFont("Arial", 50)

cap = cv2.VideoCapture(0)
cap.set(3, 320); cap.set(4, 240)

# ================= LOAD IMAGE =================
def load_img(name, size, color):
    if os.path.exists(name):
        return pygame.transform.scale(pygame.image.load(name), size)
    s = pygame.Surface(size); s.fill(color); return s

bg = load_img("background.jpg", (600,300), (200,200,200))
robot = load_img("robot.png", (50,50), (0,0,255))
spike = load_img("spike.png", (40,40), (255,0,0))

# ================= GAME STATE =================
def reset():
    return dict(xv=5, yv=8, bgx=0, ry=230, sx=550,
                score=0, pause=False, jump=False)

g = reset()

# ================= HAND GESTURE =================
def get_gesture(frame):
    # Chuyển từ BGR (OpenCV) sang RGB (MediaPipe yêu cầu)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe phân tích ảnh để tìm các điểm mấu chốt trên tay
    res = hands.process(rgb)
    gesture = None

    # Nếu phát hiện có bàn tay trong khung hình
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0] # Lấy dữ liệu bàn tay đầu tiên
        # Vẽ các điểm và đường nối xương tay lên ảnh gốc
        draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # Định nghĩa hàm y(i) để lấy tọa độ chiều dọc của điểm thứ i
        y = lambda i: lm.landmark[i].y
        
        # Đếm số ngón mở: Nếu đầu ngón (8,12,16,20) nằm cao hơn khớp (6,10,14,18)
        fingers = sum([y(8)<y(6), y(12)<y(10),
                       y(16)<y(14), y(20)<y(18)])

        # Phân loại: 0 ngón mở là BÚA, từ 3 ngón trở lên là BAO
        if fingers == 0: gesture = "ROCK"
        elif fingers >= 3: gesture = "PAPER"

    # Trả về kết quả cử chỉ, ảnh đã vẽ và ảnh RGB
    return gesture, frame, rgb

# ================= MAIN LOOP =================
run = True
while run:
    clock.tick(180)
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)

    gesture, frame, rgb = get_gesture(frame)

    for e in pygame.event.get():
        if e.type == pygame.QUIT: run = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            if g["pause"]: g = reset()
            elif g["ry"] == 230: g["jump"] = True

    # Gesture control
    if gesture == "ROCK" and not g["pause"] and g["ry"] == 230:
        g["jump"] = True
    if gesture == "PAPER" and g["pause"]:
        g = reset()

    screen.fill((255,255,255))

    # Update game
    if not g["pause"]:
        g["bgx"] = (g["bgx"] - g["xv"]) % -600
        g["sx"] -= g["xv"]
        if g["sx"] <= -20:
            g["sx"] = 550; g["score"] += 1

    # Jump logic
    if g["jump"] and g["ry"] > 80:
        g["ry"] -= g["yv"]
    else:
        g["jump"] = False
    if g["ry"] < 230 and not g["jump"]:
        g["ry"] += g["yv"]

    # Draw game
    screen.blit(bg, (g["bgx"],0))
    screen.blit(bg, (g["bgx"]+600,0))
    r_rect = screen.blit(robot, (0,g["ry"]))
    s_rect = screen.blit(spike, (g["sx"],230))

    screen.blit(font.render(f"Score: {g['score']}",1,(0,0,0)),(10,10))
    screen.blit(font.render(f"AI: {gesture or '...'}",1,(0,100,0)),(10,40))

    if r_rect.inflate(-10,-10).colliderect(s_rect.inflate(-10,-10)):
        g["pause"] = True
        screen.blit(font_big.render("THUA - XOE TAY",1,(255,0,0)),(100,150))

    # ================= REMOVE BACKGROUND =================
    seg = segmentation.process(rgb)
    mask = (seg.segmentation_mask > 0.4).astype(np.uint8)*255
    rgba = np.dstack((rgb, mask))
    cam = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
    cam = pygame.transform.scale(cam,(200,150))
    screen.blit(cam,(W-210,10))

    pygame.display.flip()

cap.release()
pygame.quit()
