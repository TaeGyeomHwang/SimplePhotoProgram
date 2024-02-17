import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

DRAW_RECTANGLE = 0  # 사각형 그리기
DRAW_CIRCLE    = 1  # 원 그리기
DRAW_ECLIPSE   = 2  # 타원 그리기
DRAW_LINE      = 3  # 직선 그리기
DRAW_BRUSH     = 4  # 브러시 그리기
ERASE          = 5  # 지우개
BLURRING       = 6  # 블러 인덱스
MOSAIC         = 7  # 모자이크
SAVE           = 8  # 저장
PLUS           = 9  # 밝게 하기 명령
MINUS          = 10 # 어둡게 하기 명령
CREAR          = 11 # 지우기   명령
BLACK          = 12  # 흑백
COLOR          = 13 # 색상 아이콘
PALETTE        = 14 # 색상 팔레트
HUE_IDX        = 15 # 색상 인덱스
TEXT        = 16 #텍스트 생성시 사용되는 임시
quality = 100 # 화질 기본값 - 100

# 전역 변수
mouse_mode, draw_mode = 0, 0                # 그리기 모드, 마우스 상태
pt1, pt2, Color = (0, 0), (0, 0), (0, 0, 0) # 시작 좌표, 종료 좌표
thickness = 3                               # 선 두께

image = cv2.imread("images/face/03.jpg", cv2.IMREAD_UNCHANGED)  #이미지 받아오기
if image is None:
    raise Exception("영상 파일 읽기 오류 발생")

def mosaic(src, ratio=0.1):     # 모자이크 함수
    # 이미지를 모자이크로 만드는 함수 모자이크의 정도는 ratio로 수정 가능

    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def create_hueIndex(image, roi):  # hue 인덱스로 색상 인덱스를 만드는 함수
    x, y, w, h =  roi                         # 관심 영역 너비, 높이
    index = [[(j, 1, 1) for j in range(w)] for i in range(h)]      # 가로로 만들기
    ratios = (180 / w, 255, 255)
    hueIndex = np.multiply(index, ratios).astype('uint8')  # HSV 화소값 행렬

    image[y:y+h, x:x+w] = cv2.cvtColor(hueIndex, cv2.COLOR_HSV2BGR)

def create_colorPlatte(image, idx, roi):    # 색상 팔레트 만드는 함수
    x, y, w, h = roi
    hue = idx-x
    palatte = [[(hue, j, h-i-1) for j in range(w)] for i in range(h)]

    ratios = (180/w, 255/w, 255/h )
    palatte = np.multiply(palatte, ratios).astype('uint8')

    image[y:y+h, x:x+w] = cv2.cvtColor(palatte, cv2.COLOR_HSV2BGR)


def onMouse(event, x, y, flags, param):

    global pt1, pt2, mouse_mode, draw_mode, ptPalette, menuImage # 추가 마우스 위치 조정

    if event == cv2.EVENT_LBUTTONUP:  # 왼쪽 버튼 떼기
        for i, (x0, y0, w, h) in enumerate(icons):  # 메뉴 아이콘 사각형 조회
            if x0 <= x < x0+ w and y0 <= y < y0 + h:  # 메뉴 클릭 여부 검사
                if i < 8 :                   # draw 함수 갯수
                    mouse_mode = 0          # 마우스 상태 초기화
                    draw_mode = i           # 그리기 모드
                else:
                    command(i)              # 일반 명령이면
                return
        # 캔버스에만 표시해야하기에 위치 조정
        pt2 = (x-menuImage.shape[1], y)                        # 종료 좌표 저장
        ptPalette = (x, y)# 팔레트 좌표
        mouse_mode = 1                      # 버튼 떼기 상태 지정

    elif event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누르기
        pt1 = (x-menuImage.shape[1], y)  # 시작좌표 저장
        mouse_mode = 2

    elif event == cv2.EVENT_RBUTTONDOWN:

        draw_mode = 16
        pt2 = (x - menuImage.shape[1], y)  # 종료좌표 저장
        ptPalette = (x, y)  # 팔레트 좌표

        pil_img = Image.fromarray(canvas)  # cv2 이미지를 pil 이미지로 변환

        fontpath = "fonts/gulim.ttc"
        font = ImageFont.truetype(fontpath, 24)

        print("텍스트를 입력하십시오 : ")
        text = input()

        draw = ImageDraw.Draw(pil_img, 'RGBA')
        draw.text(pt2, text, font=font, fill=(Color))

        img_cv2 = np.array(pil_img)

        image2[:, menuImage.shape[1]:] = img_cv2
        mouse_mode = 2

    if mouse_mode >= 2:  # 왼쪽 버튼 누르기 또는 드래그
        mouse_mode = 0 if x < 125 else 3  # 메뉴 영역 확인- 마우스 상태 지정
        pt2 = (x-menuImage.shape[1], y)
        ptPalette = (x, y) # 팔레트 좌표

def draw(canvas, color=(200, 200, 200)):
    global draw_mode, thickness, pt1, pt2, image2,bluImage, mosaImage, masked,  c1, c2, mask,eraser,erasermask, mosaic,mask1

    if draw_mode == DRAW_RECTANGLE:                 # 사각형 그리기
        cv2.rectangle(canvas , pt1, pt2, color, thickness)

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

    elif draw_mode == DRAW_LINE:                    # 직선 그리기
        cv2.line(canvas , pt1, pt2, color, thickness)

        #추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

    elif draw_mode == DRAW_BRUSH:                   # 브러시 그리기
        cv2.line(canvas , pt1, pt2, color, thickness * 3)

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

        pt1 = pt2 # 종료 좌표를 시작 좌표로 지정

    elif draw_mode == ERASE:                        # 지우개

        # 지우개도 블러링이랑 구현 방식이 같음
        # 지우개의 경우 블러가된 이미지가 아닌 원본이미지와 합침

        cv2.line(erasermask, pt1, pt2, (255, 255, 255), thickness * 5)

        erasermasked = cv2.bitwise_and(eraserImage, erasermask)

        cv2.line(eraser, pt1, pt2, (0, 0, 0), thickness * 5)
        image2[:, menuImage.shape[1]:] = cv2.add(eraser, erasermasked)

        pt1 = pt2
        # 블러 이미지를 수정된 이미지에 블러된 이미지로 갱신
        # 블러 이미지도 지우개 이미지도 다른 기능이 끝날때 갱신을 이렇게 갱신
        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

    #블러링 구현 임시
    elif draw_mode == BLURRING:         #블러링

        # 마스크(검은 창)에 블러할 영역을 흰색으로 그림
        cv2.line(mask, pt1, pt2, (255, 255, 255), thickness * 5)

        # 블러가 된 이미지와 마스크를 합쳐서 maskd를 만듬
        masked = cv2.bitwise_and(bluImage, mask)

        # 두 영역을 합쳐서 canvas 위치에 블러가 된 이미지를 갱신 시켜준다.
        cv2.line(c1, pt1, pt2, (0, 0, 0), thickness * 5)
        image2[:, menuImage.shape[1]:]= cv2.add(c1,masked)

        pt1 = pt2

        # 현재 캔버스를 저장한다.
        eraser = canvas.copy()

        # 마스크를 초기화 해주지 않으면, 이전의 마스크 결과가 남아 문제가 생긴다.
        erasermask = np.zeros_like(eraserImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

    elif draw_mode == MOSAIC:           #모자이크

        cv2.line(mask1, pt1, pt2, (255, 255, 255), thickness * 5)

        masked = cv2.bitwise_and(mosaImage, mask1)

        # 알맹이만 검은 부분
        cv2.line(c2, pt1, pt2, (0, 0, 0), thickness * 5)
        image2[:, menuImage.shape[1]:]= cv2.add(c2,masked)
        pt1 = pt2
        # 추가한 코드 지우개 + 블러
        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        eraser = canvas.copy()
        # 마스크를 초기화 해주지 않으면, 이전의 마스크 결과가 남아 문제가 생긴다.
        erasermask = np.zeros_like(eraserImage)

        # mosaImage = mosaic(canvas)
        # c2 = canvas.copy()
        # mask = np.zeros_like(mosaImage)

    elif draw_mode == DRAW_CIRCLE:                  # 원 그리기
        d = np.subtract(pt1, pt2)           # 두 좌표 차분
        radius = int(np.sqrt(d[0] ** 2 + d[1] ** 2))
        cv2.circle(canvas , pt1, radius, color, thickness)

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask = np.zeros_like(mosaImage)

    elif draw_mode == DRAW_ECLIPSE:                 # 타원 그리기
        center = np.abs(np.add(pt1, pt2)) // 2      # 두 좌표의 중심점 구하기
        size = np.abs(np.subtract(pt1, pt2)) // 2   # 두 좌표의 크기의 절반
        cv2.ellipse(canvas , tuple(center), tuple(size), 0, 0, 360, color, thickness)

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask = np.zeros_like(mosaImage)

    elif draw_mode == TEXT:                   #텍스트 사용시 불러오는 기능

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

    cv2.imshow("test", image2)

def command(mode):
    global icons, image2, canvas, Color, hue, mouse_mode, ptPalette, image2, \
        bluImage, mosaImage, masked, c1, c2, mask, eraser, erasermask,mask1

    if mode == PALETTE:  # 색상 팔레트 영역 클릭 시
        pixel = image2[ptPalette[::-1]] # 팔레트 클릭 시 좌표의 화소값을 가져옴, 팔레트 이용시 그리는 x좌표가 다름 캔버스 때문에
        x, y, w, h = icons[COLOR]
        image2[y:y + h , x:x + w - 1] = pixel   # 색상 아이콘영역에 관심 영역을 지정, 색상 팔레트의 클릭 화소값을 저장
        Color = tuple(map(int, pixel))  # 그리기에는 정수형 튜플만 사용가능

    elif mode == HUE_IDX:  # 색상 인덱스 클릭 시
        create_colorPlatte(image2, ptPalette[0], icons[PALETTE])  # 팔레트 새로 그리기

    elif mode == BLACK:                                   # 영상 파일 열기
        r1 = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        image2[:, menuImage.shape[1]:] = cv2.merge((r1, r1, r1))

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask1 = np.zeros_like(mosaImage)

    elif mode == SAVE:                                  # 캔버스 영역 저장
        cv2.imwrite("images/my_save.jpg", canvas)
        cv2.imwrite("images/test1.jpg", canvas, (cv2.IMWRITE_JPEG_QUALITY, quality))  # 가연이

    elif mode == PLUS:                                  # 캔버스 영상 밝게 변경
        val = np.full(canvas.shape, 10, np.uint8)     # 증가 화소값 행렬 생성
        cv2.add(canvas, val, canvas)

    elif mode == MINUS:                                 # 캔버스 영상 어둡게 변경
        val = np.full(canvas.shape, 10, np.uint8)     # 증가 화소값 행렬 생성
        cv2.subtract(canvas, val, canvas)

    elif mode == CREAR:                                 # 클리어 수정 원본 이미지로
        canvas[:] = image
        mouse_mode = 0

        # 추가한 코드 지우개 + 블러
        eraser = canvas.copy()
        erasermask = np.zeros_like(eraserImage)

        bluImage = cv2.filter2D(canvas, -1, kernel)
        c1 = canvas.copy()
        mask = np.zeros_like(bluImage)

        mosaImage = mosaic(canvas)
        c2 = canvas.copy()
        mask = np.zeros_like(mosaImage)

    cv2.imshow("test", image2)

def onTrackbar(value):                                   # 트랙바 콜백 함수
    global mouse_mode, thickness
    mouse_mode = 0                                       # 마우스 상태 초기화
    thickness = value

def onTrackbar2(value):                                   # 트랙바 콜백 함수 가연이
    global mouse_mode, quality
    mouse_mode = 0                                       # 마우스 상태 초기화
    quality = value


def place_icons(menuImage, size):
    icon_name = ["rect", "circle", "eclipe", "line",  # 아이콘 파일 이름 리스트 생성
                 "brush", "eraser", "blur", "mosaic", "quality",
                 "plus", "minus", "clear", "black","color"]

    #아이콘 위치와 크기를 관심 영역 리스트로 생성
    icons = [(i%2, i//2, 1, 1) for i in range(len(icon_name))]

    icons = np.multiply(icons, size*2)                  # icons 모든 원소에 size 곱합

    for roi, name in zip(icons, icon_name):
        icon = cv2.imread('images/icon/%s.jpg' %name, cv2.IMREAD_COLOR)
        if icon is None: continue
        x, y, w, h = roi
        menuImage[y:y+h, x:x+w] = cv2.resize(icon, size)

    return list(icons)                   # 팔레트 생성

# 메뉴창 만드는 코드 세로의 크기는 불러온 이미지와 동일
# 가로의 크기는 불러온 이미지의 4분의 1 크기 ()
menuImage = np.full((image.shape[0], int(image.shape[1]/4), 3), 255, np.uint8)

#아이콘 크기조절
xIcon = int(menuImage.shape[1]/2)
yIcon = int(image.shape[0]/12)
icons = place_icons(menuImage, (xIcon, yIcon))

# 마지막 아이콘의 위치를 불러온다.
x, y, w, h = icons[-1]                               # 아이콘 사각형 마지막 원소

icons.append((0, y + h, menuImage.shape[1],int(menuImage.shape[0]*9/30)))
icons.append((0, y + h + int(menuImage.shape[0]*9/30), menuImage.shape[1], int(menuImage.shape[0]/30)))  # 색상 인덱스 사각형 추가

# 팔레트와 색상인덱스를 실제로 생성 함수를 호출
create_colorPlatte(menuImage, 0, icons[PALETTE])    # 팔레트 생성
create_hueIndex(menuImage, icons[HUE_IDX])          # 색상 인덱스 생성

# 메뉴이미지를 기존이미지와 병합
image2 = np.hstack((menuImage,image))
cv2.imshow("test",image2)
cv2.setMouseCallback("test", onMouse)                 # 마우스 콜백 함수
cv2.createTrackbar("Thickness", "test", thickness, 255, onTrackbar)
cv2.setTrackbarMin('Thickness', 'test', 1)
cv2.createTrackbar("quality", "test", quality, 100, onTrackbar2)

xIcon = int(menuImage.shape[1]/2)
yIcon = int(menuImage.shape[0]/12)
canvas = image2[:, menuImage.shape[1]:] # 메뉴를 제외한 캔버스 영역

# 초기 블러이미지 생성을 위한 기본 이미지를 복사
bluImage = image

# 초기 마스크 이미지를 생성한다.
mask = np.zeros_like(bluImage)

# 블러 처리를 위한 커널을 생성
kernel = np.ones((5, 5)) / 5 ** 2

# 블러 이미지 생성
bluImage = cv2.filter2D(bluImage, -1, kernel)

c1 = canvas.copy()

# 초기 마스크 생성을 위한 코드
bluImage = image                                # 초기 블러 이미지 생성을 위한 기본 이미지를 복사
mask = np.zeros_like(bluImage)                  # 초기 마스크 이미지를 생성한다.
kernel = np.ones((5, 5)) / 5 ** 2               # 블러 처리를 위한 커널을 생성
bluImage = cv2.filter2D(bluImage, -1, kernel)   # 블러 이미지 생성
c1 = canvas.copy()

mosaImage = image
mask1 = np.zeros_like(mosaImage)
kernel = np.ones((5, 5)) / 5 ** 2
mosaImage = mosaic(mosaImage)
c2 = canvas.copy()

#지우개
eraserImage = image
erasermask = np.zeros_like(bluImage)
eraser = canvas.copy()

while True:
    if mouse_mode == 1:                                # 마우스 버튼 떼기
        draw(canvas, Color)                             # 원본에 그림
    elif mouse_mode == 3:                              # 마우스 드래그
        if draw_mode == DRAW_BRUSH or draw_mode == ERASE:
            draw(canvas, Color)                         # 원본에 그림
        else:
            draw(np.copy(canvas), (200, 200, 200))      # 복사본에 회색으로 그림
    if cv2.waitKey(30) == 27:                          # ESC 키를 누르면 종료
        break