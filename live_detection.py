import cv2
import pytesseract
from imutils import contours
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # kurduğunuz tesseract motorunun exe yolu

mock_data = ["06 CRR 058"] # örnek data, veritabanı varsa bunun yerine kullanılabilir

cezalilar = [] # cezalıların kaydedilecegi liste

TEMPLATES = [ # plaka ayrıştırma için regex şablonları
    r'\b\d{2}.{0,1}[^\d\W]{0,1}.{0,1}\b\d{4,5}\b',
    r'\b\d{2}.{0,1}[^\d\W]{0,1}.{0,1}\b[^\d\W]{0,1}.{0,1}\b\d{3,4}\b',
    r'\b\d{2}.{0,1}[^\d\W]{0,1}.{0,1}\b[^\d\W]{0,1}.{0,1}\w{0,1}.{0,1}\b\d{2,3}\b'
]

def getRegexPlate(raw_data: str): # plakayı regexten geçirme fonksiyonu
    for temp in TEMPLATES:
        regex_plate = re.findall(temp, raw_data) # şablonlardan geçirme
        for i in regex_plate: # eğer tespit edildiyse
            if i: return i # döndür
    return None

vid = cv2.VideoCapture(0)

while vid.isOpened(): # video aktifken

    ret, frame = vid.read()

    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale çevirme
    blur = cv2.GaussianBlur(gray, (5,5), 0) # gaussian blurleme: https://en.wikipedia.org/wiki/Gaussian_blur
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # resmi segmente etme
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # potansiyel köşe-çizgiler (contours)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right") # contourları sıralama

    plate = "" # plakanın depolanacağı değişken
    for c in cnts:
        area = cv2.contourArea(c) # contour bölgesi belirleme
        x,y,w,h = cv2.boundingRect(c) # contourun koordinatlarını/büyüklüklerini belirleme
        center_y = y + h/2
        if area > 3000 and (w > h) and center_y > height/2: # bölgenin uygun olup olmadığını hesaplamalar
            plaka = frame[y:y+h, x:x+w] # resmin plaka kısmını belirleme
            data = pytesseract.image_to_string(plaka, lang='eng', config='--psm 6') # tesseract motoruyla resme ocr gerçekleştirme
            data = "".join([i if i.isalnum() or i.isspace() else " " for i in data]) # gereksiz karakterlerden temizleme
            regex_dat = getRegexPlate(data) # regex ile plaka çekme
            if regex_dat != None: # bir veri geldiyse
                plate += regex_dat # plakayı ayarla
                if plate in mock_data:
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (36,255,12), 1)
                    cv2.putText(frame, 'Engelli Birey', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                else:
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)
                    cv2.putText(frame, 'Engelli Birey Degil!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    if plate not in cezalilar:
                        cezalilar.append(plate)
                break
            else: # veri gelmediyse
                plate = "Plaka tespit edilemedi!"
                break

    cv2.imshow("Plaka", frame)
    print('Plaka:', plate)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()