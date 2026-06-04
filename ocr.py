import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

#Author:Chen Zeyue
#Funtion:recognition_text
def recognize_text(img):
    text = pytesseract.image_to_string(img)
    return text

#Author:Chen Zeyue
#Function:get_text_boxes
def get_text_boxes(img):
    boxes = []
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    for i in range(len(data['text'])):
        if data['text'][i].strip() != '':
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            boxes.append((x, y, x + w, y + h))
    return boxes