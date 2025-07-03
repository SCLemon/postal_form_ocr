from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def paddleOCR(url):
    result = ocr.predict(url)

    text = ''
    for t in result[0]['rec_texts']:
        text += t

    print(text)
    return text