import cv2

def read_image():
    img = cv2.imread('aa.jpg', cv2.IMREAD_COLOR)
    #ok = cv2.imread('옥태현.jpg', cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = classifier.detectMultiScale(gray_img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('aa', img)
    cv2.imshow('Grayed Aespa', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    read_image()
