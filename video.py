import pafy
import cv2

def show_video():
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = pafy.new('https://youtu.be/9ccYwC3pv4Y')
    best = video.getbest(preftype='mp4')
    capture = cv2.VideoCapture(best.url)
    while True:
        success, frame = capture.read()
        img = frame
        faces = classifier.detectMultiScale(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if not success:
            break
        cv2.imshow(video.title, frame)

        if cv2.waitKey(1) > 0:
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    show_video()
