import cv2
import utils

def main():
    engine = utils.AudioEngine()
    vid = cv2.VideoCapture(0)
    detector =  utils.ObjectDetector(match_threshold=0.9)

    while True:
        ret, frame = vid.read()
        if ret:
            is_detected, detection, score = detector.detect(frame)

            if is_detected:
                print("Detected:", detection, " | Score:", score)
                engine.say(detection)

            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

if __name__ == "__main__":
    main()