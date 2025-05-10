import os
import cv2
import time
import uuid

IMAGE_PATH = 'Dataset-BISINDO'

labels = ['A', 'B', 'C', 'D',
          'E', 'F', 'G', 'H', 
          'I', 'J', 'K', 'L', 
          'M', 'N', 'O', 'P', 
          'Q', 'R', 'S', 'T', 
          'U', 'V', 'W', 'X', 
          'Y', 'Z', 'Halo', 'NamaAku']

number_of_images = 20

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera disabled!")

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)
    print('Collecting images for {}'.format(label))
    time.sleep(3)
    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image!")
            continue
        imagename = os.path.join(IMAGE_PATH, label, '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()