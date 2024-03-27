import cv2
import numpy as np
import face_recognition
import pandas as pd
import matplotlib.pyplot as plt
import os



players = ['Andy', 'anuj','dinesh karthik','faf du plesis','mahipal','KING','vyshak']
address = ['C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\andy.jpeg',
            'C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\anuj.jpg',
            'C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\dk.jpg',
            'C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\faf.jpg',
            'C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\lomo.jpg',
            'C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\vk.jpg',
            'C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\vyshak.jpg']


# Load a sample picture and learn how to recognize it.
andy_image = face_recognition.load_image_file(address[0])
andy_face_encoding = face_recognition.face_encodings(andy_image)[0]

anuj_image = face_recognition.load_image_file(address[1])
anuj_face_encoding = face_recognition.face_encodings(anuj_image)[0]

dk_image = face_recognition.load_image_file(address[2])
dk_face_encoding = face_recognition.face_encodings(dk_image)[0]

faf_image = face_recognition.load_image_file(address[3])
faf_face_encoding = face_recognition.face_encodings(faf_image)[0]

lomo_image = face_recognition.load_image_file(address[4])
lomo_face_encoding = face_recognition.face_encodings(lomo_image)[0]

vk_image = face_recognition.load_image_file(address[5])
vk_face_encoding = face_recognition.face_encodings(vk_image)[0]

vyshak_image = face_recognition.load_image_file(address[6])
vyshak_face_encoding = face_recognition.face_encodings(vyshak_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    andy_face_encoding,
    anuj_face_encoding,
    dk_face_encoding,
    faf_face_encoding,
    lomo_face_encoding,
    vk_face_encoding,
    vyshak_face_encoding,
    ]



def extract_frames_per_second(video_path, output_folder):
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
      print("Error opening video!")
      return

  fps = cap.get(cv2.CAP_PROP_FPS)


  frame_count = 0   python -m 
  extracted_frame_count = 0

  while True:
      ret, frame = cap.read()

      if not ret:
          print("Can't receive frame (stream end?). Exiting...")
          break

      if frame_count % int(fps) == 0:
          filename = f"{output_folder}/frame_{extracted_frame_count}.jpg"

          cv2.imwrite(filename, frame)
          extracted_frame_count += 1

      frame_count += 1
  cap.release()
  print(f"Extracted {extracted_frame_count} frames to {output_folder}")




# Example usage
video_path = "C:\Users\iamna\Downloads\Face attendance\Face-attendance-system\Faces\rcbvscsk.mp4"
output_folder = "Frames"
extract_frames_per_second(video_path, output_folder)



'''def get_image_path():
    image_paths = []
    current_folder = os.getcwd()
    current_folder = current_folder + '\\Frames'
    files = [x for x in os.listdir(current_folder)]
    for filename in files:
        if filename.lower().endswith((".jpg")):
            image_path = f"{current_folder}\\{filename}"
            image_paths.append(image_path)
    return image_paths'''



'''image_paths = get_image_path()'''



# Initialize some variables
def face_recognition(image_paths,known_face_encodings,players):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    for path in image_paths:
        # Grab a single frame of video
        frame = face_recognition.load_image_file(path)

        rgb_small_frame = frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.api.face_locations(rgb_small_frame, number_of_times_to_upsample=1, model='hog')
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True in matches:
                     first_match_index = matches.index(True)
                     name = players[first_match_index]

                '''face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = players[best_match_index]'''

                face_names.append(name)

        process_this_frame = not process_this_frame
    return set(face_names)


    face_recognition(address,known_face_encodings,players)