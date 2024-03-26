#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2
import numpy as np
import face_recognition
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[3]:


players = ['Andy', 'anuj','dinesh karthik','faf du plesis','mahipal','KING','vyshak']
address = ['andy.jpg','anuj.jpg','dk.jpg','faf.jpg','lomo.jpg','vk.jpg','vyshak.jpg']


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


# In[4]:


def extract_frames_per_second(video_path, output_folder):
  # Open the video capture object
  cap = cv2.VideoCapture(video_path)

  # Check if video opened successfully
  if not cap.isOpened():
      print("Error opening video!")
      return

  # Get frame rate (FPS)
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Define frame count variable and counter for extracted frames
  frame_count = 0
  extracted_frame_count = 0

  while True:
      ret, frame = cap.read()

      # Check if frame is read correctly
      if not ret:
          print("Can't receive frame (stream end?). Exiting...")
          break

      # Extract frame every 1/fps seconds (assuming constant FPS)
      if frame_count % int(fps) == 0:
          # Create filename with frame number
          filename = f"{output_folder}/frame_{extracted_frame_count}.jpg"

          # Save the frame
          cv2.imwrite(filename, frame)
          extracted_frame_count += 1

      frame_count += 1

  # When everything done, release the capture object
  cap.release()
  print(f"Extracted {extracted_frame_count} frames to {output_folder}")


# In[5]:


# Example usage
video_path = "rcbvscsk.mp4"
output_folder = "Frames"
extract_frames_per_second(video_path, output_folder)


# In[6]:


def get_image_path():
    image_paths = []
    current_folder = os.getcwd()
    current_folder = current_folder + '\\Frames'
    files = [x for x in os.listdir(current_folder)]
    for filename in files:
        if filename.lower().endswith((".jpg")):
            image_path = f"{current_folder}\\{filename}"
            image_paths.append(image_path)
    return image_paths


# In[7]:


image_paths = get_image_path()


# In[30]:


get_ipython().run_cell_magic('time', '', "# Initialize some variables\ndef face_recognition(image_paths,known_face_encodings,players):\n    face_locations = []\n    face_encodings = []\n    face_names = []\n    process_this_frame = True\n\n    for path in image_paths:\n        # Grab a single frame of video\n        frame = face_recognition.load_image_file(path)\n\n        rgb_small_frame = frame[:, :, ::-1]\n\n        if process_this_frame:\n            face_locations = face_recognition.api.face_locations(rgb_small_frame, number_of_times_to_upsample=1, model='hog')\n            face_encodings = face_recognition.face_encodings(frame, face_locations)\n\n            for face_encoding in face_encodings:\n                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)\n\n                if True in matches:\n                     first_match_index = matches.index(True)\n                     name = players[first_match_index]\n\n                '''face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)\n                best_match_index = np.argmin(face_distances)\n                if matches[best_match_index]:\n                    name = players[best_match_index]'''\n\n                face_names.append(name)\n\n        process_this_frame = not process_this_frame\n    return set(face_names)")

