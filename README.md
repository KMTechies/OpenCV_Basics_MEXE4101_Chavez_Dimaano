# OpenCV_Basics_MEXE4101_Chavez_Dimaano

# Use Haar Cascade Face Detection Code to Identify and Highlight BINI Members in a Crowd
## Introduction

  Face detection is one of the key applications of computer vision that allows machines to identify and locate human faces in images or video streams. This technology has been increasingly popular due to its application in security, social media, and human-computer interaction fields. Among the tools available for implementing face detection, OpenCV stands out as a robust and versatile library.
  
  By combining OpenCV with datasets like BINI Members, developers can create sophisticated applications that leverage advanced face detection capabilities across multiple domains. This synergy not only enhances operational efficiencies but also improves user experiences by offering tailored solutions that meet specific needs in real time.

## Abstract

  OpenCV provides several methods for face detection, with the Haar Cascade Classifier being one of the most popular techniques. This was first proposed by Viola and Jones in their original paper on rapid object detection, which uses a cascade of classifiers to detect features associated with faces. The Haar Cascade Classifier is pre-trained on large datasets of positive (images containing faces) and negative (images without faces) samples, hence it is able to correctly classify facial and non-facial regions in images.

  When using the Haar Cascade Classifier in OpenCV, it is essential to ensure that the classifier is trained on a comprehensive dataset like those provided by Dataset BINI members. This training allows the classifier to generalize better across different scenarios and improve its performance in real-world applications.

## Methodology
This is the step-by-step methodology for creating a face detection and recognition of BINI members using OpenCV with Haar cascades in Google Colab:

### Project Methodology
1. **Set Up the Environment:**
   - Install necessary libraries: OpenCV, NumPy, and other required Python packages in Google Colab.
   - Import required modules: `cv2` for OpenCV and other libraries for file handling and visualization.

2. **Load the Haar Cascade Model:**
   - Download the Haar cascade XML file for face detection (e.g., `haarcascade_frontalface_default.xml`) from the OpenCV repository.
   - Load the model using `cv2.CascadeClassifier`.

3. **Load and Preprocess Images:**
   - Upload an image or capture video from the webcam using OpenCV.
   - Convert the image to grayscale using `cv2.cvtColor` for better detection accuracy.

4. **Detect Faces:**
   - Use the `detectMultiScale` method from the loaded Haar cascade classifier to detect faces in the image or video frames.
   - Tweak parameters like `scaleFactor` and `minNeighbors` for optimal detection.

5. **Draw Bounding Boxes:**
   - Draw rectangles around detected faces using `cv2.rectangle` for visualization.

6. **Recognize Faces (Optional):**
   - Integrate a face recognition step by adding a trained classifier (e.g., k-Nearest Neighbors or a deep learning-based model) for identifying detected faces.
   - Prepare a dataset of labeled faces and perform training.

7. **Display Results:**
   - Use OpenCV to display the results, showing detected faces with bounding boxes (and labels if recognition is implemented).
   - Provide options to save the output as an image or video.

8. **Testing and Debugging:**
   - Test the system with various images and video inputs.
   - Fine-tune detection parameters and handle edge cases (e.g., partial faces or multiple faces in an image).

9. **Documentation and Repository Setup:**
   - Document the code with comments and explanations for each step.
   - Add a detailed `README.md` file describing the project, setup instructions, and usage.
   - Upload all necessary files (e.g., script, XML models, sample data) to a GitHub repository.

### Example Projects
For inspiration and practical examples we have checked out similar repositories such as: 
- [OpenCV Haar Cascade Example](https://github.com/akshaykalson/face_detection_using_haarCascades).
- [Face Detection with k-NN and Haar Cascade](https://github.com/Shag0r/OpenCV-Face-Recognition-with-Haar-Cascade-and-k-NN). 

## Conclusion

During this project on face detection and recognition using OpenCV's Haar Cascade in Google Colab, we gained valuable insights into implementing computer vision algorithms. Here's a summary of the findings, challenges, and outcomes:

**Findings**:  
- Haar Cascades are efficient for face detection and work well in controlled environments with proper lighting.  
- Converting images to grayscale improves detection accuracy, as it reduces the computational complexity by focusing on intensity rather than color.  
- Parameters like `scaleFactor` and `minNeighbors` significantly influence detection performance and need to be carefully adjusted for different datasets.  

**Challenges**:  
- **Model Sensitivity**: Haar Cascade struggles with edge cases like partially occluded faces, tilted head positions, or low-resolution images.  
- **Integration Issues**: Incorporating a face recognition step proved to be challenging due to limited computational resources on Colab and the need for additional datasets for training.  
- **Runtime Errors**: Handling runtime errors, such as missing or empty images (`!_src.empty()`), was necessary for robust execution.  

**Outcomes**:  
- Successfully implemented a basic face detection pipeline that detects multiple faces in images and videos.  
- Enhanced understanding of OpenCV's tools and Google Colab's capabilities for machine learning projects.  
- Documented the methodology and provided a repository for future development and collaboration.  

This project not only highlighted the strengths of OpenCV's Haar Cascade but also its limitations, which can be addressed by integrating advanced techniques like deep learning in future iterations.

## Additional Materials
### Codes

### Step 1: Clone OpenCV Tutorial Repository
Clone the repository containing the OpenCV tutorial files.
```bash
!git clone https://github.com/misbah4064/opencvTutorial.git
```

### Step 2: Setup Environment
Clear the outputs and import necessary libraries.
```python
from IPython.display import clear_output
clear_output()

import cv2
from google.colab.patches import cv2_imshow  # For displaying images in Colab
```

### Step 3: Face Detection Using Haar Cascade
Load the pre-trained Haar Cascade Classifier for face detection, process the image, and draw rectangles around detected faces.
```python
# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")

# Read an image and convert to grayscale
img = cv2.imread("/content/3b962895-243d-4d9b-b9d8-85a98dacf4f0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces and draw rectangles
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Display the result
cv2_imshow(img)
```

### Step 4: Clone and Install Face Recognition Dependencies
Clone the repository and install the necessary library.
```bash
!git clone https://github.com/misbah4064/face_recognition.git
!pip install face_recognition
```

### Step 5: Face Recognition Using Pre-Encoded Profiles
Create encodings for known faces, detect unknown faces in an image, and label them accordingly.
```python
# Import libraries
import face_recognition
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# Create encoding profiles for known faces
face_1 = face_recognition.load_image_file("/content/Colet.PNG")
face_1_encoding = face_recognition.face_encodings(face_1)[0]
# Repeat for other faces...

known_face_encodings = [
    face_1_encoding,  # Add all other encodings here
]

known_face_names = [
    "BINI Colet",  # Add corresponding names here
]

# Recognize faces in an unknown image
file_name = "/content/photocarts.jpeg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Label detected faces
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Fan"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(unknown_image_to_draw, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

# Display the annotated image
cv2_imshow(unknown_image_to_draw)
```

  * The code is provided in this repository: Face_Detection_Chavez_Dimaano.ipynb
    
  * The recorded demonstration of the model is in this link: https://drive.google.com/file/d/1qikTxu705-dBjlFanf15D29DM0_uA4UY/view?usp=sharing
    
  * The following images are the result of this project:

  **Face Detection Using Haar Cascade Classifier**
  ![image](https://github.com/user-attachments/assets/f8dd6187-d2af-4a13-bbd9-bbd2e2f30e38)

  **Face Recognition of Bini Members**
  ![image](https://github.com/user-attachments/assets/00b4158f-c166-478e-8230-e027262b882d)

  **Face Detection of Bini Members and Fans**
  ![image](https://github.com/user-attachments/assets/d4140cb8-d331-41f9-a069-a0075a5c4860)



















  
