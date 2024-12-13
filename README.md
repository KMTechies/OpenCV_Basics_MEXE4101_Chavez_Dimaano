# OpenCV_Basics_MEXE4101_Chavez_Dimaano

# Use Haar Cascade Face Detection Code to Identify and Highlight BINI Members in a Crowd
## Introduction

  Face detection is one of the key applications of computer vision that allows machines to identify and locate human faces in images or video streams. This technology has been increasingly popular due to its application in security, social media, and human-computer interaction fields. Among the tools available for implementing face detection, OpenCV stands out as a robust and versatile library.
  By combining OpenCV with datasets like BINI Members, developers can create sophisticated applications that leverage advanced face detection capabilities across multiple domains. This synergy not only enhances operational efficiencies but also improves user experiences by offering tailored solutions that meet specific needs in real time.

## Abstract

  OpenCV provides several methods for face detection, with the Haar Cascade Classifier being one of the most popular techniques. This was first proposed by Viola and Jones in their original paper on rapid object detection, which uses a cascade of classifiers to detect features associated with faces. The Haar Cascade Classifier is pre-trained on large datasets of positive (images containing faces) and negative (images without faces) samples, hence it is able to correctly classify facial and non-facial regions in images.

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
### Findings
### Challenges
### Outcome

## Additional Materials

  * The code is provided in this repository: Face_Detection_Chavez_Dimaano.ipynb
    
  * The recorded demonstration of the model is in this link: https://drive.google.com/file/d/1qikTxu705-dBjlFanf15D29DM0_uA4UY/view?usp=sharing
    
  * The following images are the result of this project:

  **Face Detection Using Haar Cascade Classifier**
  ![image](https://github.com/user-attachments/assets/f8dd6187-d2af-4a13-bbd9-bbd2e2f30e38)

  **Face Recognition of Bini Members**
  ![image](https://github.com/user-attachments/assets/00b4158f-c166-478e-8230-e027262b882d)

  **Face Detection of Bini Members and Fans**
  ![image](https://github.com/user-attachments/assets/d4140cb8-d331-41f9-a069-a0075a5c4860)



















  
