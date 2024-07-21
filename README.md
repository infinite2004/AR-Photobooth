# AR-Photobooth
the script allows for switching between different glasses styles using hand gestures and displays a QR code on the video feed.


This Python script utilizes OpenCV and MediaPipe libraries to implement an augmented reality (AR) application that overlays virtual glasses on faces detected in the webcam feed. Additionally, the script allows for switching between different glasses styles using hand gestures and displays a QR code on the video feed.

Summary:

	1.Initialization:
	  • Import necessary libraries: OpenCV for video processing, MediaPipe for face and hand detection, and NumPy for numerical operations.
	  •	Initialize MediaPipe modules for face detection and hand tracking.
	  •	Set up the webcam to capture video input.
	2.	Loading Resources:
	  •	Load various styles of glasses images with predefined adjustments.
	  •	Load a QR code image.
	3.	Helper Functions:
	  •	overlay_image(): Overlays an image onto the video frame at specified coordinates, accounting for transparency.
	  •	overlay_glasses(): Overlays the selected glasses image on detected faces, adjusting for position and scale.
	4.	Main Loop:
	  •	Continuously capture frames from the webcam.
	  •	Convert each frame to RGB for processing.
	  •	Detect faces and hands in the frame.
	  •	If a face is detected, overlay the selected glasses image onto the face.
	  •	If a hand gesture (all fingers open) is detected, cycle through the available glasses styles.
	  •	Draw hand landmarks for visual feedback.
  	•	Overlay the QR code on the frame.
  	•	Display the processed frame in a window.
  	•	Exit the loop when the ‘q’ key is pressed.

How to Run the Code Step-by-Step:

	  1.Install Dependencies:
        Ensure you have Python installed and install the necessary libraries using pip:

        pip install opencv-python mediapipe numpy

    2.Prepare Resources:
	    •	Save the script to a Python file, e.g., ar_glasses.py.
	    •	Ensure the glasses images (glasses.png, circle_glasses.png, flower_glasses.png) and the QR code image (qr_code.png) are in the same directory as the script.
	  
    3.Run the Script:
      Open a terminal or command prompt, navigate to the directory containing the script and resources, and execute the script:
     
      python ___filename___.py
      
    4.Interact with the Application:
    	•	The webcam feed will open in a new window.
    	•	If a face is detected, virtual glasses will be overlaid on the face.
    	•	To switch glasses styles, open your hand (thumb and all fingers extended) in front of the camera.
    	•	To exit the application, press the ‘q’ key.
      
