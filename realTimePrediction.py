# import cv2
# from tensorflow.keras.models import load_model
# import numpy as np
# from cvzone.HandTrackingModule import HandDetector

# # Parameters
# model = load_model("gesture_model.h5")  # Load trained model
# categories = [chr(i) for i in range(65, 91)]  # Dynamically generate "A" to "Z"
# cap = cv2.VideoCapture(0)  # Start the webcam
# detector = HandDetector(maxHands=1)  # Initialize hand detector
# offset = 20  # Offset for cropping
# img_size = 150  # Ensure this matches the training size
# sequence_length = 5  # Sequence length used during training
# sequence = []  # Store frames for prediction

# print("Starting real-time hand gesture recognition...")
# while True:
#     try:
#         success, img = cap.read()  # Capture a frame from the webcam
#         if not success:
#             print("Failed to capture video frame.")
#             continue

#         hands, img = detector.findHands(img)  # Detect hands in the frame
#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']  # Bounding box of the hand
#             imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255  # Create a white image

#             # Crop and resize the hand region
#             y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
#             x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
#             imgCrop = img[y1:y2, x1:x2]

#             aspect_ratio = h / w
#             try:
#                 if aspect_ratio > 1:
#                     k = img_size / h
#                     wCal = int(k * w)
#                     imgResize = cv2.resize(imgCrop, (wCal, img_size))
#                     wGap = (img_size - wCal) // 2
#                     imgWhite[:, wGap:wCal + wGap] = imgResize
#                 else:
#                     k = img_size / w
#                     hCal = int(k * h)
#                     imgResize = cv2.resize(imgCrop, (img_size, hCal))
#                     hGap = (img_size - hCal) // 2
#                     imgWhite[hGap:hCal + hGap, :] = imgResize
#             except Exception as e:
#                 print(f"Error during resizing: {e}")
#                 continue

#             # Normalize and add the processed frame to the sequence
#             sequence.append(imgWhite / 255.0)  # Normalize to match training input
#             if len(sequence) > sequence_length:
#                 sequence.pop(0)  # Ensure sequence length matches the training requirement

#             # Make predictions when the sequence is ready
#             if len(sequence) == sequence_length:
#                 sequence_array = np.expand_dims(sequence, axis=0)  # Shape: (1, 5, 150, 150, 3)
#                 print("Input Sequence Shape:", sequence_array.shape)  # Debug input shape

#                 prediction = model.predict(sequence_array)  # Get model predictions
#                 predicted_index = np.argmax(prediction)

#                 if prediction.shape[1] == len(categories):  # Ensure the model's output matches categories
#                     predicted_label = categories[predicted_index]  # Map to the corresponding category
#                     print("Predicted Gesture:", predicted_label)

#                     # Display the prediction on the frame
#                     cv2.putText(img, f"Gesture: {predicted_label}", (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 else:
#                     print(f"Error: Model output shape {prediction.shape[1]} does not match categories.")
#             else:
#                 print("Sequence not ready for prediction.")
#         else:
#             print("No hands detected.")

#         # Display the webcam feed
#         cv2.imshow("Image", img)
#         if cv2.waitKey(1) == ord("q"):  # Press 'q' to quit
#             break
#     except KeyboardInterrupt:
#         print("Exiting program...")
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Parameters
model = load_model("best_gesture_model.h5")  # Load the best trained model
categories = [chr(i) for i in range(65, 91)]  # "A" to "Z"
cap = cv2.VideoCapture(0)  # Start the webcam
detector = HandDetector(maxHands=1)  # Initialize hand detector
offset = 20  # Offset for cropping
img_size = 150  # Ensure this matches the training size
sequence_length = 5  # Sequence length used during training
sequence = []  # Store frames for prediction

print("Starting real-time hand gesture recognition...")
try:
    while True:
        success, img = cap.read()  # Capture a frame from the webcam
        if not success:
            print("Failed to capture video frame.")
            continue

        hands, img = detector.findHands(img)  # Detect hands in the frame
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']  # Bounding box of the hand
            imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255  # Create a white image

            # Crop and resize the hand region
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            aspect_ratio = h / w
            try:
                if aspect_ratio > 1:
                    k = img_size / h
                    wCal = int(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, img_size))
                    wGap = (img_size - wCal) // 2
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = img_size / w
                    hCal = int(k * h)
                    imgResize = cv2.resize(imgCrop, (img_size, hCal))
                    hGap = (img_size - hCal) // 2
                    imgWhite[hGap:hCal + hGap, :] = imgResize
            except Exception as e:
                print(f"Error during resizing: {e}")
                continue

            # Normalize and add the processed frame to the sequence
            imgWhite = imgWhite.astype('float32') / 255.0  # Normalize
            sequence.append(imgWhite)
            if len(sequence) > sequence_length:
                sequence.pop(0)  # Ensure sequence length matches the training requirement

            # Make predictions when the sequence is ready
            if len(sequence) == sequence_length:
                sequence_array = np.expand_dims(sequence, axis=0)  # Shape: (1, 5, 150, 150, 3)
                # Debug input shape if needed
                # print("Input Sequence Shape:", sequence_array.shape)  

                prediction = model.predict(sequence_array)  # Get model predictions
                predicted_index = np.argmax(prediction)

                if prediction.shape[1] == len(categories):  # Ensure the model's output matches categories
                    predicted_label = categories[predicted_index]  # Map to the corresponding category
                    confidence = prediction[0][predicted_index]
                    # print(f"Predicted Gesture: {predicted_label} (Confidence: {confidence:.2f})")

                    # Display the prediction on the frame
                    cv2.putText(img, f"{predicted_label} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print(f"Error: Model output shape {prediction.shape[1]} does not match categories.")
            # Optionally, remove the else clause to reduce console output
            # else:
            #     pass  # Sequence not ready for prediction
        # Optionally, remove this else clause to reduce console output
        # else:
        #     pass  # No hands detected

        # Display the webcam feed
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):  # Press 'q' to quit
            break
except KeyboardInterrupt:
    print("Exiting program...")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
